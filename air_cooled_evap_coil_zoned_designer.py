
# -*- coding: utf-8 -*-
"""
DX Evaporator (Air-side wet coil) — Zoned ε–NTU (Single-file Streamlit App)

Key modeling choices (kept intentionally simple/robust for early-stage sizing):
- Air enters the SUPERHEAT rows first (maximum ΔT for superheat)
- Then air crosses the EVAPORATION (wet) rows
- Wet rows use an enthalpy-bypass-factor formulation:
    BF = exp(-NTU_h) with NTU_h = UA/(m_da * cp_ref)
    Leaving state is a mix of inlet state and saturated state at coil surface temperature (≈ Tsat):
        h_out = BF*h_in + (1-BF)*h_sat(Ts)
        W_out = BF*W_in + (1-BF)*W_sat(Ts)
    Then DB_out is solved from h_out and W_out.
- Superheat rows are treated as "dry" sensible transfer to a mean refrigerant temperature.
- Refrigerant Δp uses Darcy–Weisbach with correct velocity v = G/ρ (mass flux / density).
"""
import math
from math import pi, sqrt
import traceback
import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="DX Evaporator — Wet Coil ε–NTU (Single File)", layout="wide")

# CoolProp
try:
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except Exception:
    HAS_COOLPROP = False

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ------------------------- Constants & Psychrometrics -------------------------
INCH = 0.0254
MM = 1e-3
P_ATM = 101325.0
R_DA = 287.055
CP_DA = 1006.0
CP_V = 1860.0
H_LV0 = 2501000.0

def K(tC): return tC + 273.15

def psat_water_Pa(T_C: float) -> float:
    # Buck/Tetens-like (good for HVAC range)
    return 611.21 * math.exp((18.678 - T_C/234.5) * (T_C/(257.14 + T_C)))

def humidity_ratio_from_T_RH(T_C: float, RH_pct: float, P: float = P_ATM) -> float:
    RH = max(min(RH_pct, 100.0), 0.1) / 100.0
    Psat = psat_water_Pa(T_C)
    Pv = RH * Psat
    return 0.62198 * Pv / max(P - Pv, 1.0)

def humidity_ratio_from_T_WB(Tdb_C: float, Twb_C: float, P: float = P_ATM) -> float:
    W_sat_wb = humidity_ratio_from_T_RH(Twb_C, 100.0, P)
    cp_da = 1006.0
    cp_wv = 1860.0
    h_fg_wb = 2501000.0 - 2369.0 * Twb_C
    numer = (W_sat_wb * (h_fg_wb + cp_wv * Twb_C) - cp_da * (Tdb_C - Twb_C))
    denom = (h_fg_wb + cp_wv * Tdb_C)
    W = numer / max(1e-9, denom)
    return max(0.0, W)

def moist_air_enthalpy_J_per_kg_da(T_C: float, W: float) -> float:
    return 1000.0*1.006*T_C + W*(H_LV0 + 1000.0*1.86*T_C)

def cp_moist_air_J_per_kgK(T_C: float, W: float) -> float:
    return CP_DA + W*CP_V

def rho_moist_air_kg_m3(T_C: float, W: float, P: float = P_ATM) -> float:
    T_K = K(T_C)
    return P / (R_DA * T_K * (1.0 + 1.6078*W))

def RH_from_T_W(T_C, W, P=P_ATM):
    Pv = W*P/(0.62198 + W)
    Ps = psat_water_Pa(T_C)
    RH = 100.0*Pv/max(Ps,1e-9)
    return max(0.1, min(100.0, RH))

def solve_T_from_h_W(h_J_per_kg_da: float, W: float) -> float:
    # h = 1000*(1.006*T) + W*(H_LV0 + 1000*1.86*T)
    denom = 1000.0*(1.006 + 1.86*W)
    T = (h_J_per_kg_da - W*H_LV0) / max(1e-12, denom)
    return T

def wb_from_T_W(Tdb_C: float, W: float) -> float:
    # Solve Twb so that humidity_ratio_from_T_WB(Tdb, Twb) = W
    lo, hi = -30.0, min(Tdb_C, 60.0)
    for _ in range(60):
        mid = 0.5*(lo+hi)
        W_mid = humidity_ratio_from_T_WB(Tdb_C, mid)
        if W_mid > W:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

# ------------------------- Heat-transfer Utilities ----------------------------
def fin_efficiency_infinite_plate(h, k_fin, t_fin, Lc):
    if Lc <= 0 or t_fin <= 0 or k_fin <= 0:
        return 1.0
    m = sqrt(2.0*h/(k_fin*t_fin))
    x = max(m*Lc, 1e-9)
    return math.tanh(x)/x

def geometry_areas(W, H, Nr, St, Do, tf, FPI):
    face_area = W*H
    fins_count = int(round(FPI * (H/INCH)))
    N_tpr = max(int(math.floor(H / max(St,1e-9))), 1)   # tubes per row
    N_tubes = N_tpr * Nr
    L_tube = W
    s = (1.0/FPI) * INCH
    A_holes_one_fin = N_tubes * (pi*(Do/2)**2)
    A_fin_one = max(2.0*(W*H - A_holes_one_fin), 0.0)   # both sides
    A_fin_total = A_fin_one * fins_count
    exposed_frac = max((s - tf)/max(s,1e-9), 0.0)
    A_bare = N_tubes * (pi * Do * L_tube) * exposed_frac
    A_total = A_fin_total + A_bare

    fin_blockage = min(tf/max(s,1e-9), 0.95)
    tube_blockage = min(A_holes_one_fin/max(W*H,1e-9), 0.5)
    A_min = max(face_area * (1.0 - fin_blockage - tube_blockage), 1e-4)

    return dict(face_area=face_area, fins=fins_count, s=s,
                N_tpr=N_tpr, N_tubes=N_tubes, L_tube=L_tube,
                A_fin=A_fin_total, A_bare=A_bare, A_total=A_total, A_min=A_min)

def zukauskas_constants(Re):
    Re = max(Re, 1.0)
    if 1e2 <= Re < 1e3:   C, m = 0.9, 0.4
    elif 1e3 <= Re < 2e5: C, m = 0.27, 0.63
    else:                 C, m = (0.27, 0.63) if Re >= 2e5 else (0.9, 0.4)
    return C, m

def row_correction(Nr):
    return 0.70 if Nr<=1 else (0.80 if Nr==2 else (0.88 if Nr==3 else (0.94 if Nr==4 else 1.00)))

def air_htc_zukauskas(rho, mu, k, Pr, Do, Nr, mdot_air, A_min):
    Vmax = mdot_air/(rho*A_min)
    Re = rho*Vmax*Do/max(mu,1e-12)
    C, m = zukauskas_constants(Re)
    Nu = C*(Re**m)*(Pr**0.36) * row_correction(Nr)
    h = Nu * k / Do
    meta = dict(Vmax=Vmax, Re=Re, Nu=Nu)
    return h, meta

def air_dp_slot_model(rho, mu, v_face, geom, Nr, St, tf, K_in=0.5, K_out=1.0):
    fins_per_m = (1.0/geom['s'])
    fin_pitch = 1.0 / fins_per_m
    s_fin = max(1e-5, fin_pitch - tf)
    D_h = 2.0*s_fin
    v_core = v_face * (geom['face_area']/geom['A_min'])
    Re_ch = rho * v_core * D_h / max(1e-12, mu)
    q = 0.5 * rho * v_core*v_core
    if Re_ch < 2300:
        f_D = 64.0 / max(1.0, Re_ch)
    else:
        f_D = 0.3164 / (Re_ch ** 0.25)
    L_flow = Nr * St
    dp_fric = f_D * (L_flow / D_h) * q
    dp_minor = (K_in + K_out) * q
    return dp_fric + dp_minor, dict(D_h=D_h, Re=Re_ch, v_core=v_core, f_D=f_D, L=L_flow)

def f_churchill(Re, e_over_D):
    Re = max(1e-9, Re)
    if Re < 2300.0:
        return 64.0 / max(1.0, Re)
    A = (2.457 * math.log( (7.0 / max(1.0, Re))**0.9 + 0.27*e_over_D ))**16
    B = (37530.0 / max(1.0, Re))**16
    f = 8.0 * ( ( (8.0 / max(1.0, Re))**12 ) + 1.0 / ( (A + B)**1.5 ) )**(1.0/12.0)
    return max(1e-6, f)

def dp_darcy(G, rho, mu, D, L, roughness=1.5e-6):
    # Darcy–Weisbach using mass flux
    Re = max(1e-9, G*D/max(1e-12, mu))
    f = f_churchill(Re, roughness/max(1e-12, D))
    dp = f * (L/max(1e-12, D)) * (G**2) / (2.0*max(1e-9, rho))
    v = G / max(1e-9, rho)  # CORRECT: v = G / rho (m/s)
    return dp, Re, v, f

# ------------------------- Design Core ----------------------------
def design_evaporator(
    face_W_m, face_H_m, St_m, Nr,
    Do_m, t_wall_m, FPI, tf_m, fin_k, tube_k,
    Vdot_m3_s,
    Tdb_in_C, RH_in_pct, use_wb_in, Twb_in_C,
    Tdb_out_req_C, RH_out_req_pct, use_wb_out, Twb_out_req_C,
    fluid, T_sat_evap_C, SH_req_K, mdot_ref_total,
    tube_circuits, wet_enhance, Rf_o, Rf_i,
):
    # --- Air inlet/outlet targets (for "required")
    if use_wb_in:
        W_in = humidity_ratio_from_T_WB(Tdb_in_C, Twb_in_C)
    else:
        W_in = humidity_ratio_from_T_RH(Tdb_in_C, RH_in_pct)

    if use_wb_out:
        W_out_req = humidity_ratio_from_T_WB(Tdb_out_req_C, Twb_out_req_C)
    else:
        W_out_req = humidity_ratio_from_T_RH(Tdb_out_req_C, RH_out_req_pct)

    h_in = moist_air_enthalpy_J_per_kg_da(Tdb_in_C, W_in)
    h_out_req = moist_air_enthalpy_J_per_kg_da(Tdb_out_req_C, W_out_req)

    rho_in = rho_moist_air_kg_m3(Tdb_in_C, W_in)
    mdot_da = (rho_in * Vdot_m3_s) / (1.0 + W_in)  # kg_dry/s
    mdot_air_total = mdot_da * (1.0 + W_in)

    Q_required_W = mdot_da * (h_in - h_out_req)
    cp_air_in = cp_moist_air_J_per_kgK(Tdb_in_C, W_in)
    Q_sensible_required_W = mdot_da * cp_air_in * (Tdb_in_C - Tdb_out_req_C)

    # --- Geometry
    geom = geometry_areas(face_W_m, face_H_m, Nr, St_m, Do_m, tf_m, FPI)
    A_face = geom['face_area']; Ao = geom['A_total']; A_min = geom['A_min']
    area_per_row = Ao / max(1, Nr)
    tubes_per_row = geom['N_tpr']
    L_tube = geom['L_tube']
    L_total_per_circuit = (tubes_per_row * Nr / max(1, tube_circuits)) * L_tube

    # --- Air properties (use inlet for correlations)
    mu_air = 1.716e-5 * ((K(Tdb_in_C)/273.15)**1.5) * ((273.15+110.4)/(K(Tdb_in_C)+110.4))
    k_air  = 0.026  # keep simple & stable
    Pr_air = cp_air_in * mu_air / max(1e-9, k_air)

    # Air-side h
    h_air_dry, meta_htc = air_htc_zukauskas(rho_in*(1+W_in), mu_air, k_air, Pr_air, Do_m, Nr, mdot_air_total, A_min)
    h_air_wet = h_air_dry * wet_enhance  # wet enhancement applied to external h

    # Fin efficiency based on wet h (more conservative)
    Lc = max(0.5*(St_m - Do_m), 1e-6)
    eta_f = fin_efficiency_infinite_plate(h_air_wet, fin_k, tf_m, Lc)
    eta_o_wet = 1.0 - (geom['A_fin']/max(Ao,1e-9))*(1.0 - eta_f)

    # --- Refrigerant properties at Tsat
    if not HAS_COOLPROP:
        raise RuntimeError("CoolProp is required. Add 'CoolProp>=6.6' to requirements.txt and redeploy.")
    T_sat_K = K(T_sat_evap_C)
    P_sat_Pa = PropsSI("P","T",T_sat_K,"Q",0,fluid)
    rho_l = PropsSI("D","T",T_sat_K,"Q",0,fluid)
    rho_v = PropsSI("D","T",T_sat_K,"Q",1,fluid)
    mu_l  = PropsSI("V","T",T_sat_K,"Q",0,fluid)
    mu_v  = PropsSI("V","T",T_sat_K,"Q",1,fluid)
    cp_v  = PropsSI("C","T",T_sat_K,"Q",1,fluid)
    k_v   = PropsSI("L","T",T_sat_K,"Q",1,fluid)

    # Tube inside
    Di = max(Do_m - 2.0*t_wall_m, 1e-5)
    Ai = pi*(Di**2)/4.0
    mdot_ref_total = max(1e-9, mdot_ref_total)
    mdot_per_circ = mdot_ref_total / max(1, tube_circuits)
    G = mdot_per_circ / max(1e-12, Ai)  # kg/s/m2 (mass flux)

    # Inside vapor HTC (superheat region)
    def h_i_vapor(mu, k, cp, D, G):
        Re = max(1e-9, G*D/max(1e-12, mu))
        Pr = cp*mu/max(1e-9, k)
        Nu = 0.023 * (max(2300.0, Re)**0.8) * (Pr**0.4)
        h = Nu * k / max(1e-9, D)
        return h, Re, Pr

    h_i_v, Re_v, Pr_v = h_i_vapor(mu_v, k_v, cp_v, Di, G)

    # Tube wall + area ratio for Uo
    A_o_per_m = pi * Do_m
    A_i_per_m = pi * Di
    Ao_Ai = A_o_per_m / max(1e-12, A_i_per_m)
    R_wall_per_Ao = (math.log(Do_m/max(1e-12, Di)) / (2.0*pi*tube_k)) / max(1e-12, A_o_per_m)

    # Uo helper
    def Uo_from(h_i, h_o, eta_o):
        invU = (1.0/max(1e-9, eta_o*h_o)) + Rf_o + Ao_Ai*((1.0/max(1e-9, h_i)) + Rf_i) + R_wall_per_Ao
        return 1.0 / invU

    # Uo values
    Uo_SH = Uo_from(h_i_v, h_air_dry, 1.0 - (geom['A_fin']/max(Ao,1e-9))*(1.0 - fin_efficiency_infinite_plate(h_air_dry, fin_k, tf_m, Lc)))
    # For evap zone we don't trust an inside HTC from simple correlations; use a conservative high value (boiling)
    h_i_boil = max(1500.0, 2.0*h_i_v)
    Uo_EV = Uo_from(h_i_boil, h_air_wet, eta_o_wet)

    # --- Allocate rows: superheat first, then evaporation
    rows_SH = max(1, int(round(0.25 * Nr)))  # start with 25% rows
    rows_EV = max(1, Nr - rows_SH)
    if rows_EV < 1:
        rows_EV = 1
        rows_SH = Nr - rows_EV

    # ---------------- Superheat zone (dry, sensible) ----------------
    T_air_1 = Tdb_in_C
    W_air_1 = W_in
    cp_air_1 = cp_moist_air_J_per_kgK(T_air_1, W_air_1)
    C_air_1 = mdot_da * cp_air_1  # W/K

    # Mean refrigerant temp across SH zone (approx): Tsat -> Tsat+SH
    T_ref_mean_SH = T_sat_evap_C + 0.5*SH_req_K

    UA_SH = Uo_SH * rows_SH * area_per_row
    NTU_SH = UA_SH / max(1e-9, C_air_1)
    # ε for C_r -> 0 (ref side effectively large due to phase change? In SH it isn't, but we use simple C_air-only model)
    eps_SH = 1.0 - math.exp(-NTU_SH)
    Q_SH_cap = eps_SH * C_air_1 * max(0.0, (T_air_1 - T_ref_mean_SH))
    Q_SH_req = mdot_ref_total * cp_v * SH_req_K
    Q_SH = min(Q_SH_req, max(0.0, Q_SH_cap))
    SH_achieved_K = Q_SH / max(1e-12, (mdot_ref_total*cp_v))

    T_air_2 = T_air_1 - Q_SH / max(1e-12, C_air_1)
    # no dehumidification in superheat region
    W_air_2 = W_air_1
    h_air_2 = moist_air_enthalpy_J_per_kg_da(T_air_2, W_air_2)

    # If SH not achieved, steal rows from EVAP (since air enters SH first)
    if SH_achieved_K + 1e-6 < SH_req_K and rows_EV > 1:
        # Move rows one-by-one until SH meets or no evap rows left
        while rows_EV > 1 and SH_achieved_K + 1e-6 < SH_req_K:
            rows_SH += 1
            rows_EV -= 1
            UA_SH = Uo_SH * rows_SH * area_per_row
            NTU_SH = UA_SH / max(1e-9, C_air_1)
            eps_SH = 1.0 - math.exp(-NTU_SH)
            Q_SH_cap = eps_SH * C_air_1 * max(0.0, (T_air_1 - T_ref_mean_SH))
            Q_SH = min(Q_SH_req, max(0.0, Q_SH_cap))
            SH_achieved_K = Q_SH / max(1e-12, (mdot_ref_total*cp_v))
            T_air_2 = T_air_1 - Q_SH / max(1e-12, C_air_1)
            W_air_2 = W_air_1
            h_air_2 = moist_air_enthalpy_J_per_kg_da(T_air_2, W_air_2)

    # ---------------- Evap zone (wet, enthalpy NTU / BF) ----------------
    # Coil surface temperature ~ Tsat (for non-frost regime)
    T_surf = T_sat_evap_C

    W_sat_surf = humidity_ratio_from_T_RH(T_surf, 100.0)
    h_sat_surf = moist_air_enthalpy_J_per_kg_da(T_surf, W_sat_surf)

    UA_EV = Uo_EV * rows_EV * area_per_row

    # Enthalpy bypass factor (Lewis≈1, use cp_ref to nondimensionalize)
    cp_ref = 1006.0
    NTU_h = UA_EV / max(1e-12, (mdot_da * cp_ref))
    BF = math.exp(-NTU_h)

    h_out = BF*moist_air_enthalpy_J_per_kg_da(T_air_2, W_air_2) + (1.0-BF)*h_sat_surf
    W_out = BF*W_air_2 + (1.0-BF)*W_sat_surf

    T_air_out = solve_T_from_h_W(h_out, W_out)
    # Physical clamps
    T_air_out = max(T_surf, min(T_air_out, T_air_2))  # cannot go below surface temp (non-frost assumption)
    # Recompute h_out after clamp for consistency
    h_out = moist_air_enthalpy_J_per_kg_da(T_air_out, W_out)

    RH_out = RH_from_T_W(T_air_out, W_out)
    Twb_out = wb_from_T_W(T_air_out, W_out)

    # Achieved totals from enthalpy drop across both zones
    h_in_total = h_in
    Q_total_W = mdot_da * (h_in_total - h_out)
    Q_sens_W = mdot_da * cp_air_in * (Tdb_in_C - T_air_out)
    Q_lat_W = max(0.0, Q_total_W - Q_sens_W)

    # --- Airside Δp
    v_face = Vdot_m3_s / max(1e-12, A_face)
    dp_air, dp_meta = air_dp_slot_model(rho_in*(1+W_in), mu_air, v_face, geom, Nr, St_m, tf_m)

    # --- Refrigerant Δp (split by length proportion)
    frac_SH = rows_SH / max(1, Nr)
    frac_EV = rows_EV / max(1, Nr)
    L_SH = frac_SH * L_total_per_circuit
    L_EV = frac_EV * L_total_per_circuit

    dp_SH, Re_SH, vel_SH, f_SH = dp_darcy(G, rho_v, mu_v, Di, L_SH)
    # For 2-phase Δp we use homogeneous density at x~0.5; still crude but stable
    rho_m = 1.0 / (0.5/max(1e-12, rho_v) + 0.5/max(1e-12, rho_l))
    mu_m = 0.5*mu_v + 0.5*mu_l
    dp_EV, Re_EV, vel_EV, f_EV = dp_darcy(G, rho_m, mu_m, Di, L_EV)
    dp_total_kPa = (dp_SH + dp_EV) / 1000.0

    # --- Sufficiency
    cap_ok = Q_total_W + 1e-6 >= Q_required_W
    sh_ok = SH_achieved_K + 1e-6 >= SH_req_K
    insuff = []
    if not cap_ok: insuff.append("Capacity shortfall")
    if not sh_ok: insuff.append("Superheat shortfall")
    insuff_text = ", ".join(insuff) if insuff else "None"

    # Tables
    rows_df = pd.DataFrame([
        ["Superheat (dry, air enters here)", Q_SH/1000.0, rows_SH, UA_SH, NTU_SH, eps_SH, L_SH, Re_SH, f_SH, vel_SH, dp_SH/1000.0],
        ["Evaporation (wet, enthalpy BF)",   (Q_total_W - Q_SH)/1000.0, rows_EV, UA_EV, NTU_h, (1.0-BF), L_EV, Re_EV, f_EV, vel_EV, dp_EV/1000.0],
    ], columns=["Zone","Q (kW)","Rows used","UA","NTU","Effectiveness","L per circuit (m)","Re","f","Vel (m/s)","Δp_zone (kPa)"])

    summary = {
        # Required vs achieved
        "Q_required_kW": Q_required_W/1000.0,
        "Q_sensible_required_kW": Q_sensible_required_W/1000.0,
        "Q_achieved_kW": Q_total_W/1000.0,
        "Q_sensible_achieved_kW": Q_sens_W/1000.0,
        "Q_latent_achieved_kW": Q_lat_W/1000.0,
        # Leaving air achieved
        "Air_out_DB_C": T_air_out,
        "Air_out_WB_C": Twb_out,
        "Air_out_RH_pct": RH_out,
        "Air_out_W_kgkg": W_out,
        # Superheat
        "Superheat_required_K": SH_req_K,
        "Superheat_achieved_K": SH_achieved_K,
        # Rows
        "Rows_entered": Nr,
        "Rows_SH": rows_SH,
        "Rows_EV": rows_EV,
        "Insufficiency": insuff_text,
        # Areas & lengths
        "A_face_m2": A_face,
        "A_total_o_m2": Ao,
        "A_min_m2": A_min,
        "Ao_per_row_m2": area_per_row,
        "Tubes_per_row": tubes_per_row,
        "Tube_length_per_circuit_m": L_total_per_circuit,
        # Airflow
        "Face_velocity_m_s": v_face,
        "Vdot_m3_s": Vdot_m3_s,
        "mdot_dry_air_kg_s": mdot_da,
        "Air_dP_Pa": dp_air,
        # Refrigerant
        "Fluid": fluid,
        "Tsat_C": T_sat_evap_C,
        "Psat_bar": P_sat_Pa/1e5,
        "mdot_ref_total_kg_s": mdot_ref_total,
        "mdot_ref_per_circuit_kg_s": mdot_per_circ,
        "Ref_dP_total_kPa": dp_total_kPa,
        "Vel_SH_m_s": vel_SH,
        "Vel_EV_m_s": vel_EV,
        # Conductivities
        "k_fin_used_W_mK": fin_k,
        "k_tube_used_W_mK": tube_k,
        # Wet-coil anchors
        "T_surface_assumed_C": T_surf,
        "h_sat_surface_kJ_kgda": h_sat_surf/1000.0,
        "W_sat_surface": W_sat_surf,
        "BF_evap": BF,
    }
    return rows_df, summary

# ------------------------- UI ----------------------------
st.title("DX Evaporator — Wet Coil ε–NTU (Single File)")
st.caption("Model: Superheat (dry) first → Evaporation (wet enthalpy BF). Pre-evap zone removed.")

st.header("Geometry & Materials")
colg1, colg2, colg3, colg4 = st.columns(4)
with colg1:
    face_W = st.number_input("Face width W (m)", 0.2, 4.0, 1.2, 0.01, format="%.2f")
with colg2:
    face_H = st.number_input("Face height H (m)", 0.2, 4.0, 1.0, 0.01, format="%.2f")
with colg3:
    St_mm = st.number_input("Row-to-row pitch (mm)", 10.00, 60.00, 22.00, 0.01, format="%.2f")
with colg4:
    _ = st.number_input("Longitudinal pitch (mm) (not used in this simplified model)", 10.00, 60.00, 24.40, 0.01, format="%.2f")

colg5, colg6, colg7, colg8 = st.columns(4)
with colg5:
    Nr = st.number_input("Rows (depth)", 1, 20, 4, 1)
with colg6:
    Do_mm = st.number_input("Tube OD (mm)", 5.00, 20.00, 9.53, 0.01, format="%.2f")
with colg7:
    tw_mm = st.number_input("Tube wall thickness (mm)", 0.20, 2.00, 0.50, 0.01, format="%.2f")
with colg8:
    FPI = st.number_input("Fins per inch (1/in)", 4.0, 24.0, 10.0, 0.5)

colg9, colg10, colg11, colg12 = st.columns(4)
with colg9:
    tf_mm = st.number_input("Fin thickness (mm)", 0.06, 0.30, 0.12, 0.01, format="%.2f")
with colg10:
    fin_mat = st.selectbox("Fin material", ["Aluminum", "Copper", "Steel"])
with colg11:
    tube_mat = st.selectbox("Tube material", ["Copper", "Aluminum", "Steel", "CuNi 90/10"])
with colg12:
    Nr_circuits = st.number_input("Tube circuits (parallel)", 2, 64, 8, 1)

MAT_K = {"Copper": 380.0, "Aluminum": 205.0, "Steel": 50.0, "CuNi 90/10": 29.0}
fin_k = MAT_K[fin_mat]
tube_k = MAT_K[tube_mat]

st.header("Airflow & Inlet/Outlet Air State")
airflow_mode = st.radio("Airflow input mode", ["Face velocity (m/s)", "Volume flow (m³/h)"], horizontal=True)
if airflow_mode == "Face velocity (m/s)":
    v_face_in = st.number_input("Face velocity (m/s)", 0.3, 6.0, 2.5, 0.1, format="%.2f")
    Vdot_m3_s = v_face_in * face_W * face_H
else:
    Vdot_m3_h = st.number_input("Air volume flow (m³/h)", 500.0, 50000.0, 8000.0, 10.0, format="%.1f")
    Vdot_m3_s = Vdot_m3_h / 3600.0

air_in_mode = st.radio("Inlet air condition input", ["DB + RH", "DB + WB"], horizontal=True)
if air_in_mode == "DB + RH":
    Tdb_in = st.number_input("DB in (°C)", 5.0, 55.0, 27.0, 0.1, format="%.1f")
    RH_in = st.number_input("RH in (%)", 5.0, 100.0, 50.0, 0.1, format="%.1f")
    use_wb_in, Twb_in = False, 0.0
else:
    Tdb_in = st.number_input("DB in (°C)", 5.0, 55.0, 27.0, 0.1, format="%.1f")
    Twb_in = st.number_input("WB in (°C)", -5.0, 40.0, 19.0, 0.1, format="%.1f")
    RH_in = 50.0
    use_wb_in = True

air_out_mode = st.radio("Required outlet air input", ["DB + RH", "DB + WB"], horizontal=True)
if air_out_mode == "DB + RH":
    Tdb_out_req = st.number_input("Required DB out (°C)", -5.0, 40.0, 14.0, 0.1, format="%.1f")
    RH_out_req = st.number_input("Required RH out (%)", 5.0, 100.0, 95.0, 0.1, format="%.1f")
    use_wb_out, Twb_out_req = False, 0.0
else:
    Tdb_out_req = st.number_input("Required DB out (°C)", -5.0, 40.0, 14.0, 0.1, format="%.1f")
    Twb_out_req = st.number_input("Required WB out (°C)", -5.0, 40.0, 13.0, 0.1, format="%.1f")
    RH_out_req = 95.0
    use_wb_out = True

st.header("Refrigerant & Operating")
fluid = st.selectbox("Refrigerant (CoolProp name)", ["R134a", "R410A", "R407C", "R404A", "R32", "R22"])
Tsat = st.number_input("Evaporating saturation temperature Tsat (°C)", -20.0, 20.0, 5.0, 0.1, format="%.1f")
SH_req = st.number_input("Required refrigerant superheat at outlet (K)", 0.0, 25.0, 6.0, 0.5, format="%.1f")
wet_enh = st.number_input("Wet enhancement factor (air)", 1.0, 2.5, 1.35, 0.05, format="%.2f")
mdot_ref_total = st.number_input("Total refrigerant mass flow (kg/s)", 0.001, 2.000, 0.080, 0.001, format="%.3f")

st.header("Fouling (optional)")
colf1, colf2 = st.columns(2)
with colf1:
    Rfo = st.number_input("Air-side fouling (m²·K/W)", 0.0, 0.001, 0.0002, 0.00005, format="%.5f")
with colf2:
    Rfi = st.number_input("Tube-side fouling (m²·K/W)", 0.0, 0.001, 0.0001, 0.00005, format="%.5f")

# Convert mm → m
St = St_mm * MM
Do = Do_mm * MM
tw = tw_mm * MM
tf = tf_mm * MM

# Run design
try:
    rows, summary = design_evaporator(
        face_W_m=face_W, face_H_m=face_H, St_m=St, Nr=int(Nr),
        Do_m=Do, t_wall_m=tw, FPI=float(FPI), tf_m=tf, fin_k=fin_k, tube_k=tube_k,
        Vdot_m3_s=Vdot_m3_s,
        Tdb_in_C=Tdb_in, RH_in_pct=RH_in, use_wb_in=use_wb_in, Twb_in_C=Twb_in,
        Tdb_out_req_C=Tdb_out_req, RH_out_req_pct=RH_out_req, use_wb_out=use_wb_out, Twb_out_req_C=Twb_out_req,
        fluid=fluid, T_sat_evap_C=Tsat, SH_req_K=SH_req, mdot_ref_total=mdot_ref_total,
        tube_circuits=int(Nr_circuits), wet_enhance=wet_enh, Rf_o=Rfo, Rf_i=Rfi,
    )
except Exception as e:
    st.error("Design solver error: " + str(e))
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    st.stop()

# Outputs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.subheader("Required (from air target)")
    st.metric("Total required (kW)", f"{summary['Q_required_kW']:.2f}")
    st.metric("Sensible required (kW)", f"{summary['Q_sensible_required_kW']:.2f}")
with c2:
    st.subheader("Achieved by coil")
    st.metric("Total achieved (kW)", f"{summary['Q_achieved_kW']:.2f}")
    st.metric("Sensible achieved (kW)", f"{summary['Q_sensible_achieved_kW']:.2f}")
with c3:
    st.subheader("Latent & superheat")
    st.metric("Latent achieved (kW)", f"{summary['Q_latent_achieved_kW']:.2f}")
    st.metric("Superheat achieved (K)", f"{summary['Superheat_achieved_K']:.2f}")
with c4:
    st.subheader("Leaving air (achieved)")
    st.metric("DB out (°C)", f"{summary['Air_out_DB_C']:.2f}")
    st.metric("WB out (°C)", f"{summary['Air_out_WB_C']:.2f}")
    st.metric("RH out (%)", f"{summary['Air_out_RH_pct']:.1f}")

st.subheader("Sufficiency")
st.info(f"Insufficiency flags: {summary['Insufficiency']}")

st.subheader("Zone table")
st.dataframe(rows.style.format({
    "Q (kW)": "{:.2f}",
    "Rows used": "{:.0f}",
    "UA": "{:.0f}",
    "NTU": "{:.2f}",
    "Effectiveness": "{:.3f}",
    "L per circuit (m)": "{:.2f}",
    "Re": "{:.0f}",
    "f": "{:.4f}",
    "Vel (m/s)": "{:.2f}",
    "Δp_zone (kPa)": "{:.2f}",
}), use_container_width=True)

with st.expander("Intermediate values & diagnostics"):
    diag = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
    st.dataframe(diag, use_container_width=True)

# ------------------------- PDF Export ----------------------------
def build_pdf(inputs_dict, rows_df, summary_dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("DX Evaporator — Wet Coil ε–NTU Report", styles["Title"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Inputs", styles["Heading2"]))
    inp_items = [[k, f"{v}"] for k, v in inputs_dict.items()]
    t = Table([["Parameter", "Value"]] + inp_items, colWidths=[190, 310])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0")),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Zone Results", styles["Heading2"]))
    zone_data = [rows_df.columns.tolist()] + rows_df.round(4).values.tolist()
    tz = Table(zone_data, repeatRows=1)
    tz.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0")),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8),
    ]))
    story.append(tz)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", styles["Heading2"]))
    sum_items = [[k, f"{v}"] for k, v in summary_dict.items()]
    ts = Table([["Metric", "Value"]] + sum_items, colWidths=[210, 290])
    ts.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0")),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8),
    ]))
    story.append(ts)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

inputs_dict = {
    "Face W (m)": face_W,
    "Face H (m)": face_H,
    "Row-to-row pitch (mm)": St_mm,
    "Rows": int(Nr),
    "Tube OD (mm)": Do_mm,
    "Tube wall (mm)": tw_mm,
    "FPI (1/in)": FPI,
    "Fin thickness (mm)": tf_mm,
    "Fin material": fin_mat,
    "Fin k (W/m·K)": fin_k,
    "Tube material": tube_mat,
    "Tube k (W/m·K)": tube_k,
    "Tube circuits": int(Nr_circuits),
    "Airflow mode": "Face velocity" if airflow_mode.startswith("Face") else "Volume flow",
    "Vol. flow (m³/s)": Vdot_m3_s,
    "Inlet mode": air_in_mode,
    "DB in (°C)": Tdb_in,
    "RH in (%)": RH_in if not use_wb_in else "—",
    "WB in (°C)": Twb_in if use_wb_in else "—",
    "Outlet target mode": air_out_mode,
    "Target DB out (°C)": Tdb_out_req,
    "Target RH out (%)": RH_out_req if not use_wb_out else "—",
    "Target WB out (°C)": Twb_out_req if use_wb_out else "—",
    "Refrigerant": fluid,
    "Tsat evap (°C)": Tsat,
    "Superheat required (K)": SH_req,
    "Wet enhancement factor": wet_enh,
    "ṁ_ref total (kg/s)": mdot_ref_total,
    "Air-side fouling": Rfo,
    "Tube-side fouling": Rfi,
}

pdf_bytes = build_pdf(inputs_dict, rows, summary)
st.download_button(
    label="📄 Download report (PDF)",
    data=pdf_bytes,
    file_name="DX_Evaporator_wet_ntu_report.pdf",
    mime="application/pdf"
)

st.caption("Important: Wet rows use BF derived from UA. DB_out is clamped to ≥ Tsat (non-frost assumption). For high-accuracy certification work, upgrade to row-by-row 2D model with real 2φ correlations and psychrometric routines.")
