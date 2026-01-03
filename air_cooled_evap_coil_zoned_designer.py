# -*- coding: utf-8 -*-
import math
from math import pi, sqrt
import traceback
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="DX Evaporator — Zoned ε–NTU (Single File)", layout="wide")

# Try to bring in CoolProp (for refrigerant properties)
try:
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except Exception:
    HAS_COOLPROP = False

# ------------------------- Constants & Psychrometrics -------------------------
INCH = 0.0254
MM = 1e-3
P_ATM = 101325.0
R_DA = 287.055
CP_DA = 1006.0      # J/kg-K dry air
CP_V = 1860.0       # J/kg-K water vapor
H_LV0 = 2501000.0   # J/kg at ~0°C (for moist-air enthalpy calc)

def K(tC): return tC + 273.15

def psat_water_Pa(T_C: float) -> float:
    # Tetens-like
    return 611.21 * math.exp((18.678 - T_C/234.5) * (T_C/(257.14 + T_C)))

def humidity_ratio_from_T_RH(T_C: float, RH_pct: float, P: float = P_ATM) -> float:
    RH = max(min(RH_pct, 100.0), 0.1) / 100.0
    Psat = psat_water_Pa(T_C)
    Pv = RH * Psat
    return 0.62198 * Pv / max(P - Pv, 1.0)

def humidity_ratio_from_T_WB(Tdb_C: float, Twb_C: float, P: float = P_ATM) -> float:
    # Adiabatic saturation approximation
    W_sat_wb = humidity_ratio_from_T_RH(Twb_C, 100.0, P)
    cp_da = 1006.0
    cp_wv = 1860.0
    h_fg_wb = 2501000.0 - 2369.0 * Twb_C  # J/kg, simple linear approx vs T
    numer = (W_sat_wb * (h_fg_wb + cp_wv * Twb_C) - cp_da * (Tdb_C - Twb_C))
    denom = (h_fg_wb + cp_wv * Tdb_C)
    W = numer / max(1e-9, denom)
    return max(0.0, W)

def moist_air_enthalpy_J_per_kg_da(T_C: float, W: float) -> float:
    # h (J/kg dry air)
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

# ------------------------- Heat-transfer Utilities ----------------------------
def fin_efficiency_infinite_plate(h, k_fin, t_fin, Lc):
    if Lc <= 0 or t_fin <= 0 or k_fin <= 0: return 1.0
    m = sqrt(2.0*h/(k_fin*t_fin))
    x = max(m*Lc, 1e-9)
    return math.tanh(x)/x

def geometry_areas(W, H, Nr, St, Sl, Do, tf, FPI):
    face_area = W*H
    fins_count = int(round(FPI * (H/INCH)))              # fins across height
    N_tpr = max(int(math.floor(H / max(St,1e-9))), 1)    # tubes per row
    N_tubes = N_tpr * Nr
    L_tube = W
    s = (1.0/FPI) * INCH
    A_holes_one_fin = N_tubes * (pi*(Do/2)**2)
    A_fin_one = max(2.0*(W*H - A_holes_one_fin), 0.0)    # both sides
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

def air_dp_slot_model(rho, mu, v_face, geom, Nr, St, tf, K_in=0.5, K_out=1.0, K_row=0.0):
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
    dp_minor = (K_in + K_out + K_row * Nr) * q
    return dp_fric + dp_minor, dict(D_h=D_h, Re=Re_ch, v_core=v_core, f_D=f_D, L=L_flow)

def f_churchill(Re, e_over_D=1.5e-6/9.5e-3):
    Re = max(1e-9, Re)
    if Re < 2300.0:
        return 64.0 / max(1.0, Re)
    A = (2.457 * math.log( (7.0 / max(1.0, Re))**0.9 + 0.27*e_over_D ))**16
    B = (37530.0 / max(1.0, Re))**16
    f = 8.0 * ( ( (8.0 / max(1.0, Re))**12 ) + 1.0 / ( (A + B)**1.5 ) )**(1.0/12.0)
    return max(1e-6, f)

def dp_single_phase_friction(G, rho, mu, D, L, roughness=1.5e-6):
    Re = max(1e-9, G*D/max(1e-12, mu))
    f = f_churchill(Re, roughness/max(1e-12, D))
    dp = f * (L/max(1e-12, D)) * (G**2) / (2.0*max(1e-9, rho))
    return dp, Re, f

def mix_rho_mu_homogeneous(x, rho_v, rho_l, mu_v, mu_l):
    rho_m = 1.0 / ( (x/max(1e-12, rho_v)) + ((1.0-x)/max(1e-12, rho_l)) )
    mu_m  = x*mu_v + (1.0 - x)*mu_l
    return rho_m, mu_m

# ------------------------- Design Core ----------------------------
def design_evaporator(
    # Geometry (SI base; UI will pass mm → m)
    face_W_m, face_H_m, St_m, Sl_m, Nr,
    Do_m, t_wall_m, FPI, tf_m, fin_k, tube_k,
    # Airflow & state
    v_face, Vdot_m3_s,
    Tdb_in_C, RH_in_pct, use_wb_in, Twb_in_C,
    Tdb_out_req_C, RH_out_req_pct, use_wb_out, Twb_out_req_C,
    # Refrigerant & operating
    fluid, T_sat_evap_C, SH_out_K, mdot_ref_total,  # <-- mass flow now required
    # Circuiting & fouling
    tube_circuits, wet_enhance, Rf_o, Rf_i, k_tube_override=None,
):
    # Convert air states
    if use_wb_in:
        W_in = humidity_ratio_from_T_WB(Tdb_in_C, Twb_in_C)
    else:
        W_in = humidity_ratio_from_T_RH(Tdb_in_C, RH_in_pct)
    if use_wb_out:
        W_out_req = humidity_ratio_from_T_WB(Tdb_out_req_C, Twb_out_req_C)
    else:
        W_out_req = humidity_ratio_from_T_RH(Tdb_out_req_C, RH_out_req_pct)

    # Required capacities from air flow and air state change
    h_in = moist_air_enthalpy_J_per_kg_da(Tdb_in_C, W_in)
    h_out_req = moist_air_enthalpy_J_per_kg_da(Tdb_out_req_C, W_out_req)
    rho_in = rho_moist_air_kg_m3(Tdb_in_C, W_in)
    # Mass flow of dry air
    mdot_da = (rho_in * Vdot_m3_s) / (1.0 + W_in)
    Q_required_W = mdot_da * (h_in - h_out_req)
    cp_air_in = cp_moist_air_J_per_kgK(Tdb_in_C, W_in)
    Q_sensible_required_W = mdot_da * cp_air_in * (Tdb_in_C - Tdb_out_req_C)

    # Geometry & areas
    geom = geometry_areas(face_W_m, face_H_m, Nr, St_m, Sl_m, Do_m, tf_m, FPI)
    A_face = geom['face_area']; Ao = geom['A_total']; A_min = geom['A_min']
    area_per_row = Ao / max(1, Nr)
    tubes_per_row = geom['N_tpr']
    L_total_per_circuit = (tubes_per_row * Nr / max(1, tube_circuits)) * geom['L_tube']

    # Air properties
    mu_air = 1.716e-5 * ((K(Tdb_in_C)/273.15)**1.5) * ( (273.15+110.4) / (K(Tdb_in_C)+110.4) )
    k_air  = 0.024 + (0.027 - 0.024) * (Tdb_in_C/40.0)
    cp_air = cp_air_in
    Pr_air = cp_air * mu_air / max(1e-9, k_air)
    mdot_air_total = mdot_da * (1.0 + W_in)

    # Air HTC (dry) & wet enhancement
    h_air_dry, meta_htc = air_htc_zukauskas(rho_in*(1+W_in), mu_air, k_air, Pr_air, Do_m, Nr, mdot_air_total, A_min)
    h_air = h_air_dry * wet_enhance
    Lc = max(0.5*(min(St_m, Sl_m) - Do_m), 1e-6)
    eta_f = fin_efficiency_infinite_plate(h_air, fin_k, tf_m, Lc)
    eta_o = 1.0 - (geom['A_fin']/max(Ao,1e-9))*(1.0 - eta_f)

    # Refrigerant properties (CoolProp)
    if not HAS_COOLPROP:
        raise RuntimeError("CoolProp is required. Add 'CoolProp>=6.6' to requirements.txt and redeploy.")
    T_sat_K = K(T_sat_evap_C)
    P_sat_Pa = PropsSI("P","T",T_sat_K,"Q",0,fluid)
    rho_l = PropsSI("D","T",T_sat_K,"Q",0,fluid)
    rho_v = PropsSI("D","T",T_sat_K,"Q",1,fluid)
    mu_l  = PropsSI("V","T",T_sat_K,"Q",0,fluid)
    mu_v  = PropsSI("V","T",T_sat_K,"Q",1,fluid)
    cp_l  = PropsSI("C","T",T_sat_K,"Q",0,fluid)
    cp_v  = PropsSI("C","T",T_sat_K,"Q",1,fluid)
    h_fg  = PropsSI("H","T",T_sat_K,"Q",1,fluid) - PropsSI("H","T",T_sat_K,"Q",0,fluid)
    k_l   = PropsSI("L","T",T_sat_K,"Q",0,fluid)
    k_v   = PropsSI("L","T",T_sat_K,"Q",1,fluid)

    # Tube wall conduction
    Di = max(Do_m - 2.0*t_wall_m, 1e-5)
    A_o_per_m = math.pi * Do_m
    A_i_per_m = math.pi * Di
    Ao_Ai = A_o_per_m / max(1e-9, A_i_per_m)
    k_tube = tube_k if k_tube_override is None else k_tube_override
    R_wall_per_Ao = (math.log(Do_m/max(1e-9, Di)) / (2.0*math.pi*k_tube)) / max(1e-12, A_o_per_m)

    def Uo_from(h_i, h_out):
        invU = (1.0/max(1e-9, eta_o*h_out)) + Rf_o + Ao_Ai*((1.0/max(1e-9, h_i)) + Rf_i) + R_wall_per_Ao
        return 1.0 / invU

    # Inside HTCs (simple Dittus-Boelter)
    Ai = pi*(Di**2)/4.0
    mdot_ref_total = max(1e-9, mdot_ref_total)  # ensure positive
    mdot_per_circ = mdot_ref_total / max(1, tube_circuits)
    G_i = mdot_per_circ / max(1e-9, Ai)

    def h_i_single(mu, k, cp, rho, D, G):
        Re = max(1e-9, G*D/max(1e-12, mu))
        Pr = cp*mu/max(1e-9, k)
        Nu = 0.023 * (max(2300.0, Re)**0.8) * (Pr**0.4)
        h = Nu * k / max(1e-9, D)
        return h, Re, Pr

    h_i_liq, Re_l, Pr_l = h_i_single(mu_l, k_l, cp_l, rho_l, Di, G_i)
    h_i_vap, Re_v, Pr_v = h_i_single(mu_v, k_v, cp_v, rho_v, Di, G_i)
    h_i_tp  = max(800.0, 1.8*h_i_liq)

    # Zone U values
    U_super = Uo_from(h_i_vap, h_air_dry)   # superheat rows dry
    U_boil  = Uo_from(h_i_tp,  h_air)       # boiling wet
    U_pre   = Uo_from(h_i_liq, h_air)       # pre-evap

    # Air capacity (dry-air basis)
    C_air = mdot_da * cp_air

    # Zone allocation (air enters at superheat rows)
    rows_SH   = max(0.5, 0.20 * Nr)
    rows_BOIL = max(1.0, 0.60 * Nr)
    rows_PRE  = max(0.5, Nr - rows_SH - rows_BOIL)
    if rows_PRE < 0.5:
        rows_PRE = 0.5; rows_BOIL = max(1.0, Nr - rows_SH - rows_PRE)

    # Superheat zone
    T_air_in = Tdb_in_C
    dT_in_SH = max(0.1, T_air_in - (T_sat_evap_C))
    C_ref_SH = mdot_ref_total * max(800.0, cp_v)
    Cmin = min(C_air, C_ref_SH); Cmax = max(C_air, C_ref_SH); Cr = Cmin/max(1e-9, Cmax)
    UA_SH = U_super * rows_SH * area_per_row
    NTU_SH = UA_SH / max(1e-9, Cmin)
    eps_SH = (1.0/Cr) * (1.0 - math.exp(-Cr * (1.0 - math.exp(-NTU_SH)))) if Cr<1.0 else (1.0 - math.exp(-NTU_SH))
    Q_SH_req = mdot_ref_total * max(800.0, cp_v) * SH_out_K
    Q_SH = min(Q_SH_req, eps_SH * Cmin * dT_in_SH)
    unmet_SH = Q_SH < 0.999*Q_SH_req

    # Row reallocation to satisfy SH first
    def rows_needed_for_Q(U, Q_target, Cmin, dT_in, Cr):
        target = min(0.999999, max(1e-8, Q_target/(Cmin*dT_in)))
        lo, hi = 1e-6, 80.0
        while hi - lo > 1e-5:
            mid = 0.5*(lo+hi)
            eps = (1.0/Cr) * (1.0 - math.exp(-Cr * (1.0 - math.exp(-mid)))) if Cr<1.0 else (1.0 - math.exp(-mid))
            if eps < target: lo = mid
            else: hi = mid
        NTU_need = 0.5*(lo+hi)
        UA_need = NTU_need * Cmin
        return UA_need / max(1e-9, U)

    if unmet_SH:
        add_rows = rows_needed_for_Q(U_super, Q_SH_req, Cmin, dT_in_SH, Cr)/max(1e-9, area_per_row) - rows_SH
        add_rows = max(0.0, add_rows)
        take_pre = min(rows_PRE - 0.2, add_rows) if rows_PRE > 0.2 else 0.0
        rows_PRE -= take_pre; rows_SH += take_pre; add_rows -= take_pre
        if add_rows > 1e-6:
            take_boil = min(rows_BOIL - 1.0, add_rows) if rows_BOIL > 1.0 else 0.0
            rows_BOIL -= take_boil; rows_SH += take_boil; add_rows -= take_boil
        UA_SH = U_super * rows_SH * area_per_row
        NTU_SH = UA_SH / max(1e-9, Cmin)
        eps_SH = (1.0/Cr) * (1.0 - math.exp(-Cr * (1.0 - math.exp(-NTU_SH)))) if Cr<1.0 else (1.0 - math.exp(-NTU_SH))
        Q_SH = min(Q_SH_req, eps_SH * Cmin * dT_in_SH)
        unmet_SH = Q_SH < 0.999*Q_SH_req

    T_air_after_SH = T_air_in - Q_SH/max(1e-9, C_air)

    # Boiling zone (enthalpy ε–NTU with isothermal cold side)
    dT_in_BOIL = max(0.1, T_air_after_SH - T_sat_evap_C)
    UA_BOIL = U_boil * rows_BOIL * area_per_row
    NTU_BOIL = UA_BOIL / max(1e-9, C_air)
    eps_BOIL = 1.0 - math.exp(-NTU_BOIL)
    Q_BOIL_cap = mdot_ref_total * h_fg * 0.85
    Q_BOIL = min(Q_BOIL_cap, eps_BOIL * C_air * dT_in_BOIL)
    T_air_after_BOIL = T_air_after_SH - Q_BOIL/max(1e-9, C_air)
    unmet_BOIL = (Q_BOIL + 1e-6) < eps_BOIL*C_air*dT_in_BOIL and Q_BOIL_cap < eps_BOIL*C_air*dT_in_BOIL

    # Pre-evap (liq → sat)
    dT_ref_liq = 5.0
    C_ref_PRE = mdot_ref_total * cp_l
    Cmin2 = min(C_air, C_ref_PRE); Cmax2 = max(C_air, C_ref_PRE); Cr2 = Cmin2/max(1e-9, Cmax2)
    dT_in_PRE = max(0.1, T_air_after_BOIL - (T_sat_evap_C - dT_ref_liq))
    UA_PRE = U_pre * rows_PRE * area_per_row
    NTU_PRE = UA_PRE / max(1e-9, Cmin2)
    eps_PRE = (1.0/Cr2) * (1.0 - math.exp(-Cr2 * (1.0 - math.exp(-NTU_PRE)))) if Cr2<1.0 else (1.0 - math.exp(-NTU_PRE))
    Q_PRE_req = mdot_ref_total * cp_l * dT_ref_liq
    Q_PRE = min(Q_PRE_req, eps_PRE * Cmin2 * dT_in_PRE)
    T_air_out = T_air_after_BOIL - Q_PRE/max(1e-9, C_air)
    unmet_PRE = Q_PRE < 0.999*Q_PRE_req

    # Totals & leaving air state
    Q_total_W = Q_SH + Q_BOIL + Q_PRE
    Q_sens_W = (Tdb_in_C - T_air_out) * C_air
    Q_lat_W = max(0.0, Q_total_W - Q_sens_W)
    W_out = max(0.0, W_in - Q_lat_W/max(1e-9, mdot_da*(H_LV0)))
    RH_out = RH_from_T_W(T_air_out, W_out)

    # Wet-bulb out (approx)
    def wb_from_T_RH_like(Tdb, W_target):
        lo, hi = 0.0, Tdb
        for _ in range(40):
            mid = 0.5*(lo+hi)
            W_mid = humidity_ratio_from_T_WB(Tdb, mid)
            if W_mid > W_target: hi = mid
            else: lo = mid
        return 0.5*(lo+hi)
    Twb_out = wb_from_T_RH_like(T_air_out, W_out)

    # Air-side Δp and velocities
    v_face_calc = Vdot_m3_s / max(1e-9, A_face)
    dp_air, dp_meta = air_dp_slot_model(rho_in*(1+W_in), mu_air, v_face_calc, geom, Nr, Sl_m, tf_m)

    # Refrigerant Δp & velocities per zone
    Ai = pi*(Di**2)/4.0
    G = mdot_per_circ / max(1e-9, Ai)

    frac_SH, frac_BOIL, frac_PRE = rows_SH/Nr, rows_BOIL/Nr, rows_PRE/Nr
    L_SH = frac_SH*L_total_per_circuit
    L_BOIL = frac_BOIL*L_total_per_circuit
    L_PRE = frac_PRE*L_total_per_circuit

    def dp_vel(G, rho, mu, D, L):
        Re = max(1e-9, G*D/max(1e-12, mu))
        f = f_churchill(Re, 1.5e-6/max(1e-12, D))
        dp = f * (L/max(1e-12, D)) * (G**2) / (2.0*max(1e-9, rho))
        v = G/max(1e-9, rho) / (pi*(D**2)/4.0)  # m/s
        return dp, Re, v, f

    dp_SH, Re_SH, vel_SH, f_SH = dp_vel(G,  rho_v, mu_v, Di, L_SH)
    rho_m, mu_m = mix_rho_mu_homogeneous(0.5, rho_v, rho_l, mu_v, mu_l)
    dp_BOIL, Re_BOIL, vel_BOIL, f_BOIL = dp_vel(G, rho_m, mu_m, Di, L_BOIL)
    dp_PRE, Re_PRE, vel_PRE, f_PRE = dp_vel(G,  rho_l, mu_l, Di, L_PRE)
    dp_total_kPa = (dp_SH + dp_BOIL + dp_PRE)/1000.0

    # Zone-limitation text
    insuff = []
    if unmet_SH: insuff.append("Superheat zone")
    if unmet_BOIL: insuff.append("Boiling zone")
    if unmet_PRE: insuff.append("Pre-evap zone")
    insuff_text = ", ".join(insuff) if insuff else "None"

    # Pack outputs
    rows_df = pd.DataFrame([
        ["Superheat (vapor) @ hot face",  Q_SH/1000.0,   rows_SH,  U_super*rows_SH*area_per_row, NTU_SH,  (eps_SH),  L_SH,   Re_SH,   f_SH,   vel_SH,  dp_SH/1000.0],
        ["Boiling (2φ @ Tsat) middle",    Q_BOIL/1000.0, rows_BOIL, U_boil*rows_BOIL*area_per_row, NTU_BOIL, (eps_BOIL), L_BOIL, Re_BOIL, f_BOIL, vel_BOIL, dp_BOIL/1000.0],
        ["Pre-evap (liq→sat) cold face",  Q_PRE/1000.0,  rows_PRE, U_pre*rows_PRE*area_per_row, NTU_PRE, (eps_PRE),  L_PRE,  Re_PRE,  f_PRE,  vel_PRE,  dp_PRE/1000.0],
    ], columns=["Zone","Q (kW)","Rows used","UA (W/K)","NTU","ε","L per circuit (m)","Re","f","Vel (m/s)","Δp_zone (kPa)"])

    summary = {
        # Required vs achieved
        "Q_required_kW": Q_required_W/1000.0,
        "Q_sensible_required_kW": Q_sensible_required_W/1000.0,
        "Q_achieved_kW": (Q_total_W)/1000.0,
        "Q_sensible_achieved_kW": Q_sens_W/1000.0,
        # Leaving air achieved
        "Air_out_DB_C": T_air_out,
        "Air_out_WB_C": Twb_out,
        "Air_out_RH_pct": RH_out,
        # Rows
        "Rows_entered": Nr,
        "Rows_SH": rows_SH, "Rows_BOIL": rows_BOIL, "Rows_PRE": rows_PRE,
        "Zone_limitation": insuff_text,
        # Areas & lengths
        "A_face_m2": A_face, "A_total_o_m2": Ao, "A_min_m2": A_min,
        "Ao/row_m2": area_per_row, "Tubes_per_row": tubes_per_row,
        "Tube_length_per_circuit_m": L_total_per_circuit,
        # Airflow
        "Face_velocity_m_s": v_face_calc, "Vdot_m3_s": Vdot_m3_s,
        "Air_dP_Pa": dp_air,
        # Refrigerant props @ Tsat
        "Fluid": fluid, "Tsat_C": T_sat_evap_C, "Psat_bar": P_sat_Pa/1e5,
        "rho_l": rho_l, "rho_v": rho_v, "mu_l_Pas": mu_l, "mu_v_Pas": mu_v,
        "cp_l": cp_l, "cp_v": cp_v, "h_fg_kJ_per_kg": h_fg/1000.0,
        # Tube/fin materials
        "k_fin_used_W_mK": fin_k, "k_tube_used_W_mK": k_tube,
        # Refrigerant mass flow
        "mdot_ref_total_kg_s": mdot_ref_total,
        "mdot_ref_per_circuit_kg_s": mdot_per_circ,
        # Ref Δp & velocities
        "Ref_dP_total_kPa": dp_total_kPa,
        "Re_SH": Re_SH, "Re_BOIL": Re_BOIL, "Re_PRE": Re_PRE,
        "Vel_SH_m_s": vel_SH, "Vel_BOIL_m_s": vel_BOIL, "Vel_PRE_m_s": vel_PRE,
    }

    return rows_df, summary

# ------------------------- UI ----------------------------

st.title("DX Evaporator — Zoned ε–NTU Designer (Single File)")

# Geometry
st.header("Geometry & Materials")
colg1, colg2, colg3, colg4 = st.columns(4)
with colg1:
    face_W = st.number_input("Face width W (m)", 0.2, 4.0, 1.2, 0.01, format="%.2f")
with colg2:
    face_H = st.number_input("Face height H (m)", 0.2, 4.0, 1.0, 0.01, format="%.2f")
with colg3:
    St_mm = st.number_input("Row-to-row pitch (mm)", 10.00, 60.00, 22.00, 0.01, format="%.2f")
with colg4:
    Sl_mm = st.number_input("Longitudinal pitch (mm)", 10.00, 60.00, 24.40, 0.01, format="%.2f")

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

# Conductivities
MAT_K = {"Copper": 380.0, "Aluminum": 205.0, "Steel": 50.0, "CuNi 90/10": 29.0}
fin_k = MAT_K[fin_mat]
tube_k = MAT_K[tube_mat]

# Airflow input
st.header("Airflow & Inlet/Outlet Air State")
airflow_mode = st.radio("Airflow input mode", ["Face velocity (m/s)", "Volume flow (m³/h)"], horizontal=True)
if airflow_mode == "Face velocity (m/s)":
    v_face = st.number_input("Face velocity (m/s)", 0.3, 6.0, 2.5, 0.1, format="%.2f")
    Vdot_m3_s = v_face * face_W * face_H
else:
    Vdot_m3_h = st.number_input("Air volume flow (m³/h)", 500.0, 50000.0, 8000.0, 10.0, format="%.1f")
    Vdot_m3_s = Vdot_m3_h / 3600.0
    v_face = Vdot_m3_s / max(1e-9, face_W * face_H)

# Air state input modes
air_in_mode = st.radio("Inlet air condition input", ["DB + RH", "DB + WB"], horizontal=True)
if air_in_mode == "DB + RH":
    Tdb_in = st.number_input("DB in (°C)", 5.0, 55.0, 27.0, 0.1, format="%.1f")
    RH_in = st.number_input("RH in (%)", 5.0, 100.0, 50.0, 0.1, format="%.1f")
    use_wb_in, Twb_in = False, 0.0
else:
    Tdb_in = st.number_input("DB in (°C)", 5.0, 55.0, 27.0, 0.1, format="%.1f")
    Twb_in = st.number_input("WB in (°C)", 0.0, 35.0, 19.0, 0.1, format="%.1f")
    RH_in = 50.0
    use_wb_in = True

air_out_mode = st.radio("Required outlet air input", ["DB + RH", "DB + WB"], horizontal=True)
if air_out_mode == "DB + RH":
    Tdb_out_req = st.number_input("Required DB out (°C)", -5.0, 40.0, 14.0, 0.1, format="%.1f")
    RH_out_req = st.number_input("Required RH out (%)", 5.0, 100.0, 95.0, 0.1, format="%.1f")
    use_wb_out, Twb_out_req = False, 0.0
else:
    Tdb_out_req = st.number_input("Required DB out (°C)", -5.0, 40.0, 14.0, 0.1, format="%.1f")
    Twb_out_req = st.number_input("Required WB out (°C)", -5.0, 35.0, 13.0, 0.1, format="%.1f")
    RH_out_req = 95.0
    use_wb_out = True

# Refrigerant selection & operating
st.header("Refrigerant & Operating")
fluid = st.selectbox("Refrigerant (CoolProp name)", ["R134a", "R410A", "R407C", "R404A", "R32", "R22"])
Tsat = st.number_input("Evaporating saturation temperature Tsat (°C)", -20.0, 15.0, 6.0, 0.1, format="%.1f")
SH = st.number_input("Required refrigerant superheat (K)", 0.0, 25.0, 6.0, 0.5, format="%.1f")
wet_enh = st.number_input("Wet enhancement factor (air)", 1.0, 2.5, 1.35, 0.05, format="%.2f")

# NEW: Refrigerant mass flow input
mdot_ref_total = st.number_input("Total refrigerant mass flow (kg/s)", 0.001, 2.000, 0.080, 0.001, format="%.3f")

# Fouling
st.header("Fouling (optional)")
colf1, colf2 = st.columns(2)
with colf1:
    Rfo = st.number_input("Air-side fouling (m²·K/W)", 0.0, 0.001, 0.0002, 0.00005, format="%.5f")
with colf2:
    Rfi = st.number_input("Tube-side fouling (m²·K/W)", 0.0, 0.001, 0.0001, 0.00005, format="%.5f")

# Convert mm → m for geometry
St = St_mm * MM
Sl = Sl_mm * MM
Do = Do_mm * MM
tw = tw_mm * MM
tf = tf_mm * MM

# Run design
try:
    rows, summary = design_evaporator(
        face_W_m=face_W, face_H_m=face_H, St_m=St, Sl_m=Sl, Nr=Nr,
        Do_m=Do, t_wall_m=tw, FPI=FPI, tf_m=tf, fin_k=fin_k, tube_k=tube_k,
        v_face=v_face, Vdot_m3_s=Vdot_m3_s,
        Tdb_in_C=Tdb_in, RH_in_pct=RH_in, use_wb_in=use_wb_in, Twb_in_C=Twb_in,
        Tdb_out_req_C=Tdb_out_req, RH_out_req_pct=RH_out_req, use_wb_out=use_wb_out, Twb_out_req_C=Twb_out_req,
        fluid=fluid, T_sat_evap_C=Tsat, SH_out_K=SH, mdot_ref_total=mdot_ref_total,
        tube_circuits=Nr_circuits, wet_enhance=wet_enh, Rf_o=Rfo, Rf_i=Rfi,
        k_tube_override=None,
    )
except Exception as e:
    st.error("Design solver error: " + str(e))
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    st.stop()

# ------------------------- Output Sections ----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Required (from airflow & states)")
    st.metric("Total required (kW)", f"{summary['Q_required_kW']:.2f}")
    st.metric("Sensible required (kW)", f"{summary['Q_sensible_required_kW']:.2f}")
with c2:
    st.subheader("Achieved by coil")
    st.metric("Total achieved (kW)", f"{summary['Q_achieved_kW']:.2f}")
    st.metric("Sensible achieved (kW)", f"{summary['Q_sensible_achieved_kW']:.2f}")
with c3:
    st.subheader("Leaving air (achieved)")
    st.metric("DB out (°C)", f"{summary['Air_out_DB_C']:.2f}")
    st.metric("WB out (°C)", f"{summary['Air_out_WB_C']:.2f}")
    st.metric("RH out (%)", f"{summary['Air_out_RH_pct']:.1f}")

st.subheader("Zone allocation & limits")
st.write(f"Rows entered = {summary['Rows_entered']}; Used → SH {summary['Rows_SH']:.2f}, Boil {summary['Rows_BOIL']:.2f}, Pre {summary['Rows_PRE']:.2f}")
st.info(f"Cooling insufficiency zones: {summary['Zone_limitation']}")

st.subheader("Zone table")
st.dataframe(rows.style.format({
    "Q (kW)": "{:.2f}",
    "Rows used": "{:.2f}",
    "UA (W/K)": "{:.0f}",
    "NTU": "{:.2f}",
    "ε": "{:.3f}",
    "L per circuit (m)": "{:.2f}",
    "Re": "{:.0f}",
    "f": "{:.4f}",
    "Vel (m/s)": "{:.2f}",
    "Δp_zone (kPa)": "{:.2f}",
}), use_container_width=True)

with st.expander("Intermediate values & diagnostics", expanded=False):
    diag = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
    st.dataframe(diag, use_container_width=True)

st.caption("Notes: Tube/fin thermal conductivities inferred from material selections and shown above. Refrigerant properties are pulled from CoolProp at the entered Tsat. Wet enhancement applied in Boiling and Pre-evap zones; Superheat rows use dry-air HTC.")
