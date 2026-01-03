
import math
from math import pi, sqrt
import pandas as pd
import numpy as np
import streamlit as st

# ==============================
# Evaporator Coil Designer Core
# ==============================

INCH = 0.0254
MM = 1e-3
P_ATM = 101325.0
R_DA = 287.055
CP_DA = 1006.0
CP_V = 1860.0
H_FG0 = 2501000.0   # latent for air-side mass balance (kJ/kg -> J/kg)

def K(tC): return tC + 273.15

def psat_water_Pa(T_C: float) -> float:
    return 611.21 * math.exp((18.678 - T_C/234.5) * (T_C/(257.14 + T_C)))

def humidity_ratio_from_T_RH(T_C: float, RH_pct: float, P: float = P_ATM) -> float:
    RH = max(min(RH_pct, 100.0), 0.1) / 100.0
    Psat = psat_water_Pa(T_C)
    Pv = RH * Psat
    return 0.62198 * Pv / max(P - Pv, 1.0)

def cp_moist_air_J_per_kgK(T_C: float, W: float) -> float:
    return CP_DA + W*CP_V

def rho_moist_air_kg_m3(T_C: float, W: float, P: float = P_ATM) -> float:
    T_K = K(T_C)
    return P / (R_DA * T_K * (1.0 + 1.6078*W))

def RH_from_T_W(T_C, W, P=P_ATM):
    Pv = W*P/(0.62198 + W)
    Ps = psat_water_Pa(T_C)
    RH = max(min(100.0*Pv/max(Ps,1e-9), 100.0), 0.1)
    return RH

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

def design_evaporator(
    face_W_m=1.2, face_H_m=1.0, St=25.4*MM, Sl=22*MM, Nr=4,
    Do=9.525*MM, t_wall=0.5*MM, FPI=10.0, tf=0.12*MM, fin_k=200.0,
    v_face=2.5, Tdb_in_C=27.0, RH_in_pct=50.0,
    wet_enhance=1.35, tube_circuits=8,
    T_sat_evap_C=6.0, SH_out_K=6.0,
    Rf_o=2e-4, Rf_i=1e-4, k_tube=380.0,
    rho_l=1200.0, mu_l=0.00035, cp_l=2000.0,
    rho_v=35.0,   mu_v=1.2e-5,  cp_v=1000.0,
    h_fg=2.1e6, mdot_ref_total_kg_s=0.08,
):
    # Geometry, areas
    Di = max(Do - 2.0*t_wall, 1e-4)
    geom = geometry_areas(face_W_m, face_H_m, Nr, St, Sl, Do, tf, FPI)
    A_face = geom['face_area']; Ao = geom['A_total']
    area_per_row = Ao / max(1, Nr)

    # Air inlet
    W_in = humidity_ratio_from_T_RH(Tdb_in_C, RH_in_pct)
    rho_air = rho_moist_air_kg_m3(Tdb_in_C, W_in)
    cp_air = cp_moist_air_J_per_kgK(Tdb_in_C, W_in)
    Vdot = v_face * A_face
    mdot_air_dry = rho_air * Vdot / (1.0 + W_in)

    # Air-side properties
    mu_air = 1.716e-5 * ((K(Tdb_in_C)/273.15)**1.5) * ( (273.15+110.4) / (K(Tdb_in_C)+110.4) )
    k_air  = 0.024 + (0.027 - 0.024) * (Tdb_in_C/40.0)
    Pr_air = cp_air * mu_air / max(1e-9, k_air)

    # HTC (dry) then "wet" enhancement
    h_air_dry, meta = air_htc_zukauskas(rho_air*(1+W_in), mu_air, k_air, Pr_air, Do, Nr, mdot_air_dry*(1+W_in), geom['A_min'])
    h_air = h_air_dry * wet_enhance

    # Fin/surface efficiency with "wet" h
    Lc = max(0.5*(min(St, Sl) - Do), 1e-6)
    eta_f = fin_efficiency_infinite_plate(h_air, fin_k, tf, Lc)
    eta_o = 1.0 - (geom['A_fin']/max(Ao,1e-9))*(1.0 - eta_f)

    # Overall U on Ao basis
    A_o_per_m = math.pi * Do
    A_i_per_m = math.pi * Di
    Ao_Ai = A_o_per_m / max(1e-9, A_i_per_m)
    R_wall_per_Ao = (math.log(Do/max(1e-9, Di)) / (2.0*math.pi*k_tube)) / max(1e-12, A_o_per_m)

    def Uo_from(h_i, h_out):
        invU = (1.0/max(1e-9, eta_o*h_out)) + Rf_o + Ao_Ai*((1.0/max(1e-9, h_i)) + Rf_i) + R_wall_per_Ao
        return 1.0 / invU

    # Inside HTCs (rough)
    Ai = pi*(Di**2)/4.0
    mdot_per_circ = mdot_ref_total_kg_s / max(1, tube_circuits)
    G_i = mdot_per_circ / max(1e-12, Ai)
    Re_v = rho_v * (G_i/rho_v) * Di / max(1e-12, mu_v)
    Re_l = rho_l * (G_i/rho_l) * Di / max(1e-12, mu_l)

    def h_i_single(mu, k, cp, rho, Re_i):
        Pr = cp*mu/max(1e-9, k)
        Nu = 0.023 * (max(2300.0, Re_i)**0.8) * (Pr**0.4)
        return Nu * k / max(1e-9, Di)

    h_i_liq = max(300.0, h_i_single(mu_l, 0.6, cp_l, rho_l, Re_l))
    h_i_vap = max(200.0, h_i_single(mu_v, 0.026, cp_v, rho_v, Re_v))
    h_i_tp  = max(800.0, 1.8*h_i_liq)

    # Zone Uo
    U_super = Uo_from(h_i_vap, h_air_dry)   # superheat rows are dry outside
    U_boil  = Uo_from(h_i_tp,  h_air)       # boiling zone: wet-enhanced 'h_out'
    U_pre   = Uo_from(h_i_liq, h_air)       # pre-evap may be wet; conservative

    # Air capacity (dry-air basis)
    C_air = mdot_air_dry * cp_air

    # ---------- ZONE ORDER (air enters at superheat rows) ----------
    rows_SH   = max(0.5, 0.20 * Nr)     # typical ~0.5–1.5 rows
    rows_BOIL = max(1.0, 0.60 * Nr)
    rows_PRE  = max(0.5, Nr - rows_SH - rows_BOIL)
    if rows_PRE < 0.5:
        rows_PRE = 0.5; rows_BOIL = max(1.0, Nr - rows_SH - rows_PRE)

    # ---- Superheat (dry ε–NTU, hot face) ----
    T_air_in = Tdb_in_C
    dT_in_SH = max(0.1, T_air_in - (T_sat_evap_C))
    C_ref_SH = mdot_ref_total_kg_s * max(800.0, cp_v)
    Cmin = min(C_air, C_ref_SH); Cmax = max(C_air, C_ref_SH); Cr = Cmin/max(1e-9, Cmax)
    UA_SH_avail = U_super * rows_SH * area_per_row
    NTU_SH = UA_SH_avail / max(1e-9, Cmin)
    eps_SH = (1.0/Cr) * (1.0 - math.exp(-Cr * (1.0 - math.exp(-NTU_SH)))) if Cr<1.0 else (1.0 - math.exp(-NTU_SH))
    Q_SH_req = mdot_ref_total_kg_s * max(800.0, cp_v) * SH_out_K
    Q_SH = min(Q_SH_req, eps_SH * Cmin * dT_in_SH)

    # If SH not met, steal rows from PRE then BOIL
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

    if Q_SH < 0.999*Q_SH_req:
        add_rows = rows_needed_for_Q(U_super, Q_SH_req, Cmin, dT_in_SH, Cr)/max(1e-9, area_per_row) - rows_SH
        add_rows = max(0.0, add_rows)
        take_pre = min(rows_PRE - 0.2, add_rows) if rows_PRE > 0.2 else 0.0
        rows_PRE -= take_pre; rows_SH += take_pre; add_rows -= take_pre
        if add_rows > 1e-6:
            take_boil = min(rows_BOIL - 1.0, add_rows) if rows_BOIL > 1.0 else 0.0
            rows_BOIL -= take_boil; rows_SH += take_boil; add_rows -= take_boil
        UA_SH_avail = U_super * rows_SH * area_per_row
        NTU_SH = UA_SH_avail / max(1e-9, Cmin)
        eps_SH = (1.0/Cr) * (1.0 - math.exp(-Cr * (1.0 - math.exp(-NTU_SH)))) if Cr<1.0 else (1.0 - math.exp(-NTU_SH))
        Q_SH = min(Q_SH_req, eps_SH * Cmin * dT_in_SH)

    T_air_after_SH = T_air_in - Q_SH/max(1e-9, C_air)

    # ---- Boiling (wet/enthalpy ε–NTU at Tsat) ----
    dT_in_BOIL = max(0.1, T_air_after_SH - T_sat_evap_C)
    UA_BOIL = U_boil * rows_BOIL * area_per_row
    NTU_BOIL = UA_BOIL / max(1e-9, C_air)       # isothermal cold side
    eps_BOIL = 1.0 - math.exp(-NTU_BOIL)
    Q_BOIL_cap = mdot_ref_total_kg_s * h_fg * 0.85
    Q_BOIL = min(Q_BOIL_cap, eps_BOIL * C_air * dT_in_BOIL)
    T_air_after_BOIL = T_air_after_SH - Q_BOIL/max(1e-9, C_air)

    # ---- Pre-evap (liq→sat @ cold face) ----
    dT_ref_liq = 5.0
    C_ref_PRE = mdot_ref_total_kg_s * cp_l
    Cmin = min(C_air, C_ref_PRE); Cmax = max(C_air, C_ref_PRE); Cr = Cmin/max(1e-9, Cmax)
    dT_in_PRE = max(0.1, T_air_after_BOIL - (T_sat_evap_C - dT_ref_liq))
    UA_PRE = U_pre * rows_PRE * area_per_row
    NTU_PRE = UA_PRE / max(1e-9, Cmin)
    eps_PRE = (1.0/Cr) * (1.0 - math.exp(-Cr * (1.0 - math.exp(-NTU_PRE)))) if Cr<1.0 else (1.0 - math.exp(-NTU_PRE))
    Q_PRE_req = mdot_ref_total_kg_s * cp_l * dT_ref_liq
    Q_PRE = min(Q_PRE_req, eps_PRE * Cmin * dT_in_PRE)
    T_air_out = T_air_after_BOIL - Q_PRE/max(1e-9, C_air)

    # Totals & leaving air
    Q_total_W = Q_SH + Q_BOIL + Q_PRE
    Q_sens_W = (Tdb_in_C - T_air_out) * C_air
    Q_lat_W = max(0.0, Q_total_W - Q_sens_W)
    W_out = max(0.0, W_in - Q_lat_W/max(1e-9, mdot_air_dry*H_FG0))
    RH_out = RH_from_T_W(T_air_out, W_out)

    # Air Δp
    dp_air, _meta_dp = air_dp_slot_model(rho_air*(1+W_in), mu_air, v_face, geom, Nr, Sl, tf)

    # Refrigerant Δp per circuit (apportion tube length by rows used)
    tubes_per_row = geom['N_tpr']
    L_total_per_circuit = (tubes_per_row * Nr / max(1, tube_circuits)) * geom['L_tube']
    frac_SH, frac_BOIL, frac_PRE = rows_SH/Nr, rows_BOIL/Nr, rows_PRE/Nr
    L_SH, L_BOIL, L_PRE = frac_SH*L_total_per_circuit, frac_BOIL*L_total_per_circuit, frac_PRE*L_total_per_circuit

    Ai = pi*(Di**2)/4.0
    mdot_per_circ = mdot_ref_total_kg_s / max(1, tube_circuits)
    G = mdot_per_circ / max(1e-9, Ai)

    def dp_single(G, rho, mu, D, L):
        Re = max(1e-9, G*D/max(1e-12, mu))
        f = f_churchill(Re, 1.5e-6/max(1e-12, D))
        return f * (L/max(1e-12, D)) * (G**2) / (2.0*max(1e-9, rho))

    dp_SH = dp_single(G,  rho_v, mu_v, Di, L_SH)
    rho_m, mu_m = mix_rho_mu_homogeneous(0.5, rho_v, rho_l, mu_v, mu_l)
    dp_BOIL = dp_single(G, rho_m, mu_m, Di, L_BOIL)
    dp_PRE  = dp_single(G,  rho_l, mu_l, Di, L_PRE)
    dp_total_kPa = (dp_SH + dp_BOIL + dp_PRE)/1000.0

    # Output tables
    rows_df = pd.DataFrame([
        ["Superheat (vapor) @ hot face",  Q_SH/1000.0,   rows_SH,  U_super*rows_SH*area_per_row, NTU_SH,  (eps_SH),  L_SH,   dp_SH/1000.0],
        ["Boiling (2φ @ Tsat) middle",    Q_BOIL/1000.0, rows_BOIL, U_boil*rows_BOIL*area_per_row, NTU_BOIL, (eps_BOIL), L_BOIL, dp_BOIL/1000.0],
        ["Pre-evap (liq→sat) cold face",  Q_PRE/1000.0,  rows_PRE, U_pre*rows_PRE*area_per_row, NTU_PRE, (eps_PRE),  L_PRE,  dp_PRE/1000.0],
    ], columns=["Zone","Q (kW)","Rows used","UA (W/K)","NTU","ε","L per circuit (m)","Δp_zone (kPa)"])

    summary = {
        "Q_total_kW": Q_total_W/1000.0,
        "Q_sensible_kW": Q_sens_W/1000.0,
        "Q_latent_kW": Q_lat_W/1000.0,
        "Air_out_C": T_air_out,
        "RH_out_pct": RH_out,
        "Air_dP_Pa": dp_air,
        "Ref_dP_total_kPa_per_circuit": dp_total_kPa,
        "Rows_SH": rows_SH, "Rows_BOIL": rows_BOIL, "Rows_PRE": rows_PRE,
        "Face_velocity_m_s": v_face,
        "A_o_total_m2": Ao,
        "eta_o": eta_o,
        "h_air_dry_W_m2K": h_air_dry, "h_air_wetproxy_W_m2K": h_air,
        "U_SH": U_super, "U_BOIL": U_boil, "U_PRE": U_pre
    }
    return rows_df, summary

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="DX Evaporator — Zoned ε–NTU", layout="wide")

st.title("DX Evaporator — Zoned ε–NTU Designer")
st.caption("Air enters at Superheat rows → Boiling → Pre-evap (refrigerant counterflow).")

with st.sidebar:
    st.header("Geometry & Air")
    face_W = st.number_input("Face width W (m)", 0.2, 3.0, 1.2, 0.01)
    face_H = st.number_input("Face height H (m)", 0.2, 3.0, 1.0, 0.01)
    St = st.number_input("Transverse pitch St (m)", 0.01, 0.05, 0.0254, 0.001)
    Sl = st.number_input("Longitudinal pitch Sl (m)", 0.01, 0.05, 0.0220, 0.001)
    Nr = st.number_input("Rows (depth)", 1, 12, 4, 1)
    Do = st.number_input("Tube OD (m)", 0.005, 0.020, 0.009525, 0.0001)
    tw = st.number_input("Tube wall (m)", 0.0002, 0.0015, 0.0005, 0.00005)
    FPI = st.number_input("Fins per inch", 4.0, 24.0, 10.0, 0.5)
    tf  = st.number_input("Fin thickness (m)", 0.00006, 0.0003, 0.00012, 0.00001)
    fin_k = st.number_input("Fin conductivity (W/m·K)", 120.0, 230.0, 200.0, 1.0)

    st.header("Operating")
    v_face = st.number_input("Face velocity (m/s)", 0.5, 4.0, 2.5, 0.1)
    Tdb = st.number_input("Air in (°C)", 10.0, 50.0, 27.0, 0.1)
    RH = st.number_input("Air in RH (%)", 10.0, 100.0, 50.0, 1.0)
    Tsat = st.number_input("Tsat evap (°C)", -10.0, 15.0, 6.0, 0.1)
    SH = st.number_input("Required superheat (K)", 0.0, 20.0, 6.0, 0.5)
    circuits = st.number_input("Tube circuits (parallel)", 2, 32, 8, 1)
    mref = st.number_input("Refrigerant mass flow (kg/s total)", 0.01, 0.5, 0.08, 0.005)

    st.header("Models")
    wet_enh = st.number_input("Wet enhancement factor (air)", 1.0, 2.5, 1.35, 0.05)
    Rfo = st.number_input("Air-side fouling (m²·K/W)", 0.0, 0.001, 0.0002, 0.00005)
    Rfi = st.number_input("Tube-side fouling (m²·K/W)", 0.0, 0.001, 0.0001, 0.00005)

rows, summary = design_evaporator(
    face_W_m=face_W, face_H_m=face_H, St=St, Sl=Sl, Nr=Nr,
    Do=Do, t_wall=tw, FPI=FPI, tf=tf, fin_k=fin_k,
    v_face=v_face, Tdb_in_C=Tdb, RH_in_pct=RH,
    wet_enhance=wet_enh, tube_circuits=circuits,
    T_sat_evap_C=Tsat, SH_out_K=SH,
    Rf_o=Rfo, Rf_i=Rfi,
    mdot_ref_total_kg_s=mref
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Totals")
    st.metric("Total cooling (kW)", f"{summary['Q_total_kW']:.2f}")
    st.metric("Sensible (kW)", f"{summary['Q_sensible_kW']:.2f}")
    st.metric("Latent (kW)", f"{summary['Q_latent_kW']:.2f}")
with col2:
    st.subheader("Leaving Air / Δp")
    st.metric("Air out (°C)", f"{summary['Air_out_C']:.2f}")
    st.metric("Air out RH (%)", f"{summary['RH_out_pct']:.1f}")
    st.metric("Air Δp (Pa)", f"{summary['Air_dP_Pa']:.1f}")
    st.metric("Ref Δp per circuit (kPa)", f"{summary['Ref_dP_total_kPa_per_circuit']:.2f}")

st.subheader("Row Allocation (air enters → SH → Boil → Pre)")
st.write(f"Rows_SH = {summary['Rows_SH']:.2f}, Rows_BOIL = {summary['Rows_BOIL']:.2f}, Rows_PRE = {summary['Rows_PRE']:.2f}")

st.subheader("Zone Table")
st.dataframe(rows.style.format({
    "Q (kW)": "{:.2f}",
    "Rows used": "{:.2f}",
    "UA (W/K)": "{:.0f}",
    "NTU": "{:.2f}",
    "ε": "{:.3f}",
    "L per circuit (m)": "{:.2f}",
    "Δp_zone (kPa)": "{:.2f}",
}), use_container_width=True)

st.caption("First-principles scaffold. For production: plug in your calibrated j/f correlations and CoolProp.")
