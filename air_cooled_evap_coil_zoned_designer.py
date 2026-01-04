# air_cooled_evap_coil_zoned_designer.py
# DX Evaporator – Wet Coil ε–NTU (Evaporation + Superheat only)
# Version v3 (physics-corrected)

import math
import traceback
import streamlit as st
from math import pi

try:
    from CoolProp.CoolProp import PropsSI
except:
    PropsSI = None

# ------------------ Constants ------------------
P_ATM = 101325.0
CP_DA = 1006.0
CP_V = 1860.0
H_LV = 2501000.0
R_DA = 287.055
INCH = 0.0254
MM = 1e-3

st.set_page_config(page_title="DX Evaporator – ε–NTU", layout="wide")

# ------------------ Psychrometrics ------------------
def K(T): return T + 273.15

def psat(T):
    return 611.21 * math.exp((18.678 - T/234.5) * (T/(257.14 + T)))

def W_from_T_RH(T, RH):
    Pv = psat(T) * RH / 100
    return 0.62198 * Pv / (P_ATM - Pv)

def h_air(T, W):
    return 1.006*T*1000 + W*(H_LV + 1.86*T*1000)

def rho_air(T, W):
    return P_ATM / (R_DA * K(T) * (1 + 1.6078*W))

def RH_from_T_W(T, W):
    Pv = W*P_ATM/(0.62198+W)
    return 100*Pv/psat(T)

# ------------------ Core Solver ------------------
def solve_evaporator(
    face_W, face_H, rows,
    Do, pitch, tf, FPI,
    Vdot, Tdb_in, W_in,
    Tdb_req, W_req,
    fluid, Tsat, SH, mdot_ref, x_in
):
    # Required cooling
    rho = rho_air(Tdb_in, W_in)
    mdot_da = rho * Vdot / (1 + W_in)
    Q_required = mdot_da * (h_air(Tdb_in,W_in) - h_air(Tdb_req,W_req))

    # Refrigerant capacity
    TsK = K(Tsat)
    h_fg = PropsSI("H","T",TsK,"Q",1,fluid) - PropsSI("H","T",TsK,"Q",0,fluid)
    cp_v = PropsSI("C","T",TsK,"Q",1,fluid)

    Q_evap_max = mdot_ref * h_fg * (1 - x_in)
    Q_SH_max   = mdot_ref * cp_v * SH
    Q_ref_max = Q_evap_max + Q_SH_max

    # Coil surface enthalpy (saturated at Tsat)
    W_sat = W_from_T_RH(Tsat, 100)
    h_surf = h_air(Tsat, W_sat)

    # NTU-based bypass factor (simple, stable)
    UA = 50 * face_W * face_H * rows     # conservative U
    BF = math.exp(-UA / (mdot_da * CP_DA))

    h_out_air = h_surf + BF*(h_air(Tdb_in,W_in) - h_surf)
    Q_air_max = mdot_da * (h_air(Tdb_in,W_in) - h_out_air)

    # Final achieved
    Q_achieved = min(Q_air_max, Q_ref_max)

    T_out = max(Tsat, Tdb_in - Q_achieved/(mdot_da*CP_DA))
    W_out = W_sat if T_out <= Tsat else W_in
    RH_out = RH_from_T_W(T_out, W_out)

    insuff = []
    if Q_achieved < Q_required:
        insuff.append("Capacity shortfall")
    if Q_SH_max < mdot_ref * cp_v * SH:
        insuff.append("Superheat shortfall")

    return {
        "Required cooling (kW)": Q_required/1000,
        "Achieved cooling (kW)": Q_achieved/1000,
        "Leaving air DB (°C)": T_out,
        "Leaving air RH (%)": RH_out,
        "Refrigerant max capacity (kW)": Q_ref_max/1000,
        "Limitation": "None" if not insuff else ", ".join(insuff)
    }

# ------------------ UI ------------------
st.title("DX Evaporator – Wet Coil ε–NTU (v3)")

face_W = st.number_input("Face width (m)", 1.2)
face_H = st.number_input("Face height (m)", 0.85)
rows = st.number_input("Rows", 4, step=1)

Do = st.number_input("Tube OD (mm)", 9.53)*MM
pitch = st.number_input("Tube pitch (mm)", 22.0)*MM
tf = st.number_input("Fin thickness (mm)", 0.12)*MM
FPI = st.number_input("FPI", 10.0)

Vdot = st.number_input("Airflow (m³/s)", 2.0)

Tdb_in = st.number_input("Air in DB (°C)", 24.0)
RH_in = st.number_input("Air in RH (%)", 50.0)
W_in = W_from_T_RH(Tdb_in, RH_in)

Tdb_req = st.number_input("Required DB out (°C)", 14.0)
RH_req = st.number_input("Required RH out (%)", 95.0)
W_req = W_from_T_RH(Tdb_req, RH_req)

fluid = st.selectbox("Refrigerant", ["R134a","R410A","R407C"])
Tsat = st.number_input("Evaporating Tsat (°C)", 5.0)
SH = st.number_input("Superheat (K)", 6.0)
mdot_ref = st.number_input("Refrigerant mass flow (kg/s)", 0.08)
x_in = st.number_input("Refrigerant inlet quality x_in", 0.25)

if st.button("Run calculation"):
    try:
        result = solve_evaporator(
            face_W, face_H, rows,
            Do, pitch, tf, FPI,
            Vdot, Tdb_in, W_in,
            Tdb_req, W_req,
            fluid, Tsat, SH, mdot_ref, x_in
        )
        st.json(result)
    except Exception as e:
        st.error(str(e))
        st.code(traceback.format_exc())
