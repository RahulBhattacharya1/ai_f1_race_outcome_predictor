import os
import re
import math
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------
# Utility
# --------------------------
@st.cache_data(show_spinner=False)
def load_lookups():
    assets_dir = "assets"
    drivers = pd.read_csv(os.path.join(assets_dir, "drivers_lookup.csv"))
    constructors = pd.read_csv(os.path.join(assets_dir, "constructors_lookup.csv"))
    circuits = pd.read_csv(os.path.join(assets_dir, "circuits_lookup.csv"))
    return drivers, constructors, circuits

def time_to_seconds(s):
    if s is None or str(s).strip() == "":
        return np.nan
    s = str(s).strip()
    if re.match(r"^\d+:\d+(\.\d+)?$", s):
        mm, ss = s.split(":")
        return int(mm) * 60 + float(ss)
    try:
        return float(s)
    except:
        return np.nan

@st.cache_resource(show_spinner=False)
def load_model():
    path = "models/race_outcome_pipeline.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Missing models/race_outcome_pipeline.joblib")
    return joblib.load(path)

# --------------------------
# Page
# --------------------------
st.set_page_config(page_title="F1 Podium Predictor", layout="centered")

st.title("F1 Podium Predictor (Top-3 vs Not)")
st.write(
    "Predict the probability of a podium finish using qualifying time, grid, recent form, and context. "
    "Select known entries or type your own values. Unseen categories are handled safely."
)

# Load resources
pipe = load_model()
drivers, constructors, circuits = load_lookups()

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("Model: scikit-learn Pipeline + LogisticRegression (class-weighted).")
    st.write("Lookups are tiny CSVs shipped with the app.")
    st.write("Qualifying time can be 'M:SS.sss' (e.g., 1:23.456) or seconds (e.g., 83.456).")

# --------------------------
# Inputs
# --------------------------
st.subheader("Inputs")

# Year and round
col_yr, col_rd = st.columns(2)
year = col_yr.number_input("Season (year)", min_value=1950, max_value=2100, value=2024, step=1)
round_num = col_rd.number_input("Round", min_value=1, max_value=30, value=1, step=1)

# Entities
col_d, col_c = st.columns(2)
driver_choice = col_d.selectbox(
    "Driver",
    options=drivers["driver_name"].tolist(),
    index=0
)
constructor_choice = col_c.selectbox(
    "Constructor",
    options=constructors["name"].tolist(),
    index=0
)
circuit_choice = st.selectbox(
    "Circuit",
    options=circuits["name"].tolist(),
    index=0
)

# Map to IDs expected by the pipeline
driver_row = drivers.loc[drivers["driver_name"] == driver_choice].iloc[0]
constructor_row = constructors.loc[constructors["name"] == constructor_choice].iloc[0]
circuit_row = circuits.loc[circuits["name"] == circuit_choice].iloc[0]

driverId = int(driver_row["driverId"])
constructorId = int(constructor_row["constructorId"])
circuitId = int(circuit_row["circuitId"])

# Grid and qualifying time
col_g, col_q = st.columns(2)
grid = int(col_g.number_input("Grid Position (1 = pole)", min_value=1, max_value=40, value=5, step=1))
quali_str = col_q.text_input("Best Qualifying Time (M:SS.sss or seconds)", value="1:23.500")
best_quali_sec = time_to_seconds(quali_str)
if math.isnan(best_quali_sec):
    st.warning("Enter qualifying time as 'M:SS.sss' or plain seconds.")
    best_quali_sec = 90.0

# Optional recent form
col_dr, col_cr = st.columns(2)
driver_recent_points_3 = float(col_dr.number_input("Driver recent points (last 3 races)", min_value=0.0, max_value=75.0, value=0.0, step=1.0))
constructor_recent_points_3 = float(col_cr.number_input("Constructor recent points (last 3 races)", min_value=0.0, max_value=100.0, value=0.0, step=1.0))

# --------------------------
# Predict
# --------------------------
if st.button("Predict Podium Probability"):
    row = {
        "grid": grid,
        "best_quali_sec": best_quali_sec,
        "driver_recent_points_3": driver_recent_points_3,
        "constructor_recent_points_3": constructor_recent_points_3,
        "round": round_num,
        "driverId": driverId,
        "constructorId": constructorId,
        "circuitId": circuitId,
        "year": int(year),
    }
    X = pd.DataFrame([row])
    proba = float(pipe.predict_proba(X)[:,1][0])
    pred_label = "Top-3" if proba >= 0.5 else "Not Top-3"

    st.success(f"Predicted probability of podium: {proba:.3f}")
    st.write(f"Predicted class: {pred_label}")

    # Simple guidance
    if proba >= 0.7:
        st.info("Strong podium chance based on inputs.")
    elif proba >= 0.4:
        st.info("Borderline podium chance; small swings in quali/grid can matter.")
    else:
        st.info("Unlikely podium; significant improvement needed in quali/grid.")

# Show feature info
with st.expander("Feature columns expected by the model"):
    st.code(json.dumps({
        "numeric": ["grid", "best_quali_sec", "driver_recent_points_3", "constructor_recent_points_3", "round"],
        "categorical": ["driverId", "constructorId", "circuitId", "year"]
    }, indent=2))
