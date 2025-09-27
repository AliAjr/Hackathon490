# ============================================================
# Crash Hotspot Planner - Streamlit Interface (Auto-generate CSV Version)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import subprocess

# ============================================================
# Step 1 - Load or Generate Predictions
# ============================================================
st.set_page_config(page_title="Crash Hotspot Planner", layout="wide")
st.title("🚦 Crash Hotspot Planner")

csv_path = "predictions.csv"

# Auto-generate predictions if not found
if not os.path.exists(csv_path):
    try:
        st.info("⚡ Generating predictions.csv (this may take a moment)...")
        subprocess.run(
            ["python", "Hotspot_crash_prediction_universal.py"],
            check=True
        )
    except Exception as e:
        st.error(f"❌ Could not generate predictions.csv automatically: {e}")
        st.stop()

# Load predictions
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success(f"✅ Loaded {len(df)} rows from predictions.csv")
else:
    st.error("❌ predictions.csv not found and could not be generated.")
    st.stop()

# Ensure required columns
required = {"segment_id", "Start_Lat", "Start_Lng", "predicted_risk", "intervention", "cost", "effectiveness"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ============================================================
# Step 2 - User Input (Budget)
# ============================================================
budget = st.sidebar.number_input("Available Budget ($)", min_value=10000, value=200000, step=10000)

# ============================================================
# Step 3 - Precompute Risk, Effectiveness, Reason
# ============================================================
df["benefit"] = df["predicted_risk"] * df["effectiveness"]
df["ratio"] = df["benefit"] / df["cost"]

def risk_label(score):
    if score < df["predicted_risk"].quantile(0.33):
        return "Low Risk"
    elif score < df["predicted_risk"].quantile(0.66):
        return "Medium Risk"
    else:
        return "High Risk"
df["Risk_Level"] = df["predicted_risk"].apply(risk_label)

df["Effectiveness_Label"] = (df["effectiveness"] * 100).round(0).astype(int).astype(str) + "% reduction"

def explain_reason(row):
    if row["Risk_Level"] == "High Risk":
        reason = "Selected because this area has a high crash risk. "
    elif row["Risk_Level"] == "Medium Risk":
        reason = "Selected as a preventive measure for moderate crash risk. "
    else:
        reason = "Chosen as a low-cost preventive action in a low-risk area. "

    if row["intervention"] == "lighting":
        reason += "Street lighting improves visibility at night and during poor weather. "
    elif row["intervention"] == "speed_bump":
        reason += "Speed bumps slow down vehicles in accident-prone zones. "
    elif row["intervention"] == "signage":
        reason += "Road signage warns drivers and improves awareness. "

    return reason.strip()
df["Reason"] = df.apply(explain_reason, axis=1)

# ============================================================
# Step 4 - Optimization Function
# ============================================================
def optimize_investment(df, budget):
    dfc = df.sort_values("ratio", ascending=False).copy()
    total_cost, selected = 0, []
    for _, row in dfc.iterrows():
        if total_cost + row["cost"] <= budget:
            selected.append(row)
            total_cost += row["cost"]
    return pd.DataFrame(selected), total_cost

selected, total_cost = optimize_investment(df, budget)

projected_before = df["predicted_risk"].sum()
projected_after = projected_before - selected["benefit"].sum() if not selected.empty else projected_before
percent_reduction = 100 * (1 - projected_after / projected_before) if projected_before > 0 else 0

if total_cost < budget and not selected.empty:
    st.info("💡 Budget exceeds all available interventions. No further gains possible.")

# ============================================================
# Step 5 - KPIs
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Budget", f"${budget:,.0f}")
c2.metric("Allocated", f"${total_cost:,.0f}")
c3.metric("Projected Harm (before)", f"{projected_before:.1f}")
c4.metric("Projected Harm (after)", f"{projected_after:.1f}")
st.metric("% Reduction", f"{percent_reduction:.1f}%")

# ============================================================
# Step 6 - Map
# ============================================================
st.subheader("Map of Predicted Hotspots and Interventions")

base_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[Start_Lng, Start_Lat]',
    get_radius=80,
    get_fill_color="[200, 30, 30, 160]",
    pickable=True,
)

highlight_layer = None
if not selected.empty:
    highlight_layer = pdk.Layer(
        "ScatterplotLayer",
        data=selected,
        get_position='[Start_Lng, Start_Lat]',
        get_radius=150,
        get_fill_color="[0, 200, 50, 220]",
        pickable=True,
    )

view_state = pdk.ViewState(
    latitude=float(df["Start_Lat"].mean()),
    longitude=float(df["Start_Lng"].mean()),
    zoom=7,
    pitch=0
)

tooltip = {
    "html": (
        "<b>Segment:</b> {segment_id}<br/>"
        "<b>Risk Level:</b> {Risk_Level}<br/>"
        "<b>Added Intervention:</b> {intervention}<br/>"
        "<b>Reason:</b> {Reason}<br/>"
        "<b>Cost:</b> ${cost}<br/>"
        "<b>Effectiveness:</b> {Effectiveness_Label}"
    ),
    "style": {"backgroundColor": "steelblue", "color": "white"}
}

layers = [base_layer] + ([highlight_layer] if highlight_layer is not None else [])
st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip=tooltip
))

# ============================================================
# Step 7 - Tables
# ============================================================
st.subheader("Selected Interventions")
if selected.empty:
    st.write("No interventions selected under the current budget.")
else:
    st.dataframe(selected[[
        "segment_id", "Risk_Level", "intervention", "cost", "Effectiveness_Label", "Reason"
    ]].reset_index(drop=True))

st.subheader("All Candidates (ranked by cost-effectiveness)")
st.dataframe(df.sort_values("ratio", ascending=False)[[
    "segment_id", "Risk_Level", "intervention", "cost", "Effectiveness_Label", "Reason"
]].reset_index(drop=True))
