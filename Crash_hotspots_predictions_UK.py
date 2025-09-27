# ============================================================
# UK Crash Hotspot Prediction with Budget Optimization
# ============================================================

# Step 1 - Imports
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import math

# ============================================================
# Step 2 - Download Dataset with kagglehub
# ============================================================
print("⬇️ Downloading UK dataset from Kaggle via kagglehub...")
path = kagglehub.dataset_download("tsiaras/uk-road-safety-accidents-and-vehicles")
print("✅ Dataset downloaded to:", path)

# Main accidents file
data_path = os.path.join(path, "Accidents0515.csv")

# ============================================================
# Step 3 - Load Dataset (Accident_Information.csv)
# ============================================================
data_path = os.path.join(path, "Accident_Information.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Expected Accident_Information.csv in {path}, but not found.")

data = pd.read_csv(data_path, low_memory=False)
print("✅ Accident dataset loaded. Shape:", data.shape)
print("📑 Columns:", list(data.columns)[:20], "...")


# ============================================================
# Step 4 - Feature Selection & Cleaning
# ============================================================
features = [
    "Longitude", "Latitude",
    "Day_of_Week", "Time", "Weather_Conditions",
    "Road_Surface_Conditions", "Urban_or_Rural_Area"
]
target = "Accident_Severity"  # 1 = fatal, 2 = serious, 3 = slight

df = data[features + [target]].dropna()

# Convert time to hour-of-day
def extract_hour(t):
    try:
        return int(str(t).split(":")[0])
    except:
        return np.nan

df["Hour"] = df["Time"].apply(extract_hour)
df = df.drop(columns=["Time"])

# One-hot encode categoricals
df = pd.get_dummies(df, columns=["Day_of_Week", "Weather_Conditions", 
                                 "Road_Surface_Conditions", "Urban_or_Rural_Area"], drop_first=True)

print("✅ Features engineered. Shape:", df.shape)

# ============================================================
# Step 5 - Train-Test Split
# ============================================================
X = df.drop(columns=[target, "Longitude", "Latitude"])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fill NaNs if any
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ============================================================
# Step 6 - Train Model
# ============================================================
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model training complete.")


# ============================================================
# Step 7 - Evaluation
# ============================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("📊 Evaluation Results:")
print(f"Accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred))

# ============================================================
# Step 8 - Budget Optimization Function
# ============================================================
def optimize_investment(pred_df, budget):
    """
    Greedy budget allocation:
    Selects locations/interventions with the best
    risk reduction per dollar until budget is used.
    """
    dfc = pred_df.copy()
    dfc["benefit"] = dfc["predicted_risk"] * dfc["effectiveness"]
    dfc["ratio"] = dfc["benefit"] / dfc["cost"]

    dfc = dfc.sort_values("ratio", ascending=False)

    total_cost, selected = 0, []
    for _, row in dfc.iterrows():
        if total_cost + row["cost"] <= budget:
            selected.append(row)
            total_cost += row["cost"]

    selected_df = pd.DataFrame(selected)
    total_benefit = selected_df["benefit"].sum() if not selected_df.empty else 0

    return {
        "budget": budget,
        "total_cost": total_cost,
        "total_benefit": total_benefit,
        "selected_interventions": selected_df
    }

# ============================================================
# Step 9 - Generate Predictions for Interventions
# ============================================================
sample = df.sample(2000, random_state=42).copy()

# Predicted risk = probability of "fatal" accident
probs = model.predict_proba(sample.drop(columns=[target, "Longitude", "Latitude"]))
if 1 in model.classes_:
    fatal_idx = list(model.classes_).index(1)
    sample["predicted_risk"] = probs[:, fatal_idx]
else:
    sample["predicted_risk"] = probs.max(axis=1)

# Add fake interventions
rng = np.random.default_rng(42)
sample["segment_id"] = [f"seg_{i}" for i in range(len(sample))]
sample["intervention"] = rng.choice(["lighting", "speed_bump", "signage"], len(sample))
sample["cost"] = rng.choice([20000, 30000, 50000], len(sample))
sample["effectiveness"] = rng.uniform(0.1, 0.3, len(sample))

predictions = sample.rename(columns={
    "Latitude": "Start_Lat",
    "Longitude": "Start_Lng"
})[[
    "segment_id", "Start_Lat", "Start_Lng",
    "predicted_risk", "intervention", "cost", "effectiveness"
]]

# Save predictions
output_path = os.path.join(os.path.dirname(__file__), "predictions_uk.csv")
predictions.to_csv(output_path, index=False)

print("✅ Saved predictions_uk.csv for Streamlit map.")
print("📂 File path:", os.path.abspath(output_path))

# ============================================================
# Step 10 - Example Usage
# ============================================================
decision = optimize_investment(predictions, budget=200000)
print("💡 Example Optimization Result:")
print("Budget:", decision["budget"])
print("Total Cost:", decision["total_cost"])
print("Total Benefit (risk reduction):", decision["total_benefit"])
print("Selected sites:", len(decision["selected_interventions"]))
