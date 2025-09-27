# ============================================================
# Crash Hotspot Prediction with Budget Optimization
# ============================================================

# Step 1 - Imports
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ============================================================
# Step 2 - Download Dataset with kagglehub
# ============================================================
print("⬇️ Downloading dataset from Kaggle via kagglehub...")
path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
print("✅ Dataset downloaded to:", path)

# Dataset file
data_path = os.path.join(path, "US_Accidents_March23.csv")

# ============================================================
# Step 3 - Load Dataset
# ============================================================
data = pd.read_csv(data_path)
print("✅ Raw dataset loaded. Shape:", data.shape)

# ============================================================
# Step 4 - Feature Selection & Cleaning
# ============================================================
# Use subset of features
features = [
    "Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
    "Amenity", "Bump", "Crossing", "Junction", "Traffic_Signal"
]
target = "Severity"

df = data[features + [target]].dropna()

# Convert booleans to integers
for col in df.select_dtypes(include=["bool"]).columns:
    df[col] = df[col].astype(int)

print("✅ Features cleaned. Shape:", df.shape)

# ============================================================
# Step 5 - Train-Test Split
# ============================================================
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# Step 6 - Train Model
# ============================================================
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# ============================================================
# Step 7 - Evaluation
# ============================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print(f"📊 Evaluation Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ============================================================
# Step 8 - Budget Optimization Function
# ============================================================
def optimize_investment(pred_df, budget):
    """
    Greedy budget allocation:
    Selects locations/interventions with the best
    risk reduction per dollar until budget is used.
    """
    pred_df = pred_df.copy()
    pred_df["benefit"] = pred_df["predicted_risk"] * pred_df["effectiveness"]
    pred_df["ratio"] = pred_df["benefit"] / pred_df["cost"]

    pred_df = pred_df.sort_values("ratio", ascending=False)
    
    total_cost, selected = 0, []
    for _, row in pred_df.iterrows():
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
sample = df.sample(500, random_state=42).copy()
sample["predicted_risk"] = model.predict(sample.drop(columns=[target]))

# Add dummy interventions
rng = np.random.default_rng(42)
sample["segment_id"] = [f"seg_{i}" for i in range(len(sample))]
sample["intervention"] = rng.choice(["lighting", "speed_bump", "signage"], len(sample))
sample["cost"] = rng.choice([20000, 30000, 50000], len(sample))
sample["effectiveness"] = rng.uniform(0.1, 0.3, len(sample))

predictions = sample[["segment_id", "Start_Lat", "Start_Lng",
                      "predicted_risk", "intervention", "cost", "effectiveness"]]

predictions.to_csv("predictions.csv", index=False)
print("✅ Saved predictions.csv for Streamlit map.")

# ============================================================
# Step 10 - Example Usage
# ============================================================
decision = optimize_investment(predictions, budget=200000)
print("💡 Example Optimization Result:")
print("Budget:", decision["budget"])
print("Total Cost:", decision["total_cost"])
print("Total Benefit (risk reduction):", decision["total_benefit"])
print("Selected sites:", len(decision["selected_interventions"]))
