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
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found at {data_path}")

data = pd.read_csv(data_path, low_memory=False)
print("✅ Raw dataset loaded. Shape:", data.shape)

# ============================================================
# Step 4 - Feature Selection & Cleaning (fixed)
# ============================================================
features = [
    "Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
    "Amenity", "Bump", "Crossing", "Junction", "Traffic_Signal"
]
target = "Severity"

# Keep only available columns
features = [f for f in features if f in data.columns]

df = data[features + [target]].dropna()

# Convert booleans → ints
for col in df.select_dtypes(include=["bool"]).columns:
    df[col] = df[col].astype(int)

# 💡 Sample down to 50k rows for training (fast + prevents memory crash)
df = df.sample(min(50000, len(df)), random_state=42)

print("✅ Features cleaned & sampled. Shape:", df.shape)

# ============================================================
# Step 5 - Train-Test Split
# ============================================================
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ensure no NaNs
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ============================================================
# Step 6 - Train Model (with safe params)
# ============================================================
model = RandomForestRegressor(
    n_estimators=30,      # fewer trees for speed
    max_depth=12,         # prevent overfitting + memory blowup
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# ============================================================
# Step 7 - Evaluation
# ============================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print("📊 Evaluation Results:")
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
sample = df.sample(500, random_state=42).copy()
sample["predicted_risk"] = model.predict(sample.drop(columns=[target]))

rng = np.random.default_rng(42)
sample["segment_id"] = [f"seg_{i}" for i in range(len(sample))]
sample["intervention"] = rng.choice(["lighting", "speed_bump", "signage"], len(sample))
sample["cost"] = rng.choice([20000, 30000, 50000], len(sample))
sample["effectiveness"] = rng.uniform(0.1, 0.3, len(sample))

predictions = sample[[
    "segment_id", "Start_Lat", "Start_Lng",
    "predicted_risk", "intervention", "cost", "effectiveness"
]]

# Save to same folder as script with absolute path
output_path = os.path.join(os.path.dirname(__file__), "predictions.csv")
predictions.to_csv(output_path, index=False)

print("✅ Saved predictions.csv for Streamlit map.")
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
