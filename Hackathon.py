# ============================================================
# Predictive Maintenance with Cost Analysis
# ============================================================

# Step 1 - Imports
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ============================================================
# Step 2 - Download Dataset with kagglehub
# ============================================================
print("⬇️ Downloading dataset from Kaggle via kagglehub...")
path = kagglehub.dataset_download("bishals098/nasa-turbofan-engine-degradation-simulation")
print("✅ Dataset downloaded to:", path)

# Use the FD001 training file
data_path = os.path.join(path, "train_FD001.txt")

# ============================================================
# Step 3 - Load Dataset
# ============================================================
data = pd.read_csv(data_path, sep=" ", header=None)
data.dropna(axis=1, how="all", inplace=True)

# Assign column names
cols = ['unit_number', 'time_in_cycles'] \
       + [f'operational_setting_{i}' for i in range(1, 4)] \
       + [f'sensor_{i}' for i in range(1, 22)]

data.columns = cols

print("✅ Raw dataset loaded. Shape:", data.shape)

# ============================================================
# Step 4 - Add Remaining Useful Life (RUL)
# ============================================================
rul = data.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul.columns = ['unit_number', 'max_cycle']

data = data.merge(rul, on=['unit_number'], how='left')
data['RUL'] = data['max_cycle'] - data['time_in_cycles']
data.drop('max_cycle', axis=1, inplace=True)

print("✅ RUL column added.")
print(data[['unit_number', 'time_in_cycles', 'RUL']].head())

# ============================================================
# Step 5 - Normalize Sensor Data
# ============================================================
sensor_cols = [col for col in data.columns if "sensor" in col]

scaler = MinMaxScaler()
data[sensor_cols] = scaler.fit_transform(data[sensor_cols])

print("✅ Sensor columns normalized.")

# ============================================================
# Step 6 - Train-Test Split
# ============================================================
feature_cols = [c for c in data.columns if c not in ['unit_number', 'time_in_cycles', 'RUL']]
X = data[feature_cols]
y = data['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# Step 7 - Train Model
# ============================================================
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("✅ Model training complete.")

# ============================================================
# Step 8 - Evaluation
# ============================================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print(f"📊 Evaluation Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ============================================================
# Step 9 - Cost Analysis Function
# ============================================================
def compare_cost(rul, maintenance_cost=5000, replacement_cost=50000, downtime_cost=1000):
    """
    Compare cost of maintaining vs replacing a machine.
    
    rul: predicted remaining useful life (in cycles)
    maintenance_cost: cost per scheduled maintenance
    replacement_cost: cost of buying new machine
    downtime_cost: cost per cycle if machine fails
    """
    # Example assumption: maintenance is required every 50 cycles
    maint_intervals = max(1, rul // 50)
    total_maintenance_cost = maint_intervals * maintenance_cost
    
    # Risk of waiting too long → add downtime penalty if rul < 30 cycles
    if rul < 30:
        total_maintenance_cost += downtime_cost * (30 - rul)
    
    decision = "Maintain" if total_maintenance_cost < replacement_cost else "Replace"
    
    return {
        "predicted_RUL": int(rul),
        "maintenance_cost": total_maintenance_cost,
        "replacement_cost": replacement_cost,
        "recommendation": decision
    }

# ============================================================
# Step 10 - Example Usage
# ============================================================
sample_features = X_test.iloc[0].values.reshape(1, -1)
predicted_rul = model.predict(sample_features)[0]

decision = compare_cost(predicted_rul)
print("🔧 Decision Support Example:")
print(decision)

# ============================================================
