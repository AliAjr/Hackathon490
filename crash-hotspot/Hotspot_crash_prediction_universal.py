# ============================================================
# Universal Crash Hotspot Prediction Script (Interactive Version)
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Step 1 - Ask User for Dataset Choice
# ============================================================
choice = input("Choose dataset (US/UK): ").strip().lower()

if choice == "us":
    filename = "US_Accidents_March23.csv"
elif choice == "uk":
    filename = "Accident_Information.csv"
else:
    raise ValueError("Invalid choice. Please enter 'US' or 'UK'.")

data_path = os.path.join(os.path.dirname(__file__), filename)
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Could not find {filename}. Place it in the project folder.")

print(f"📂 Loading {filename} ...")
df_raw = pd.read_csv(data_path, low_memory=False)

# ============================================================
# Step 2 - Feature selection
# ============================================================
if choice == "us":
    features = [
        "Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Amenity", "Bump", "Crossing", "Junction", "Traffic_Signal"
    ]
    target = "Severity"

    df = df_raw[features + [target]].dropna()
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    X = df.drop(columns=[target])
    y = df[target]

elif choice == "uk":
    features = ["Longitude", "Latitude", "Day_of_Week", "Time",
                "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"]
    target = "Accident_Severity"

    df = df_raw[features + [target]].dropna()

    def extract_hour(t):
        try:
            return int(str(t).split(":")[0])
        except:
            return np.nan
    df["Hour"] = df["Time"].apply(extract_hour)
    df = df.drop(columns=["Time"])

    df = pd.get_dummies(
        df,
        columns=["Day_of_Week", "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"],
        drop_first=True
    )

    model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    X = df.drop(columns=[target, "Longitude", "Latitude"])
    y = df[target]

# ============================================================
# Step 3 - Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if choice == "uk" else None
)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print("⚡ Training model ...")
model.fit(X_train, y_train)
print("✅ Training complete")

# ============================================================
# Step 4 - Generate predictions
# ============================================================
sample_size = min(2000, len(df))
sample = df.sample(sample_size, random_state=42).copy()

if choice == "us":
    sample["predicted_risk"] = model.predict(sample.drop(columns=[target]))
    sample.rename(columns={"Start_Lat": "Start_Lat", "Start_Lng": "Start_Lng"}, inplace=True)

elif choice == "uk":
    probs = model.predict_proba(sample.drop(columns=[target, "Longitude", "Latitude"]))
    if 1 in model.classes_:
        fatal_idx = list(model.classes_).index(1)
        sample["predicted_risk"] = probs[:, fatal_idx]
    else:
        sample["predicted_risk"] = probs.max(axis=1)
    sample.rename(columns={"Latitude": "Start_Lat", "Longitude": "Start_Lng"}, inplace=True)

# Fake interventions
rng = np.random.default_rng(42)
sample["segment_id"] = [f"seg_{i}" for i in range(len(sample))]
sample["intervention"] = rng.choice(["lighting", "speed_bump", "signage"], len(sample))
sample["cost"] = rng.choice([20000, 30000, 50000], len(sample))
sample["effectiveness"] = rng.uniform(0.1, 0.3, len(sample))

predictions = sample[[
    "segment_id", "Start_Lat", "Start_Lng",
    "predicted_risk", "intervention", "cost", "effectiveness"
]]

# ============================================================
# Step 5 - Save predictions
# ============================================================
output_path = os.path.join(os.path.dirname(__file__), "predictions.csv")
predictions.to_csv(output_path, index=False)

print("✅ Saved predictions.csv")
print("📂 File path:", output_path)
