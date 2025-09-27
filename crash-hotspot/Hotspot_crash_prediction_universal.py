# ============================================================
# Universal Crash Hotspot Prediction Script (Deployment Ready)
# ============================================================

import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# ============================================================
# Step 1 - Dataset Choice (via env var)
# ============================================================
choice = os.getenv("DATASET", "US").lower()

if choice == "us":
    dataset_name = "sobhanmoosavi/us-accidents"
    filename = "US_Accidents_March23.csv"
elif choice == "uk":
    dataset_name = "tsiaras/uk-road-safety-accidents-and-vehicles"
    filename = "Accident_Information.csv"
else:
    raise ValueError("❌ Invalid dataset choice. Use DATASET=US or DATASET=UK")

print(f"⬇️ Downloading dataset: {dataset_name}")
path = kagglehub.dataset_download(dataset_name)
print("✅ Dataset downloaded to:", path)

# ============================================================
# Step 2 - Load dataset
# ============================================================
data_path = os.path.join(path, filename)
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Expected file not found: {data_path}")

print(f"📂 Loading {filename} ...")
df_raw = pd.read_csv(data_path, low_memory=False)

# ============================================================
# Step 3 - Feature selection
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

    model = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
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

    model = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
    X = df.drop(columns=[target, "Longitude", "Latitude"])
    y = df[target]

# ============================================================
# Step 4 - Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if choice == "uk" else None
)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ============================================================
# Step 5 - Train model
# ============================================================
print("⚡ Training model ...")
model.fit(X_train, y_train)
print("✅ Training complete")

# ============================================================
# Step 6 - Generate predictions
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

# Fake interventions for demo
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
# Step 7 - Save predictions
# ============================================================
output_path = os.path.join(os.getcwd(), "predictions.csv")
predictions.to_csv(output_path, index=False)

print("✅ Saved predictions.csv")
print("📂 File path:", output_path)
