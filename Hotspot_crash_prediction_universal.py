
# Universal Crash Hotspot Prediction Script
# Trained on both US & UK Kaggle datasets


import os #used for building file path and  reading environment in Kaggle
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report


# Importing the Dataset

choice = input("Choose dataset (US/UK): ").strip().lower()

if choice == "us":
    dataset_name = "sobhanmoosavi/us-accidents"
elif choice == "uk":
    dataset_name = "tsiaras/uk-road-safety-accidents-and-vehicles"
else:
    raise ValueError("Invalid choice. Please enter 'US' or 'UK'.")

print(f" Downloading dataset: {dataset_name} ...")
path = kagglehub.dataset_download(dataset_name)
print("Dataset downloaded to:", path)


# Loading the  Dataset

if choice == "us":
    data_path = os.path.join(path, "US_Accidents_March23.csv")
    data = pd.read_csv(data_path, low_memory=False)
    print(" US dataset loaded. Shape:", data.shape)

    # Features for US in which  the Model has learned 
    features = [
        "Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Amenity", "Bump", "Crossing", "Junction", "Traffic_Signal"
    ]
    target = "Severity"

    df = data[features + [target]].dropna()

    # Convert from booleans to integers
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    model = RandomForestRegressor(
        n_estimators=50, max_depth=15, random_state=42, n_jobs=-1
    )
    X = df.drop(columns=[target])
    y = df[target]

elif choice == "uk":
    data_path = os.path.join(path, "Accident_Information.csv")
    data = pd.read_csv(data_path, low_memory=False)
    print("UK dataset loaded. Shape:", data.shape)

    # Features for UK in which  the Model has learned
    features = [
        "Longitude", "Latitude", "Day_of_Week", "Time",
        "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"
    ]
    target = "Accident_Severity"

    df = data[features + [target]].dropna()

    # Convert from  time to hour
    def extract_hour(t):
        try:
            return int(str(t).split(":")[0])
        except:
            return np.nan
    df["Hour"] = df["Time"].apply(extract_hour)
    df = df.drop(columns=["Time"])

    #  encode categoricals features using  One-hot method
    df = pd.get_dummies(
        df,
        columns=["Day_of_Week", "Weather_Conditions",
                 "Road_Surface_Conditions", "Urban_or_Rural_Area"],
        drop_first=True
    )

    model = RandomForestClassifier(
        n_estimators=50, max_depth=15, random_state=42, n_jobs=-1
    )
    X = df.drop(columns=[target, "Longitude", "Latitude"])
    y = df[target]


# Split the data part for train and part for test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if "uk" in dataset_name else None
)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# Train the Model

print("Training model...")
model.fit(X_train, y_train)
print("Training complete.")

if "us-accidents" in dataset_name:
    y_pred = model.predict(X_test)
    print("Sample Prediction:", y_pred[:5])
else:
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Generate Predictions

sample_size = min(5000, len(df))  # smaller sample for speed
sample = df.sample(sample_size, random_state=42).copy()

if "us-accidents" in dataset_name:
    sample["predicted_risk"] = model.predict(sample.drop(columns=[target]))
    sample.rename(columns={"Start_Lat": "Start_Lat", "Start_Lng": "Start_Lng"}, inplace=True)

elif "uk-road-safety" in dataset_name:
    probs = model.predict_proba(sample.drop(columns=[target, "Longitude", "Latitude"]))
    if 1 in model.classes_:
        fatal_idx = list(model.classes_).index(1)
        sample["predicted_risk"] = probs[:, fatal_idx]
    else:
        sample["predicted_risk"] = probs.max(axis=1)
    sample.rename(columns={"Latitude": "Start_Lat", "Longitude": "Start_Lng"}, inplace=True)


# Adding Fake Interventions

rng = np.random.default_rng(42)
sample["segment_id"] = [f"seg_{i}" for i in range(len(sample))]
sample["intervention"] = rng.choice(["lighting", "speed_bump", "signage"], len(sample))
sample["cost"] = rng.choice([20000, 30000, 50000], len(sample))
sample["effectiveness"] = rng.uniform(0.1, 0.3, len(sample))

predictions = sample[[
    "segment_id", "Start_Lat", "Start_Lng",
    "predicted_risk", "intervention", "cost", "effectiveness"
]]

#  Save Predictions

output_path = os.path.join(os.path.dirname(__file__), "predictions.csv")
predictions.to_csv(output_path, index=False)

print("Saved predictions.csv")
print("File path:", os.path.abspath(output_path))

