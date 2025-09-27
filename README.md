# Hackathon490
FA-2025/2026 AUB EECE 490 Hackathon assigned by Prof. Ammar Mhanna

Crash Hotspot Planner

A lightweight prototype to spot road crash hotspots and plan cost-effective safety fixes all on an interactive map.

It includes:
- A Streamlit app (`interface.py`) with a simple budget picker and a PyDeck map.
- A data script (`Hotspot_crash_prediction_universal.py`) that downloads a public crash dataset (US or UK), trains a quick baseline model, and exports a `predictions.csv` file for the app.

The business problem

Cities want fewer crashes but budgets are tight. This tool helps answer:
- Where are the highest-risk locations?
- What interventions (lighting, speed bumps, signage) give the most risk reduction per dollar?
- How do we pick a set of fixes that fits a fixed budget?

We keep it simple: estimate risk, estimate impact, then choose the best mix within your budget.


Dataset and usage

The universal script pulls one of two public datasets via `kagglehub` and prepares a standardized `predictions.csv` for the app.

Choose at runtime:
- US: `sobhanmoosavi/us-accidents` (Kaggle)
- UK: `tsiaras/uk-road-safety-accidents-and-vehicles` (Kaggle)


What the app expects in `predictions.csv`:
  -column"segment_id";type"string or int";description"road segment identifier"
  -column"Start_Lat";type"float";description"Latitude"
  -column"Start_Lng";type"float";description"Longitude"
  -column"predicted_risk";type"float";description"model estimated relative crash risk"
  -column"intervention";type"string";description"one of:lighting, speed bump,signage"
  -column"cost";type"decimal";description"estimated cost"
  -column"effectiveness";type"float";description"expected fractional risk reduction if the intervention is used"



Approach and architecture

How it works:
1. Data to Features
   The script downloads the chosen dataset and builds a minimal feature set.
2. Risk scoring
   A fast baseline model (Random Forest) estimates a relative crash risk score per segment.
3. Interventions & impact
   Each candidate gets a proposed intervention, cost and expected effectiveness.
4. Budget selection
   In the app, items are ranked by benefit/cost where `benefit = predicted_risk × effectiveness`.  
5. Visualization & insight  
   - Map: all candidates in red; selected ones highlighted in green.  
   - Summary: total spend and projected risk reduction.  
   - Table: chosen segments with a short human explanation (“Reason”).

Key files
- `Hotspot_crash_prediction_universal.py` — downloads data, trains a quick model, writes `predictions.csv`.
- `interface.py` — Streamlit app for upload, validation, selection, and map table views.


How to run / test
first, run the 'Hotspot_crash_prediction_universal.py' file to get a the predictions file. Then run the 'interface.py' file and drag the prediction file to get the result.

Librairies used:(all are in the requirements file)
-streamlit
-pandas
-numpy
-pydeck
-scikit-learn
-kagglehub

Deployed puiblic link: [https://hackathon490-uwfxjw7gqrebwcznh8vqnc.streamlit.app/](https://hackathon490-uwfxjw7gqrebwcznh8vqnc.streamlit.app/)
