# ========== Imports ==========
import json
import pickle
from pathlib import Path
import pandas as pd


# ========== Variables ==========
BASE_DIR = Path(__file__).resolve(strict=True).parent
DEMOGRAPHICS_PATH = "../data/zipcode_demographics.csv"
MODEL_FEATURE_PATH = "../model/model_features.json"

# ========= Loading Trained Model =========
with open(f"{BASE_DIR}/../model/model.pkl", "rb") as f:
    model = pickle.load(f)

# ========= Loading Model Features List =========
with open(f"{BASE_DIR}/{MODEL_FEATURE_PATH}") as f:
    model_features = json.load(f)

# ========= Predict Function =========
def predict_pipeline(new_data):

    # Load payload data as Pandas DataFrame
    df_new = pd.DataFrame(new_data)
    df_new["zipcode"] = df_new["zipcode"].astype(str)


    # Load demographic data
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})

    # Join Demographic Data
    merged_data = df_new.merge(demographics, how="left", on="zipcode")

    # Selecting only necessary columns for prediction
    merged_data = merged_data.loc[:, model_features]

    # Making predictions
    pred = model.predict(merged_data)

    return pred

