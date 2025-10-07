# =======================
# Imports
# =======================
import pandas as pd
import requests
from pathlib import Path
import pytest


# =======================
# Variables
# =======================
BASE_DIR = Path(__file__).resolve(strict=True).parent
TEST_DATA_PATH = f"{BASE_DIR}/../data/future_unseen_examples.csv"

def test_api():
    # Loading test data
    df_payload = pd.read_csv(TEST_DATA_PATH)

    # Convert to Json
    payload = df_payload.to_json(orient="records")

    # Send Post request to the API and get the response
    r = requests.post("http://localhost:8000/predict", payload)

    # response
    df_payload.loc[:, "preds"] = r.json().get("predictions")

    # Check if status code of API request is 200 OK
    assert r.status_code == 200, f"API request failed with status code {r.status_code}."

    # Check if pred column is of type float
    assert df_payload["preds"].dtype == float, f"Wrong data Type of predictions column."

    # Check if there are null values in pred column
    assert df_payload["preds"].isna().any() == False, "There are Null values in pred column"

    print(df_payload)

if __name__ == "__main__":
    test_api()