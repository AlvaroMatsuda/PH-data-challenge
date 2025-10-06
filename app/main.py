# ======= Imports =======
from fastapi import FastAPI
from make_prediction.make_prediction import predict_pipeline
from typing import List, Dict

# ======= Creating App =======
app = FastAPI()

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict")
async def predict_endpoint(payload: List[Dict]):

    preds = predict_pipeline(payload)

    return {"predictions": preds.astype(float).tolist()}