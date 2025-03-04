api.py

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from config import MODEL_PATH
from pydantic import BaseModel

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError("Model file not found! Ensure model training is completed before running the API.")

# Initialize FastAPI app
app = FastAPI()

# Define request model
class CompanyFeatures(BaseModel):
    features: list[float]

@app.post("/predict/")
def predict(features: CompanyFeatures):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([features.features])
        prediction = model.predict(df)[0]
        return {"bankrupt": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "Corporate Bankruptcy Prediction API"}