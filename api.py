from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the trained model
MODEL_PATH = "best_rf_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

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