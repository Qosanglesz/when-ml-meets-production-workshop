from typing import List

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


# FastAPI app instance
app = FastAPI()

# Load the trained model and scaler from disk
model_v1 = joblib.load("artifacts/model_v1.joblib")
model_v2 = joblib.load("artifacts/model_v2.joblib")

# Define input data structure
class InputData(BaseModel):
    data: List[float]


@app.post("/api/v1.0/predict")
def predict(input_data: InputData):
    df = pd.DataFrame([input_data.data])
    print(df.head())

    prediction = model_v1.predict(df)

    return {"prediction": prediction.tolist()}

@app.post("/api/v1.1/predict")
def predict(input_data: InputData):
    df = pd.DataFrame([input_data.data])
    print(df.head())

    prediction = model_v2.predict(df)

    return {"prediction": prediction.tolist()}