from typing import List

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


# FastAPI app instance
app = FastAPI()

# Load the trained model and scaler from disk
model = joblib.load("artifacts/model.joblib")

# Define input data structure
class InputData(BaseModel):
    data: List[float]


@app.post("/api/v1.0/predict")
def predict(input_data: InputData):
    df = pd.DataFrame([input_data.data])
    print(df.head())

    prediction = model.predict(df)

    return {"prediction": prediction.tolist()}
