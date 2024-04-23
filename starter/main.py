from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from starter.ml.model import inference
from joblib import load

class Data(BaseModel):
    features: List[float] = Field(..., example=[0.1, 0.2, 0.3, 0.4])

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to our machine learning model API!"}

@app.post("/predict")
def predict(data: Data):
    model = load("./model/trained_model.pkl")
    prediction = inference(model, data.features)
    return {"prediction": prediction}