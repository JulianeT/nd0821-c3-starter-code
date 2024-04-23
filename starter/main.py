from fastapi import FastAPI, File, UploadFile
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from joblib import load

from starter.ml.data import process_data


class Data(BaseModel):
    features: List[float] = Field(..., example=[0.1, 0.2, 0.3, 0.4])


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to our machine learning model API!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file.file)

    # Define the categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the data
    encoder = joblib.load("./model/encoder.pkl")
    lb = joblib.load("./model/lb.pkl")
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Load model
    model = load("./model/trained_model.pkl")

    # Perform inference on the processed data
    predictions = model.predict(X)

    return {"predictions": predictions.tolist()}
