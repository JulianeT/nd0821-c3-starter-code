from fastapi import Body, FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from joblib import load

from starter.ml.data import process_data


app = FastAPI()


class Data(BaseModel):
    age: int
    capital_gain: int
    capital_loss: int
    education: str
    education_num: int
    fnlgt: int
    hours_per_week: int
    marital_status: str
    native_country: str
    occupation: str
    race: str
    relationship: str
    sex: str
    workclass: str


@app.get("/")
def read_root():
    return {"message": "Welcome to our machine learning model API!"}


@app.post("/predict")
async def predict(data: Data = Body(...)):
    # Convert the data into a pandas DataFrame
    df = pd.DataFrame([dict(data)])

    # Define the categorical features
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    # Process the data
    encoder = joblib.load("./model/encoder.pkl")
    lb = joblib.load("./model/lb.pkl")
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = load("./model/trained_model.pkl")

    # Perform inference on the processed data
    predictions = model.predict(X)

    label_map = {0: "<=50K", 1: ">50K"}
    predictions = np.array([label_map[pred] for pred in predictions])

    return {"salary": predictions.tolist()}
