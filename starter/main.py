from dataclasses import Field
from fastapi import Body, FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

from starter.starter.ml.data import process_data


app = FastAPI()


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class Data(BaseModel):
    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
            alias_generator = hyphen_to_underscore
            allow_population_by_field_name = True


@app.on_event("startup")
async def startup_event(): 
    global model, encoder, binarizer
    model = joblib.load("./starter/model/trained_model.pkl")
    encoder = joblib.load("./starter/model/encoder.pkl")
    binarizer = joblib.load("./starter/model/lb.pkl")


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
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=binarizer,
    )

    # Perform inference on the processed data
    predictions = model.predict(X)

    label_map = {0: "<=50K", 1: ">50K"}
    predictions = np.array([label_map[pred] for pred in predictions])

    return {"salary": predictions.tolist()}
