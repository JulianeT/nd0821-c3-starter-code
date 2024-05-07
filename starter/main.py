from fastapi import Body, FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from starter.ml.data import process_data


app = FastAPI()
model = joblib.load("./starter/model/trained_model.pkl")
encoder = joblib.load("./starter/model/encoder.pkl")
binarizer = joblib.load("./starter/model/lb.pkl")


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class Data(BaseModel):
    age: int = Field(default=45)
    capital_gain: int = Field(default=2174)
    capital_loss: int = Field(default=0)
    education: str = Field(default="Bachelors")
    education_num: int = Field(default=13)
    fnlgt: int = Field(default=2334)
    hours_per_week: int = Field(default=60)
    marital_status: str = Field(default="Never-married")
    native_country: str = Field(default="Cuba")
    occupation: str = Field(default="Prof-specialty")
    race: str = Field(default="Black")
    relationship: str = Field(default="Wife")
    sex: str = Field(default="Female")
    workclass: str = Field(default="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


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
