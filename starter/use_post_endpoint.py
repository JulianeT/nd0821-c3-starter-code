# flake8: noqa
import requests

data = {
    "age": 45,
    "capital_gain": 2174,
    "capital_loss": 0,
    "education": "Bachelors",
    "education_num": 13,
    "fnlgt": 2334,
    "hours_per_week": 60,
    "marital_status": "Never-married",
    "native_country": "Cuba",
    "occupation": "Prof-specialty",
    "race": "Black",
    "relationship": "Wife",
    "sex": "Female",
    "workclass": "State-gov",
}

response = requests.post(
    "https://nd0821-c3-starter-code-c9l2.onrender.com/predict",
    json=data,
)
print("Status code:", response.status_code)
print("Result of the model inference:", response.json())
