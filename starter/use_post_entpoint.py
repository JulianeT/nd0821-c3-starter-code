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

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
