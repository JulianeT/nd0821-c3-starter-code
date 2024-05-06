import pytest
from fastapi.testclient import TestClient
from starter.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    message = "Welcome to our machine learning model API!"
    assert response.json() == {"message": message}


def test_prediction_endpoint_for_class_0(client):
    response = client.post(
        "/predict",
        json={
            "age": 39,
            "capital_gain": 2174,
            "capital_loss": 0,
            "education": "Bachelors",
            "education_num": 13,
            "fnlgt": 77516,
            "hours_per_week": 40,
            "marital_status": "Never-married",
            "native_country": "United-States",
            "occupation": "Adm-clerical",
            "race": "white",
            "relationship": "Not-in-family",
            "sex": "Male",
            "workclass": "State-gov",
            }
        )
    assert response.status_code == 200
    assert response.json() == {'salary': ['<=50K']}


def test_prediction_endpoint_for_class_1(client):
    response = client.post(
        "/predict",
        json={
            "age": 52,
            "capital_gain": 0,
            "capital_loss": 0,
            "education": "HS-grad",
            "education_num": 9,
            "fnlgt": 209642,
            "hours_per_week": 45,
            "marital_status": "Married-civ-spouse",
            "native_country": "United-States",
            "occupation": "Exec-managerial",
            "race": "White",
            "relationship": "Husband",
            "sex": "Male",
            "workclass": "Self-emp-not-inc",
            }
        )
    assert response.status_code == 200
    assert response.json() == {'salary': ['>50K']}
