import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@pytest.fixture
def client():
    return TestClient(app)

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to our machine learning model API!"}

def test_prediction_endpoint_for_class_0(client):
    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3, 0.4]})
    assert response.status_code == 200
    assert response.json()["prediction"] == 0

def test_prediction_endpoint_for_class_1(client):
    response = client.post("/predict", json={"features": [0.5, 0.6, 0.7, 0.8]})
    assert response.status_code == 200
    assert response.json()["prediction"] == 1