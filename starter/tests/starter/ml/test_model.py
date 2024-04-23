import numpy as np
import pytest
from sklearn.datasets import make_classification
from starter.ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def classification_dataset():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    return X, y


def test_train_model(classification_dataset):
    X, y = classification_dataset
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(classification_dataset):
    X, y = classification_dataset
    model = train_model(X, y)
    predictions = inference(model, X)
    assert predictions.shape == y.shape
