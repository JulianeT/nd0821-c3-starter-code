import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference, save_model, train_model

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

# Load in the data
data = pd.read_csv("../data/census.csv")

# Split data int train and test set
train, test = train_test_split(data, test_size=0.20)

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

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save model
model = train_model(X_train, y_train)
save_model(model, "../model/trained_model.pkl")


# Evaluate model on the test set
with open("../model/slice_output.txt", "w") as f:
    X_slice, y_slice, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    predictions = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, predictions)
    f.write(
        f"Model metrics: "
        f"Precision={precision}, Recall={recall}, F1={fbeta}\n"
    )
