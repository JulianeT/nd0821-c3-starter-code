import requests

file_path = "./data/census.csv"

with open(file_path, "rb") as file:
    files = {"file": file}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)

print(response.json())
