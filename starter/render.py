import requests

# TODO: Replace data with your own data
data = {"features": [0.1, 0.2, 0.3, 0.4]}
response = requests.post("https://your-app.herokuapp.com/predict", json=data)
print(response.json())
