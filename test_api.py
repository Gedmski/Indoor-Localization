# Test the API
import requests
import json

# Test health endpoint
try:
    response = requests.get("http://localhost:8080/health")
    print("Health check:", response.json())
except Exception as e:
    print("Health check failed:", e)

# Test prediction endpoint with sample data
sample_data = {
    "rssi": {
        "WAP001": -50,
        "WAP002": -60,
        "WAP003": -110,  # Missing AP
        "WAP004": -70
    }
}

try:
    response = requests.post("http://localhost:8080/predict",
                           json=sample_data)
    print("Prediction:", response.json())
except Exception as e:
    print("Prediction failed:", e)