import requests


BASE_URL = "http://localhost:8080"


def main():
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        print("Health check:", health.json())
    except Exception as exc:
        print("Health check failed:", exc)
        return

    sample_data = {
        "rssi": {
            "AP1": -58,
            "AP2": -66,
            "AP3": -100,
            "AP4": -73,
            "AP5": -85,
        },
        "top_k": 3,
    }

    try:
        prediction = requests.post(
            f"{BASE_URL}/predict", json=sample_data, timeout=10
        )
        print("Prediction:", prediction.json())
    except Exception as exc:
        print("Prediction failed:", exc)


if __name__ == "__main__":
    main()
