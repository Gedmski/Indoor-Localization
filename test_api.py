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
        "imu": {
            "accel_x": 0.01,
            "accel_y": 0.02,
            "accel_z": 1.01,
            "gyro_x": 0.1,
            "gyro_y": -0.1,
            "gyro_z": 0.0,
            "mag_x": -40.0,
            "mag_y": 5.0,
            "mag_z": -6.0,
            "mag_heading": 170.0,
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
