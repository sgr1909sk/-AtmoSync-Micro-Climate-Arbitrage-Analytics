
import random
import json
import time
import csv
from pathlib import Path
from datetime import datetime, timezone

CONTAINERS = ["ATMO-001", "ATMO-002", "ATMO-003"]

# Find the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_FILE = DATA_DIR / "sensor_data.csv"

FIELDNAMES = [
    "container_id",
    "commodity",
    "timestamp",
    "temperature",
    "humidity",
    "vibration"
]


def generate_sensor_reading(container_id):
    if container_id == "ATMO-001":
        temperature = random.uniform(26, 30)
    else:
        temperature = random.uniform(4, 10)

    return {
        "container_id": container_id,
        "commodity": "avocados",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": round(temperature, 2),
        "humidity": round(random.uniform(65, 95), 2),
        "vibration": round(random.uniform(0.1, 2.0), 2)
    }


def save_reading(reading):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(reading)


if __name__ == "__main__":
    print("AtmoSync IoT Simulator Started")
    print(f"Saving sensor data to: {CSV_FILE}")

    try:
        while True:
            for container in CONTAINERS:
                reading = generate_sensor_reading(container)
                save_reading(reading)

                print(json.dumps(reading))

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")
