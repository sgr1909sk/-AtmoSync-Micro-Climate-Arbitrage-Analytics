
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "data" / "sensor_data.csv"


def analyze_sensor_data():
    if not CSV_FILE.exists():
        print("Sensor data not found. Run the simulator first.")
        return

    df = pd.read_csv(CSV_FILE)

    if df.empty:
        print("The sensor data file is empty.")
        return

    print("\n--- AtmoSync Sensor Data ---")
    print(df.head())

    print("\n--- Dataset Information ---")
    print("Total readings:", len(df))
    print("Containers:", df["container_id"].nunique())

    print("\n--- Average Temperature ---")
    print(
        df.groupby("container_id")["temperature"]
        .mean()
        .round(2)
    )

    print("\n--- Average Humidity ---")
    print(
        df.groupby("container_id")["humidity"]
        .mean()
        .round(2)
    )

    # Demonstration threshold, not a validated avocado quality limit.
    threshold = 25

    df["temperature_alert"] = (
        df["temperature"] > threshold
    )

    print("\n--- Temperature Alerts ---")
    print(
        df.groupby("container_id")["temperature_alert"]
        .sum()
        .astype(int)
    )


if __name__ == "__main__":
    analyze_sensor_data()
