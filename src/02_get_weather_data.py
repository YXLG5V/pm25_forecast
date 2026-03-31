import requests
import pandas as pd

LAT = 47.6875
LON = 17.6504


def download_weather(start, end):

    print("Downloading weather data...")
    print("TIME WINDOW:", start, "→", end)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = dict(
        latitude=LAT,
        longitude=LON,
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        hourly=[
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation"
        ],
        timezone="UTC"
    )

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "precipitation": data["hourly"]["precipitation"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]

    return df


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    args = parser.parse_args()

    start = pd.to_datetime(args.start_date, utc=True)
    end   = pd.to_datetime(args.end_date, utc=True)

    weather = download_weather(start, end)

    weather.to_csv("./data/raw/weather.csv")

    print("Saved weather dataset.")

