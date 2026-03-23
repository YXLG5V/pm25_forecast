import requests
import pandas as pd

from _utils import get_time_window


LAT = 47.6875
LON = 17.6504


def download_weather(days_back):

    start, end = get_time_window(days_back)

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
    df = df.set_index("datetime")

    df = df.loc[start:end]

    return df


if __name__ == "__main__":

    weather = download_weather(days_back=730)

    weather.to_csv("./data/raw/weather.csv")

    print("Saved weather dataset.")