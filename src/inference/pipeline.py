import pandas as pd
from datetime import timedelta

from src._preprocessing import build_base_dataset
from src._feature_engineering import build_features

class ForecastPipeline:

    def __init__(self, model):
        self.model = model

    def build_history(self, pollutants, weather_hist):

        return build_base_dataset(
            pollution=pollutants,
            weather=weather_hist
        )

    def forecast(self, history, weather_fc, horizon):

        history = history.copy()
        predictions = []

        current_time = history["datetime"].max()

        for step in range(1, horizon + 1):

            future_time = current_time + timedelta(hours=step)

            future = history.iloc[-1:].copy()
            future["datetime"] = future_time

            # weather
            weather_slice = weather_fc.loc[:future_time]

            if weather_slice.empty:
                weather = weather_fc.iloc[-1]
            else:
                weather = weather_slice.iloc[-1]

            future["temperature"] = weather.temperature
            future["humidity"] = weather.humidity
            future["wind_speed"] = weather.wind_speed
            future["precipitation"] = weather.precipitation

            history = pd.concat([history, future], ignore_index=True)
            history = history.sort_values(
                ["location", "datetime"]
            ).reset_index(drop=True)

            df_features = build_features(history.copy(), fit=False)

            if df_features.empty:
                raise RuntimeError("Feature engineering returned empty dataframe")
            
            row = df_features.iloc[[-1]][self.model.features]

            pred = self.model.predict(row)[0]

            history.loc[
                history["datetime"] == future_time,
                "pm25"
            ] = pred

            predictions.append({
                "datetime": future_time,
                "pm25_pred": pred
            })

        return pd.DataFrame(predictions)