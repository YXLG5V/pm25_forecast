import pandas as pd
from datetime import timedelta

from src._preprocessing import build_base_dataset
from src._feature_engineering import build_features

class ForecastPipeline:

    def __init__(self, model):
        self.model = model

    def build_history(self, pollutants, weather_hist):

        return build_base_dataset(
            pollution=pollutants.reset_index(),
            weather=weather_hist.reset_index()
        )

    def forecast(self, history, weather_fc, horizon):

        history = history.copy()
        predictions = []

        current_time = history["datetime"].max()

        for step in range(1, horizon + 1):

            future_time = current_time + timedelta(hours=step)

            future = history.iloc[-1:].copy()
            future["datetime"] = future_time

            future["location_id"] = self.model.categories[future["location"].iloc[0]]

            # weather
            idx = weather_fc.index.get_indexer(
                [future_time],
                method="nearest"
            )[0]

            weather = weather_fc.iloc[idx]

            future["temperature"] = weather.temperature
            future["humidity"] = weather.humidity
            future["wind_speed"] = weather.wind_speed
            future["precipitation"] = weather.precipitation

            history = pd.concat([history, future], ignore_index=True)
            history = history.sort_values(
                ["location", "datetime"]
            ).reset_index(drop=True)

            df_feat_input = history.copy()
            df_feat_input = df_feat_input.reset_index()
            if "datetime" not in df_feat_input.columns:
                df_feat_input = df_feat_input.rename(columns={"index": "datetime"})

            df_features = build_features(df_feat_input)

            row = df_features.loc[df_features.index == future_time]

            if row.empty:
                row = df_features.iloc[[-1]]

            row = self.model.prepare_input(row)

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