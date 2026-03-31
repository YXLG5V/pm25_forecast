import pandas as pd
import numpy as np
from datetime import timedelta

from src._preprocessing import build_base_dataset, interpolate_station
from src._feature_engineering import build_features

class ForecastPipeline:

    def __init__(self, model):
            self.model = model
            self.features = model.features

    def build_history(self, pollutants, weather_hist):
        
        df = build_base_dataset(
            pollution=pollutants,
            weather=weather_hist
        )
        
        df = interpolate_station(df)
        
        return df

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
            
            X = df_features.iloc[[-1]][self.features]

            if X.isna().any().any():
                print("NaN a feature vectorban!")

            model = self.model.model

            if isinstance(model, dict):
                preds = []
                for m in model.values():
                    p = np.maximum(0, np.expm1(m.predict(X)[0]))
                    preds.append(p)
                pred = np.mean(preds)
            else:
                pred = np.maximum(0, np.expm1(model.predict(X)[0]))

            history.loc[
                history["datetime"] == future_time,
                "pm25"
            ] = pred

            predictions.append({
                "datetime": future_time,
                "pm25_pred": pred
            })

        return pd.DataFrame(predictions)