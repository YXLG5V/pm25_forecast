import pandas as pd
import numpy as np
from datetime import timedelta

from src._preprocessing import build_base_dataset, interpolate_station
from src._feature_engineering import build_features

class ForecastPipeline:
    FEATURE_MAP = {
        "pm25_lag1": "Recent PM",
        "pm25_lag3": "PM (3h ago)",
        "pm25_lag6": "PM (6h ago)",
        "pm25_lag24": "PM (24h ago)",
        "pm25_trend_3h": "Trend",
        "pm25_roll6": "Avg PM (6h)",
        "pm25_roll24": "Avg PM (24h)",
        "pm25_std_12h": "Variability",

        "temperature": "Temperature",
        "humidity": "Humidity",
        "wind_speed": "Wind",
        "precipitation": "Rain",

        "temp_change_3h": "Temp Change",
        "humidity_change_3h": "Humidity Change",
        "wind_change_3h": "Wind Change",

        "stagnation_index": "Air Stagnation",
        "mixing_index": "Mixing",
        "ventilation_index": "Ventilation",

        "pm10": "PM10",
        "no2": "NO₂",
        "so2": "SO₂"
    }

    IGNORE_FEATURES = [
        "hour_sin", "hour_cos",
        "month", "month_sin", "month_cos",
        "weekend_flag",
        "location_id",
        "lat_norm", "lon_norm"
    ]


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
        logged_nan = False

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

            if X.isna().any().any() and not logged_nan:
                print(f"NaN a feature vectorban!")
                missing_pct = X.isna().mean().mean() * 100
                print(f"Missing ratio: {missing_pct:.1f}%")
                missing_features = X.columns[X.isna().any()].tolist()
                print(f"Missing features: {missing_features}")
                logged_nan = True

            model = self.model.model

            # ===== SHAP =====
            shap_values = self.explainer.shap_values(X)

            row = shap_values[0]

            effects = []

            for i, f in enumerate(self.features):
                v = row[i]
                if f in self.IGNORE_FEATURES:
                    continue

                effects.append({
                    "feature": self.FEATURE_MAP.get(f, f),
                    "value": float(v),
                    "direction": "up" if v > 0 else "down"
                })

            # top 5
            effects = sorted(
                effects,
                key=lambda x: abs(x["value"]),
                reverse=True
            )[:5]

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

            from zoneinfo import ZoneInfo

            if future_time.tzinfo is None:
                future_time = future_time.replace(tzinfo=ZoneInfo("UTC"))

            local_time = future_time.astimezone(ZoneInfo("Europe/Budapest"))

            predictions.append({
                "datetime": local_time,
                "pm25_pred": pred,
                "effects": effects
            })
            
        return pd.DataFrame(predictions)