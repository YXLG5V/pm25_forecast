# =========================
# OPTUNA + TimeSeriesSplit
# =========================

import optuna
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("./data/preprocessed/preprocessed_with_FE.parquet")
FEATURES = joblib.load("./artifacts/features.pkl")
LOCATION_MAPPING = joblib.load("./artifacts/location_mapping.pkl")

df["pm25_next"] = (
    df.groupby("location")["pm25"].shift(-1)
)

TARGET = "pm25_next"

columns = [
    "pm25_next",
    "pm25_lag1",
    "pm25_lag3",
    "pm25_lag6",
    "pm25_lag24",
    "pm25_roll6",
    "pm25_roll24",
    "pm25_trend_3h",
    "pm25_std_12h",
    "temp_change_3h",
    "humidity_change_3h",
    "wind_change_3h",
    "stagnation_hours_6h"
]

df = df.dropna(subset=columns)


split_date = "2025-09-01"

train = df[df.index < split_date]
test  = df[df.index >= split_date]

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# --- LOG TARGET ---
y_train_log = np.log1p(y_train)

# =========================
# TimeSeries CV
# =========================
tscv = TimeSeriesSplit(n_splits=5)

# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial):

    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42
    }

    # params = {
    #     "n_estimators": trial.suggest_int("n_estimators", 100, 500),
    #     "max_depth": trial.suggest_int("max_depth", 5, 30),
    #     "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    #     "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    #     "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    #     "n_jobs": -1,
    #     "random_state": 42
    # }

    maes = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train_log.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = HistGradientBoostingRegressor(**params)
        # model = RandomForestRegressor(**params)

        model.fit(X_tr, y_tr)

        pred_log = model.predict(X_val)
        pred = np.maximum(0, np.expm1(pred_log))

        maes.append(mean_absolute_error(y_val, pred))

    return np.mean(maes)

# =========================
# RUN OPTUNA
# =========================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest params:")
print(study.best_params)

print("Best CV MAE:", study.best_value)

# =========================
# TRAIN FINAL MODEL
# =========================
best_model = HistGradientBoostingRegressor(
    **study.best_params,
    random_state=42
)

# best_model = RandomForestRegressor(
#     **study.best_params,
#     n_jobs=-1,
#     random_state=42
# )

best_model.fit(X_train, y_train_log)

# =========================
# FINAL TEST EVALUATION
# =========================
pred_log = best_model.predict(X_test)
pred = np.maximum(0, np.expm1(pred_log))

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nTest MAE:", mae)
print("Test R2 :", r2)

# =========================
# SAVE MODEL
# =========================
joblib.dump(best_model, "./models/model_optuna.pkl")

print("Optimized model saved.")