# =========================
# OPTUNA + TimeSeriesSplit
# =========================

import optuna
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# =========================
# LOAD DATA
# =========================
train = pd.read_parquet("./data/preprocessed/train.parquet")
test  = pd.read_parquet("./data/preprocessed/test.parquet")
FEATURES = joblib.load("./artifacts/features.pkl")
LOCATION_MAPPING = joblib.load("./artifacts/location_mapping.pkl")

train = train.sort_values(["datetime", "location"])
test = test.sort_values(["location", "datetime"])

train["pm25_next"] = train.groupby("location")["pm25"].shift(-1)
test["pm25_next"]  = test.groupby("location")["pm25"].shift(-1)

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

train = train.dropna(subset=columns)
test  = test.dropna(subset=columns)

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

y_train_log = np.log1p(y_train)

# =========================
# TimeSeries CV
# =========================
tscv = TimeSeriesSplit(n_splits=5)

# =========================
# 🔁 MODEL SWITCH
# =========================
MODEL_TYPE = "ridge"
# options:
# "histgb", "rf", "lgbm", "xgb", "ridge"

# =========================
# MODEL CONFIG (SWITCH-CASE)
# =========================
MODEL_CONFIGS = {

    # =====================================================
    # HIST GRADIENT BOOSTING
    # =====================================================
    "histgb": lambda trial: (
        {
            "max_iter": trial.suggest_int("max_iter", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "max_bins": trial.suggest_int("max_bins", 64, 255),
            "early_stopping": False,
            "random_state": 42
        },
        lambda params: HistGradientBoostingRegressor(**params)
    ),

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    "rf": lambda trial: (
        {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "n_jobs": -1,
            "random_state": 42
        },
        lambda params: RandomForestRegressor(**params)
    ),

    # =====================================================
    # LIGHTGBM
    # =====================================================
    "lgbm": lambda trial: (
        {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbosity": -1
        },
        lambda params: LGBMRegressor(**params)
    ),

    # =====================================================
    # XGBOOST
    # =====================================================
    "xgb": lambda trial: (
        {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42
        },
        lambda params: XGBRegressor(**params)
    ),

    # =====================================================
    # RIDGE (pipeline + imputer!)
    # =====================================================
    "ridge": lambda trial: (
        {
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True)
        },
        lambda params: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(**params))
        ])
    )
}

# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial):

    # --- model kiválasztása ---
    params, builder = MODEL_CONFIGS[MODEL_TYPE](trial)
    model_builder = lambda: builder(params)

    # =========================
    # CV LOOP
    # =========================
    maes = []

    for train_idx, val_idx in tscv.split(X_train):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train_log.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = model_builder()
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
# FINAL MODEL
# =========================
params = study.best_params

if MODEL_TYPE == "ridge":
    best_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", Ridge(**params))
    ])
else:
    MODEL_CLASSES = {
        "histgb": HistGradientBoostingRegressor,
        "rf": RandomForestRegressor,
        "lgbm": LGBMRegressor,
        "xgb": XGBRegressor
    }

    best_model = MODEL_CLASSES[MODEL_TYPE](**params, random_state=42)

best_model.fit(X_train, y_train_log)

# =========================
# TEST
# =========================
pred_log = best_model.predict(X_test)
pred = np.maximum(0, np.expm1(pred_log))

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nTest MAE:", mae)
print("Test R2 :", r2)

# =========================
# SAVE
# =========================
joblib.dump(best_model, f"./models/model_optuna_{MODEL_TYPE}.pkl")

print("Optimized model saved.")