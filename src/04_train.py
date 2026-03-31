# ============================================================
# TRAIN.PY
# ============================================================

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

import matplotlib.pyplot as plt

import os

# ============================================================
# 1. Load data
# ============================================================

train = pd.read_parquet("./data/preprocessed/train.parquet")
test  = pd.read_parquet("./data/preprocessed/test.parquet")
FEATURES = joblib.load("./artifacts/features.pkl")
LOCATION_MAPPING = joblib.load("./artifacts/location_mapping.pkl")
ENSEMBLE_PATH = "./models/models_ensemble.pkl"

train = train.sort_values(["location", "datetime"])
test  = test.sort_values(["location", "datetime"])

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

# ============================================================
# 2. Train
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# --- LOG TRANSFORM TARGET ---
y_train_log = np.log1p(y_train)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

print("Train period:", train.index.min(), "→", train.index.max())
print("Test period :", test.index.min(), "→", test.index.max())

# ============================================================
# 3. Models
# ============================================================

# Optimized
# Optimized
models = {
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=715,
            max_depth=16,
            min_samples_split=9,
            min_samples_leaf=3,
            max_features=None,
            random_state=42
        ))
    ]),

    "HistGB": Pipeline([
        ("model", HistGradientBoostingRegressor(
            max_iter=309,
            learning_rate=0.052392441315122884,
            max_depth=6,
            min_samples_leaf=89,
            l2_regularization=0.10712858441423956,
            max_bins=172,
            random_state=42
        ))
    ]),

    "LGBM": Pipeline([
    ("model", LGBMRegressor(
        n_estimators=723,
        learning_rate=0.012885472793169907,
        max_depth=11,
        num_leaves=71,
        subsample=0.6904437214202783,
        colsample_bytree=0.6924371848724423,
        random_state=42
    ))
    ]),

    "XGB": Pipeline([
        ("model", XGBRegressor(
            n_estimators=615,
            learning_rate=0.08447106507617709,
            max_depth=3,
            subsample=0.9593316490919434,
            colsample_bytree=0.743423049049403,
            random_state=42
        ))
    ]),

    "Ridge": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", Ridge(alpha=0.01007089724896645))
    ]),
}


print("\nTraining models...")
results = []
trained_models = {}

for name, model in models.items():
    
    # log target
    model.fit(X_train, y_train_log)

    trained_models[name] = model
    
    pred_log = model.predict(X_test)
    pred = np.maximum(0, np.expm1(pred_log))
    
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    results.append({
        "model": name,
        "MAE": mae,
        "R2": r2
    })

# ============================================================
# ENSEMBLE (ha létezik)
# ============================================================

if os.path.exists(ENSEMBLE_PATH):

    print("\nEvaluating existing ensemble...")

    ensemble_models = joblib.load(ENSEMBLE_PATH)

    preds = []

    for model in ensemble_models.values():
        p_log = model.predict(X_test)
        p = np.maximum(0, np.expm1(p_log))
        preds.append(p)

    ensemble_pred = np.mean(preds, axis=0)

    mae = mean_absolute_error(y_test, ensemble_pred)
    r2  = r2_score(y_test, ensemble_pred)

    results.append({
        "model": "ENSEMBLE",
        "MAE": mae,
        "R2": r2
    })

results_df = pd.DataFrame(results).sort_values("MAE")
top_models = (
    results_df[results_df["model"] != "ENSEMBLE"]
    .head(2)["model"]
    .tolist()
)
print("Top models:", top_models)
print(results_df)

plot_df = results_df[results_df["model"] != "Ridge"]

plot_df.set_index("model")["MAE"].plot.bar()
plt.title("Model comparison (MAE, without Ridge)")
plt.show()

# ============================================================
# 4. Save model
# ============================================================

best_model_name = results_df.iloc[0]["model"]

if best_model_name == "ENSEMBLE":
    best_model = ensemble_models
else:
    best_model = trained_models[best_model_name]

joblib.dump(best_model, "./models/model.pkl")
print(f"Best model = {best_model_name} saved.")

top_trained_models = {
    name: trained_models[name]
    for name in top_models
}

joblib.dump(top_trained_models, ENSEMBLE_PATH)
print("Top-2 ensemble saved.")

# ============================================================
# 5. MAE
# ============================================================

print(f"\nBest model: {best_model_name}")

if best_model_name == "ENSEMBLE":

    # --- TRAIN ---
    train_preds = []
    for m in best_model.values():
        p = np.maximum(0, np.expm1(m.predict(X_train)))
        train_preds.append(p)
    train_pred = np.mean(train_preds, axis=0)

    # --- TEST ---
    test_preds = []
    for m in best_model.values():
        p = np.maximum(0, np.expm1(m.predict(X_test)))
        test_preds.append(p)
    test_pred = np.mean(test_preds, axis=0)

else:

    # --- TRAIN ---
    train_pred_log = best_model.predict(X_train)
    train_pred = np.maximum(0, np.expm1(train_pred_log))

    # --- TEST ---
    test_pred_log  = best_model.predict(X_test)
    test_pred = np.maximum(0, np.expm1(test_pred_log))

    
# --- METRICS ---
train_mae = mean_absolute_error(y_train, train_pred)
test_mae  = mean_absolute_error(y_test, test_pred)

train_r2 = r2_score(y_train, train_pred)
test_r2  = r2_score(y_test, test_pred)

print("\n=== OVERFITTING CHECK ===")
print(f"Train MAE: {train_mae:.3f}", "µg/m³")
print(f"Test  MAE: {test_mae:.3f}", "µg/m³")
print(f"Train R2 : {train_r2:.3f}")
print(f"Test  R2 : {test_r2:.3f}")