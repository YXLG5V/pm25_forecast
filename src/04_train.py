# ============================================================
# TRAIN.PY
# ============================================================

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

import matplotlib.pyplot as plt

# ============================================================
# 1. Load data
# ============================================================

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

# ============================================================
# 2. Split data
# ============================================================

split_date = "2025-09-01"

train = df[df.index < split_date]
test  = df[df.index >= split_date]

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

# Basic test
# models = {
#     "RandomForest": RandomForestRegressor(
#         n_estimators=100,
#         max_depth=10,
#         n_jobs=-1,
#         random_state=42
#     ),
#     "HistGB": HistGradientBoostingRegressor(
#         max_iter=200,
#         random_state=42
#     )
# }

# Optimized
models = {
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=7,
            max_features=None,
            random_state=42
        ))
    ]),
    
    "HistGB": Pipeline([
        ("model", HistGradientBoostingRegressor(
            max_iter=365,
            learning_rate=0.04211984714387256,
            max_depth=6,
            min_samples_leaf=62,
            l2_regularization=0.42610860193308486,
            max_bins=230,
            random_state=42
        ))
    ])
}

print("\nTraining model...")
results = []

for name, model in models.items():
    
    # log target
    model.fit(X_train, y_train_log)
    
    pred_log = model.predict(X_test)
    pred = np.maximum(0, np.expm1(pred_log))
    
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    results.append({
        "model": name,
        "MAE": mae,
        "R2": r2
    })

results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df)

results_df.set_index("model")["MAE"].plot.bar()
plt.title("Model comparison (MAE)")
plt.show()

# ============================================================
# 4. Save model
# ============================================================

joblib.dump(models["HistGB"], "./models/model.pkl")
print("Model saved.")


# ============================================================
# 5. MAE
# ============================================================

best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")

# --- TRAIN PRED ---
train_pred_log = best_model.predict(X_train)
train_pred = np.maximum(0, np.expm1(train_pred_log))

# --- TEST PRED ---
test_pred_log  = best_model.predict(X_test)
test_pred = np.maximum(0, np.expm1(test_pred_log))

# --- METRICS ---
from sklearn.metrics import mean_absolute_error, r2_score

train_mae = mean_absolute_error(y_train, train_pred)
test_mae  = mean_absolute_error(y_test, test_pred)

train_r2 = r2_score(y_train, train_pred)
test_r2  = r2_score(y_test, test_pred)

print("\n=== OVERFITTING CHECK ===")
print(f"Train MAE: {train_mae:.3f}", "µg/m³")
print(f"Test  MAE: {test_mae:.3f}", "µg/m³")
print(f"Train R2 : {train_r2:.3f}")
print(f"Test  R2 : {test_r2:.3f}")