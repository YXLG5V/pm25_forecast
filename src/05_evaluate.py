# ============================================================
# 05_EVALUATE.PY
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from sklearn.inspection import permutation_importance

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = "./models/model.pkl"
FEATURES_PATH = "./artifacts/features.pkl"
LOCATION_MAPPING_PATH = "./artifacts/location_mapping.pkl"

# ============================================================
# 0. Load model
# ============================================================

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)
LOCATION_MAPPING = joblib.load(LOCATION_MAPPING_PATH)
print("Model loaded: ", MODEL_PATH)

# ============================================================
# 1. Load data
# ============================================================

df = pd.read_parquet("./data/preprocessed/preprocessed_with_FE.parquet")

print("Dataset loaded:", df.shape)

df["pm25_next"] = (
    df.groupby("location")["pm25"].shift(-1)
)

df = df.dropna(subset=["pm25_next"])

lag_columns = [
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
df = df.dropna(subset=lag_columns)

if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")

df = df.sort_index()

# ============================================================
# 2. Time-aware split
# ============================================================

TARGET = "pm25_next"

split_date = pd.to_datetime("2025-09-01", utc=True)

test = df[df.index >= split_date]

X_test = test[FEATURES]
y_test = test["pm25_next"]

y_pred_log = model.predict(X_test)
y_pred = np.maximum(0, np.expm1(y_pred_log))


# ============================================================
# 3. 1-step sanity validation
# ============================================================

# --- MAE ---
mae = mean_absolute_error(y_test, y_pred)

# --- RMSE ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# --- R² ---
r2 = r2_score(y_test, y_pred)

# --- MAPE ---
mape = mean_absolute_percentage_error(y_test, y_pred)

# --- SMAPE ---
smape = np.mean(
    2 * np.abs(y_pred - y_test) /
    (np.abs(y_test) + np.abs(y_pred) + 1e-6)
)

# --- MASE ---
naive_pred = (
    test.groupby("location")[TARGET]
    .shift(1)
)

valid = naive_pred.notna()

naive_mae = mean_absolute_error(
    y_test[valid],
    naive_pred[valid]
)

mase = mae / naive_mae

print("\nValidation results")
print("1-step MAE:", round(mae, 3), "µg/m³")
print("RMSE :", round(rmse,3))
print("MAPE :", round(mape,3))
print("SMAPE:", round(smape,3))
print("MASE :", round(mase,3))
print("R2   :", round(r2,3))


# ============================================================
# 4. Validation metrics
# ============================================================

# Predikció vs valós érték scatter
# Megmutatja mennyire közel van a modell az ideálishoz.

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")
plt.xlabel("True PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Prediction vs True")
plt.show()

# Hiba (residual) eloszlás, ideális ha közép ≈ 0 szimmetrikus
# Megmutatja a modell biasát.

residuals = y_test - y_pred

plt.figure()
plt.hist(residuals, bins=50)
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Distribution")
plt.show()

#Residual vs prediction plot, Ha tölcsér alakú → heteroszkedaszticitás.
#Outliereket mutat.

plt.figure()
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0)
plt.xlabel("Predicted PM2.5")
plt.ylabel("Residual")
plt.title("Residual vs Prediction")
plt.show()

#Idősoros predikció grafikon
# spike detection, lag, drift

plt.figure(figsize=(12,5))
plt.plot(y_test.sort_index(), label="True")
plt.plot(pd.Series(y_pred, index=y_test.index).sort_index(), label="Prediction")
plt.legend()
plt.title("Prediction vs True Time Series")
plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.show()

#Feature permutation importance grafikon

y_test_log = np.log1p(y_test)

result = permutation_importance(
    model,
    X_test,
    y_test_log,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance = pd.Series(
    result.importances_mean,
    index=FEATURES
).sort_values()

plt.figure(figsize=(8,6))
importance.plot.barh()
plt.title("Permutation Feature Importance")
plt.show()

# hiba vs idő
plt.figure(figsize=(12,4))
plt.plot(y_test.index, residuals)
plt.axhline(0)
plt.title("Residuals over Time")
plt.xlabel("Time")
plt.ylabel("Error")
plt.show()

# hiba vs érték
plt.figure()
plt.scatter(y_test, residuals, alpha=0.4)
plt.axhline(0)
plt.xlabel("True PM2.5")
plt.ylabel("Residual")
plt.title("Error vs True Value")
plt.show()


# ============================================================
# 5. SHAP ANALYSIS
# ============================================================

import shap

print("\nRunning SHAP analysis...")

# --- pipeline unwrap ---
if hasattr(model, "named_steps"):
    model_for_shap = model.named_steps["model"]
else:
    model_for_shap = model

# --- sample---
X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)

# --- explainer ---
explainer = shap.TreeExplainer(model_for_shap)

shap_values = None
shap_values = explainer.shap_values(X_sample)

# ============================================================
# 5.1. SUMMARY PLOT
# ============================================================

shap.summary_plot(shap_values, X_sample)

# ============================================================
# 5.2. FEATURE IMPORTANCE (bar)
# ============================================================

shap.summary_plot(shap_values, X_sample, plot_type="bar")

# ============================================================
# 5.3. TOP FEATURE AUTOMATIKUSAN
# ============================================================

importance = np.abs(shap_values).mean(axis=0)
feat_imp = pd.Series(importance, index=X_sample.columns).sort_values(ascending=False)

print("\nTop features (SHAP):")
print(feat_imp.head(10))

# ============================================================
# 6. Save metrics
# ============================================================

top_features = feat_imp.head(10).to_dict()

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "SMAPE": smape,
    "MASE": mase,
    "R2": r2,
    "top_features": top_features
}

pd.Series(metrics).to_json("./models/evaluation.json")

print("Metrics saved to ./models/evaluation.json")