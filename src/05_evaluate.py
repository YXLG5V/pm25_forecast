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
import shap

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

train = pd.read_parquet("./data/preprocessed/train.parquet")
test  = pd.read_parquet("./data/preprocessed/test.parquet")

print("Dataset loaded:", train.shape)

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

X_test = test[FEATURES]
y_test = test["pm25_next"]

# y_pred_log = model.predict(X_test)
# y_pred = np.maximum(0, np.expm1(y_pred_log))

if isinstance(model, dict):  # ensemble
    preds = []
    for m in model.values():
        p_log = m.predict(X_test)
        p = np.maximum(0, np.expm1(p_log))
        preds.append(p)
    y_pred = np.mean(preds, axis=0)
else:
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
naive_pred = test["pm25_lag1"]

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

# --- ensemble eset kezelése ---
if isinstance(model, dict):
    model_for_perm = model["LGBM"]
else:
    model_for_perm = model

result = permutation_importance(
    model_for_perm,
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

print("\nRunning SHAP analysis...")

if isinstance(model, dict):
    model_for_shap = model["LGBM"].named_steps["model"]
elif hasattr(model, "named_steps"):
    model_for_shap = model.named_steps["model"]
else:
    model_for_shap = model

X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(model_for_shap)
shap_values = []
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

import numpy as np
import pandas as pd

shap_importance = np.abs(shap_values).mean(axis=0)
feat_imp = pd.Series(shap_importance, index=X_sample.columns).sort_values(ascending=False)

print("\nTop features (SHAP):")
print(feat_imp.head(10))

# TOP 10 mentéshez
top_features = feat_imp.head(10).to_dict()

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