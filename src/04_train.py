from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, TimeSeriesSplit
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from _model_wrappers import NNWrapper

np.random.seed(42)
tf.random.set_seed(42)

# Log target kapcsoló
USE_LOG_TARGET = False
PLOT = False

def transform_target(y):
    return np.log1p(y) if USE_LOG_TARGET else y

def inverse_target(y):
    return np.maximum(0, np.expm1(y)) if USE_LOG_TARGET else y

def plot_learning_curve(model, X, y, model_name):

    cv = TimeSeriesSplit(n_splits=5)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)

    plt.figure(figsize=(8,5))

    plt.plot(train_sizes, train_mae, label="Train MAE")
    plt.plot(train_sizes, val_mae, label="Validation MAE")

    plt.axhline(
        y=baseline_mae,
        color="blue",
        linestyle="--",
        label=f"Lag1 baseline = {baseline_mae:.2f}"
    )

    plt.axhline(
        y=3.0,
        color="green",
        linestyle="-",
        label=f"Desired"
    )

    plt.xlabel("Training samples")
    plt.ylabel("MAE")
    plt.title(f"Learning Curve - {model_name}")

    plt.legend()
    plt.grid(True)
    
    gap = val_mae[-1] - train_mae[-1]

    plt.text(
        train_sizes[-1],
        val_mae[-1],
        f"Gap={gap:.2f}",
    )

    plt.show()

def plot_nn_learning_curve(history):

    train_mae = history.history["loss"]
    val_mae = history.history["val_loss"]

    plt.figure(figsize=(8,5))

    plt.plot(train_mae, label="Train MAE")
    plt.plot(val_mae, label="Validation MAE")

    plt.axhline(
        y=baseline_mae,
        color="blue",
        linestyle="--",
        label=f"Lag1 baseline = {baseline_mae:.2f}"
    )

    plt.axhline(
        y=3.0,
        color="green",
        linestyle="-",
        label="Desired"
    )

    gap = val_mae[-1] - train_mae[-1]

    plt.text(
        len(train_mae) * 0.8,
        val_mae[-1],
        f"Gap={gap:.2f}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Learning Curve - NeuralNet")

    plt.legend()
    plt.grid(True)
    plt.show()

# Adatok betöltése
train = pd.read_parquet("./data/preprocessed/train.parquet")
test  = pd.read_parquet("./data/preprocessed/test.parquet")
FEATURES = joblib.load("./artifacts/features.pkl")
ENSEMBLE_PATH = "./models/models_ensemble.pkl"

train = train.sort_values(["location", "datetime"])
test  = test.sort_values(["location", "datetime"])

train["pm25_next"] = train.groupby("location")["pm25"].shift(-1)
test["pm25_next"]  = test.groupby("location")["pm25"].shift(-1)

# NAN-ok törlése
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

# Train és teszt adatok létrehozása
X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

y_train_used = transform_target(y_train)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# Model paraméterezés
models = {
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            min_samples_split=30,
            max_features="sqrt",
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
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=0.01007089724896645))
    ]),
}

results = []
trained_models = {}

# BASELINE (lag1)
baseline_pred = X_test["pm25_lag1"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)

results.append({
    "model": "Baseline_lag1",
    "MAE": baseline_mae,
    "R2": baseline_r2
})

if PLOT:
    for name, model in models.items():

        plot_learning_curve(
            model,
            X_train,
            y_train_used,
            name
        )

print("\nTraining models...")

# Sklearn models
for name, model in models.items():
    
    model.fit(X_train, y_train_used)
    trained_models[name] = model
    
    pred_raw = model.predict(X_test)
    pred = inverse_target(pred_raw)
    
    results.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred)
    })

# Neural network implementáció

# SPLIT a nyers adatokon
val_ratio = 0.2
n = len(X_train)
split = int(n * (1 - val_ratio))

X_tr_raw = X_train.iloc[:split]
X_val_raw = X_train.iloc[split:]

y_tr = y_train_used.values[:split]
y_val = y_train_used.values[split:]


# IMPUTER (csak trainen fit)
imputer = SimpleImputer(strategy="median")

X_tr_imp = imputer.fit_transform(X_tr_raw)
X_val_imp = imputer.transform(X_val_raw)

# SCALER (csak trainen fit)
scaler = StandardScaler()

X_tr = scaler.fit_transform(X_tr_imp)
X_val = scaler.transform(X_val_imp)

nn_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='swish'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation = 'linear')
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss = "mae",
    metrics=[
        "mae",
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=10,
        restore_best_weights=True
    )
]

history = nn_model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    verbose=1,
    callbacks=callbacks
)

if PLOT:
    plot_nn_learning_curve(history)

## Tanítás (teljes trainen)
nn_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='swish'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='linear')
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mae"
)

imputer_full = SimpleImputer(strategy="median")
scaler_full = StandardScaler()

X_train_imp_full = imputer_full.fit_transform(X_train)
X_train_scaled_full = scaler_full.fit_transform(X_train_imp_full)

nn_model.fit(
    X_train_scaled_full,
    y_train_used.values,
    epochs = np.argmin(history.history["val_loss"]) + 1,
    batch_size=256,
    verbose=0
)

trained_models["NeuralNet"] = NNWrapper(
    nn_model,
    imputer_full,
    scaler_full
)

# --- NN evaluation ---
nn_wrapper = trained_models["NeuralNet"]

test_pred_raw = nn_wrapper.predict(X_test)
test_pred = inverse_target(test_pred_raw)

results.append({
    "model": "NeuralNet",
    "MAE": mean_absolute_error(y_test, test_pred),
    "R2": r2_score(y_test, test_pred)
})

# Ensemble (ha létezik)
if os.path.exists(ENSEMBLE_PATH):

    print("\nEvaluating existing ensemble...")

    ensemble_models = joblib.load(ENSEMBLE_PATH)

    preds = []

    for name, model in ensemble_models.items():

        pred_raw = model.predict(X_test)

        pred = inverse_target(pred_raw)
        preds.append(pred)

    ensemble_pred = np.mean(preds, axis=0)

    results.append({
        "model": "ENSEMBLE",
        "MAE": mean_absolute_error(y_test, ensemble_pred),
        "R2": r2_score(y_test, ensemble_pred)
    })

# Eredmények
results_df = pd.DataFrame(results).sort_values("MAE")

top_models = (
    results_df[results_df["model"] != "ENSEMBLE"]
    .head(2)["model"]
    .tolist()
)

print("Top models:", top_models)
print(results_df)

if PLOT:
    results_df.set_index("model")["MAE"].plot.bar()
    plt.title("Model comparison (MAE, without Ridge)")
    plt.show()

# Model mentése
best_model_name = results_df.iloc[0]["model"]

if best_model_name == "ENSEMBLE":
    best_model = ensemble_models
else:
    best_model = trained_models[best_model_name]

joblib.dump(best_model, "./models/model.pkl")
print(f"Best model = {best_model_name} saved.")

# LGBM külön mentése SHAP-hoz
for name, mdl in trained_models.items():

    filename = name.lower()

    joblib.dump(
        mdl,
        f"./models/{filename}.pkl"
    )

print("Individual models saved.")

top_trained_models = {
    name: trained_models[name]
    for name in top_models
}

joblib.dump(top_trained_models, ENSEMBLE_PATH)
print("Top-2 ensemble saved.")