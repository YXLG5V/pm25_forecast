# ============================================================
# 03_PREPROCESS.PY
# Build ML dataset from raw pollutants + weather
# ============================================================

import pandas as pd
import joblib

from _preprocessing import (
    interpolate_station,
    build_base_dataset
)

from _feature_engineering import (
    build_features,
    FEATURES
)


# ============================================================
# LOAD RAW DATA
# ============================================================

print("Loading raw datasets...")

pollution = pd.read_csv("./data/raw/pollutants.csv", parse_dates=["datetime"])
weather = pd.read_csv("./data/raw/weather.csv")

print("Pollutants shape:", pollution.shape)
print("Weather shape:", weather.shape)

# ============================================================
# BUILD BASE DATASET
# ============================================================

print("\nBuilding base dataset...")

df = build_base_dataset(
    pollution=pollution,
    weather=weather
)

# ============================================================
# SPLIT
# ============================================================

#split_date = "2025-09-01"
split_date = df["datetime"].quantile(0.75)
df = df.sort_values(["location", "datetime"])

# SPLIT
train = df[df["datetime"] < split_date].copy()
test  = df[df["datetime"] >= split_date].copy()

train = train.sort_values(["location", "datetime"])
test  = test.sort_values(["location", "datetime"])

# INTERPOLATE
train = interpolate_station(train)

# CONTEXT
history = train.groupby("location").tail(24)
test_ctx = pd.concat([history, test])

test_ctx = interpolate_station(test_ctx)

test = test_ctx.reset_index().merge(
    test.reset_index()[["location", "datetime"]],
    on=["location", "datetime"],
    how="inner"
).sort_values(["location", "datetime"])

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nBuilding features...")

train = train.sort_values(["location", "datetime"])
test  = test.sort_values(["location", "datetime"])

# train = train.drop_duplicates(["location", "datetime"])
# test  = test.drop_duplicates(["location", "datetime"])

train = build_features(train, fit=True)
test  = build_features(test, fit=False)


print("Train dataset with features:", train.shape)
print("Test dataset with features:", test.shape)


# ============================================================
# SAVE DATASET
# ============================================================
train = train.set_index("datetime")
test  = test.set_index("datetime")

train.to_parquet("./data/preprocessed/train.parquet")
test.to_parquet("./data/preprocessed/test.parquet")

print("\nDatasets saved.")

print("Train dataset shape:", train.shape)
print("Time range:", train.index.min(), "→", train.index.max())

print("Test dataset shape:", test.shape)
print("Time range:", test.index.min(), "→", test.index.max())

# ============================================================
# SAVE ARTIFACTS
# ============================================================
joblib.dump(FEATURES,"./artifacts/features.pkl")
print("\nSaved feature list.")
