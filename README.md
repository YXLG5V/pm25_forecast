# PM2.5 Forecasting System – Győr

End-to-end machine learning system for short-term PM2.5 air pollution forecasting using time series modeling, feature engineering, and API deployment.

---

## Project Goal

- Predict PM2.5 concentration up to **12 hours ahead**
- Support environmental awareness and decision making
- Target performance: **MAE < 5 µg/m³**

More details: see [domain.md](./domain.md)

---

## Problem Type

- Time series forecasting
- Supervised regression
- Multi-station environmental data

---

## System Architecture

1. Data ingestion (API)
2. Preprocessing  
3. Feature engineering  
4. Model training (Optuna + TimeSeriesSplit)  
5. Evaluation (metrics + SHAP)  
6. Forecasting (recursive)  
7. REST API (FastAPI)

---

## Data

Sources:
- Air pollution measurements: PM2.5, NO2, PM10, SO2 (OpenAQ)
- Weather data: temperature, humidity, wind, precipitation (Open-Meteo)

Key challenges:
- Missing values
- Irregular timestamps
- Measurement noise

---

## Feature Engineering

### Time series features
- Lag features (1, 3, 6, 24 hours)
- Rolling statistics
- Trend and volatility

### Time features
- Cyclical encoding (hour, month)
- Weekend flag
- Heating season

### Weather features
- Change rates
- Physical indices (stagnation, mixing, ventilation)

### Spatial features
- Location encoding
- Normalized coordinates

---

## Models

- RandomForest
- HistGradientBoosting
- LightGBM
- XGBoost
- Ridge (baseline)

### Optimization
- Optuna
- TimeSeriesSplit (time-aware CV)

---

## Validation

- Train/test split (time-based)
- Walk-forward validation (real forecasting simulation)

---

## Evaluation

Metrics:
- MAE, RMSE, R²
- MAPE, SMAPE
- MASE

### Key findings

- Lag features dominate predictions
- Gradient boosting performs best
- Ensemble improves stability
- Model underestimates extreme pollution spikes

---

## Interpretability

- Permutation importance
- SHAP analysis

---

## Forecasting Approach

- Recursive (autoregressive)
- Uses model predictions as future inputs
- Weather forecast integrated

---

## API

Built with FastAPI.

### Endpoint

POST /forecast

### Example request

{
  "location_name": "Gyor Szent Istvan",
  "lat": 47.6875,
  "lon": 17.6504,
  "horizon": 12
}

### Example response

{
  "location": "Gyor Szent Istvan",
  "forecast": [
    {"datetime": "...", "pm25_pred": 12.3}
  ]
}

Run locally:

uvicorn src.inference.app:app --reload

Docs:

http://127.0.0.1:8000/docs

---

## Pipeline

End-to-end pipeline executable:

python run_pipeline.py

Steps:
- get_pollutants_data
- get_weather_data
- preprocessing
- training
- evaluation
- forecasting

---

## Reproducibility

Saved artifacts:
- trained model
- feature list
- location mapping

---

## Testing

- Walk-forward validation
- API test calls
- Script-based execution

---

## Limitations

- Extreme values underpredicted
- Weather data not station-specific
- Short forecast horizon

---

## Future Work

- Better extreme event modeling
- Additional data sources
- Deep learning (LSTM / Transformers)
- Docker
- Frontend implementation
- Database implementation


---

## Tech Stack

### Core & Data Processing
- Python
- pandas
- numpy

### Machine Learning
- scikit-learn
- LightGBM
- XGBoost
- Optuna

### Time Series Analysis
- statsmodels (autocorrelation analysis)

### Data Visualization
- matplotlib
- seaborn

### Model Interpretation
- SHAP

### API & Backend
- FastAPI
- Uvicorn

### Data Ingestion & External APIs
- requests
- httpx
- openaq

### Utilities & Workflow
- joblib (model persistence)
- tqdm (progress tracking)
- python-dotenv (configuration management)
---

## 🤖 AI Usage

AI tools (ChatGPT) were used during the project in the following ways:

- Understanding domain-specific aspects of air pollution and time series modeling
- Assisting with Python syntax and library usage
- Rapid prototyping of ideas (e.g. feature engineering and model experimentation)
- Supporting documentation writing

Important:
All generated code and ideas were manually reviewed, validated, and adapted.  
The final implementation reflects the author's own understanding and design decisions.

---

## Author

Project developed as part of a machine learning course.