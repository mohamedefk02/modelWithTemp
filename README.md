# Physiological Risk Scoring API

A FastAPI-based risk scoring service for physiological sensor data. This repository includes tools to train and evaluate a binary stress detection model using WESAD-derived features, synthetic stress-like data generation, and an ESP32 serial bridge for live predictions.

## Features

- Train a risk scoring model from CSV datasets
- Benchmark using k-fold cross-validation
- Test on holdout datasets
- Strict evaluation with subject-aware splitting and overlap removal
- Real-time bridged predictions from ESP32 sensor serial data
- Synthetic WESAD-like dataset generation for model experimentation

## Repository Structure

- `app.py` - FastAPI application and request handlers
- `risk_service.py` - model training, evaluation, benchmarking, and prediction logic
- `sensor_bridge.py` - serial bridge for ESP32 sensor output to `/predict`
- `prepare_wesad.py` - convert raw WESAD pickle data into CSV training features
- `generate_wesad_like_data.py` - generate synthetic WESAD-like CSV datasets
- `data/` - sample datasets and generated outputs
- `models/` - saved `risk_model.joblib`

## Requirements

Install Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `GET /` - health check
- `POST /train` - train the risk model
- `POST /benchmark` - run k-fold benchmark
- `POST /test` - evaluate on a test dataset
- `POST /strict_evaluate` - strict subject-aware evaluation
- `POST /predict` - single-sample prediction
- `POST /batch_predict` - batch prediction for multiple samples

## Example Requests

### Train

```json
{
  "dataset_path": "data/wesad_features.csv"
}
```

### Predict

```json
{
  "bpm": 88,
  "skin_temperature": 36.3,
  "temperature_delta": -0.3
}
```

### Benchmark

```json
{
  "dataset_path": "data/processed/wesad_like_train_small.csv",
  "folds": 5
}
```

### Test

```json
{
  "dataset_path": "data/test.csv"
}
```

### Strict Evaluate

```json
{
  "train_dataset_path": "data/processed/wesad_like_train.csv",
  "test_dataset_path": "data/test.csv",
  "cv_folds": 5,
  "subject_column": "subject_id",
  "drop_overlap": false,
  "tune_threshold_for_recall": true,
  "target_recall": 0.9
}
```

### Batch Predict

```json
{
  "data": [
    {"bpm": 80, "skin_temperature": 36.0, "temperature_delta": -0.1},
    {"bpm": 95, "skin_temperature": 36.8, "temperature_delta": 0.3}
  ]
}
```

## Generate Training Data

### Convert WESAD data

Prepare real WESAD-derived features from raw pickle files:

```bash
python prepare_wesad.py --wesad-root "C:/path/to/WESAD" --output data/wesad_features.csv --window-sec 60 --step-sec 10
```

The output CSV includes:

- `subject_id`
- `bpm`
- `skin_temperature`
- `temperature_delta`
- `label`

Stress is mapped to `label = 1` and baseline/amusement/meditation is mapped to `label = 0`.

### Create synthetic WESAD-like data

```bash
python generate_wesad_like_data.py --output data/processed/wesad_like_train.csv --target-mb 200
```

## ESP32 Sensor Bridge

Use `sensor_bridge.py` to stream serial sensor readings to the model prediction endpoint.

```bash
python sensor_bridge.py --port COM5 --baud 115200 --predict-url http://127.0.0.1:8000/predict
```

Optional advanced bridge command:

```bash
python sensor_bridge.py \
  --port COM4 \
  --baud 115200 \
  --predict-url http://127.0.0.1:8000/predict \
  --window-seconds 10 \
  --min-bpm 45 \
  --max-bpm 180 \
  --min-ir 50000 \
  --min-valid-samples 20
```

Forward predictions to another API:

```bash
python sensor_bridge.py --port COM5 --baud 115200 --predict-url http://127.0.0.1:8000/predict --forward-url http://127.0.0.1:9000/ingest
```

## Notes

- The model pipeline uses `SimpleImputer` and `StandardScaler` for preprocessing.
- Candidate models include Random Forest and Logistic Regression, with optional XGBoost when installed.
- The saved model file is `models/risk_model.joblib`.
- `risk_service.py` validates required feature columns before training or prediction.

