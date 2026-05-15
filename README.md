# Physiological Risk Scoring API

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Build Real Training Data (WESAD)

1. Download and extract WESAD so folders look like `.../WESAD/S2/S2.pkl`, `.../WESAD/S3/S3.pkl`, etc.
2. Run:

```bash
python prepare_wesad.py --wesad-root "C:/path/to/WESAD" --output data/wesad_features.csv --window-sec 60 --step-sec 10
```

This creates CSV with schema:
`subject_id,bpm,skin_temperature,temperature_delta,label`

## Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /` -> health check
- `POST /train`
- `POST /benchmark`
- `POST /test`
- `POST /predict`
- `POST /batch_predict`

## Example Train Request

```json
{
  "dataset_path": "data/wesad_features.csv"
}
```

## Example Predict Request

```json
{
  "bpm": 88,
  "skin_temperature": 36.3,
  "temperature_delta": -0.3
}
```

## Example Benchmark Request

```json
{
  "dataset_path": "data/processed/wesad_like_train_small.csv",
  "folds": 5
}
```

## Example Test Request

```json
{
  "dataset_path": "data/test.csv"
}
```

## Notes

- Model uses missing-value handling (`SimpleImputer`) and normalization (`StandardScaler`).
- Candidate models: Random Forest, Logistic Regression, optional XGBoost.
- Risk score is computed as `probability_of_class_1 * 100`.
- Saved model path: `models/risk_model.joblib`.
- WESAD mapping used by converter: stress label `2 -> 1`, baseline/amusement/meditation (`1/3/4 -> 0`).

## ESP32 Sensor Bridge

Run backend first, then run:

```bash
python sensor_bridge.py --port COM5 --baud 115200 --predict-url http://127.0.0.1:8000/predict
```

Optional forward to another API:

```bash
python sensor_bridge.py --port COM5 --baud 115200 --predict-url http://127.0.0.1:8000/predict --forward-url http://127.0.0.1:9000/ingest
```
