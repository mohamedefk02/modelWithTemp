from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risk_service import FEATURE_COLUMNS, RiskScoringService


app = FastAPI(title="Physiological Risk Scoring API", version="1.0.0")
service = RiskScoringService(model_path="models/risk_model.joblib", use_xgboost=True)


class PhysiologicalInput(BaseModel):
    bpm: float = Field(..., gt=0)
    skin_temperature: float = Field(..., gt=20, lt=45)
    temperature_delta: float


class TrainRequest(BaseModel):
    dataset_path: str


class BenchmarkRequest(BaseModel):
    dataset_path: str
    folds: int = Field(default=5, ge=2, le=10)


class TestRequest(BaseModel):
    dataset_path: str


class StrictEvalRequest(BaseModel):
    train_dataset_path: str
    test_dataset_path: str
    cv_folds: int = Field(default=5, ge=2, le=10)
    subject_column: str | None = None
    drop_overlap: bool = False
    tune_threshold_for_recall: bool = False
    target_recall: float = Field(default=0.9, gt=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    data: List[PhysiologicalInput]


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train_model(request: TrainRequest):
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

    try:
        result = service.train(str(dataset_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "model trained",
        "accuracy": result.accuracy,
        "confusion_matrix": result.confusion_matrix,
        "selected_model": result.selected_model,
        "total_samples": result.total_samples,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "specificity": result.specificity,
        "balanced_accuracy": result.balanced_accuracy,
    }


@app.post("/benchmark")
def benchmark_model(request: BenchmarkRequest):
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

    try:
        result = service.benchmark_accuracy(str(dataset_path), folds=request.folds)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "benchmark complete",
        "model": result.model,
        "folds": result.folds,
        "mean_accuracy": result.mean_accuracy,
        "std_accuracy": result.std_accuracy,
        "fold_accuracies": result.fold_accuracies,
    }


@app.post("/test")
def test_model(request: TestRequest):
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

    try:
        result = service.evaluate_test_set(str(dataset_path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not trained yet. Call /train first.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "test complete",
        "accuracy": result.accuracy,
        "confusion_matrix": result.confusion_matrix,
        "total_samples": result.total_samples,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "specificity": result.specificity,
        "balanced_accuracy": result.balanced_accuracy,
    }


@app.post("/strict_evaluate")
def strict_evaluate_model(request: StrictEvalRequest):
    train_path = Path(request.train_dataset_path)
    test_path = Path(request.test_dataset_path)
    if not train_path.exists():
        raise HTTPException(status_code=404, detail=f"Train dataset not found: {train_path}")
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"Test dataset not found: {test_path}")

    try:
        result = service.strict_evaluate(
            train_dataset_path=str(train_path),
            test_dataset_path=str(test_path),
            cv_folds=request.cv_folds,
            subject_column=request.subject_column,
            drop_overlap=request.drop_overlap,
            tune_threshold_for_recall=request.tune_threshold_for_recall,
            target_recall=request.target_recall,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "strict evaluation complete",
        "selected_model": result.selected_model,
        "cv_folds": result.cv_folds,
        "train_samples": result.train_samples,
        "test_samples": result.test_samples,
        "train_class_distribution": result.train_class_distribution,
        "test_class_distribution": result.test_class_distribution,
        "shared_exact_feature_rows": result.shared_exact_feature_rows,
        "shared_exact_feature_label_rows": result.shared_exact_feature_label_rows,
        "shared_subjects": result.shared_subjects,
        "warnings": result.warnings,
        "accuracy": result.accuracy,
        "confusion_matrix": result.confusion_matrix,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "specificity": result.specificity,
        "balanced_accuracy": result.balanced_accuracy,
        "cleaned_test_samples": result.cleaned_test_samples,
        "cleaned_dropped_rows": result.cleaned_dropped_rows,
        "cleaned_accuracy": result.cleaned_accuracy,
        "cleaned_confusion_matrix": result.cleaned_confusion_matrix,
        "cleaned_precision": result.cleaned_precision,
        "cleaned_recall": result.cleaned_recall,
        "cleaned_f1_score": result.cleaned_f1_score,
        "cleaned_specificity": result.cleaned_specificity,
        "cleaned_balanced_accuracy": result.cleaned_balanced_accuracy,
        "decision_threshold": result.decision_threshold,
        "threshold_tuned_for_recall": result.threshold_tuned_for_recall,
        "target_recall": result.target_recall,
    }


@app.post("/predict")
def predict(payload: PhysiologicalInput):
    try:
        result = service.predict_one(payload.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not trained yet. Call /train first.")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.post("/batch_predict")
def batch_predict(payload: BatchPredictRequest):
    if len(payload.data) == 0:
        raise HTTPException(status_code=400, detail="data must contain at least 1 sample")

    rows = [x.model_dump() for x in payload.data]
    for row in rows:
        missing = set(FEATURE_COLUMNS).difference(row.keys())
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields: {sorted(missing)}")

    try:
        results = service.predict_batch(rows)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not trained yet. Call /train first.")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"results": results}
