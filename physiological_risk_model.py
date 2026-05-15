from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass
class SessionSample:
    bpm: float
    rr_intervals_ms: List[float]
    skin_temp_c: float
    baseline_temp_c: float


def compute_hrv(rr_intervals_ms: Iterable[float]) -> Tuple[float, float]:
    """
    Compute RMSSD and SDNN from RR intervals in milliseconds.
    """
    rr = np.asarray(list(rr_intervals_ms), dtype=float)
    if rr.size < 3:
        raise ValueError("Need at least 3 RR intervals to compute reliable HRV metrics.")

    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(np.square(diffs))))
    sdnn = float(np.std(rr, ddof=1))
    return rmssd, sdnn


def build_feature_vector(sample: SessionSample) -> np.ndarray:
    rmssd, sdnn = compute_hrv(sample.rr_intervals_ms)
    temp_dev = sample.skin_temp_c - sample.baseline_temp_c

    # Feature order used end-to-end in training + inference.
    return np.array([sample.bpm, rmssd, sdnn, sample.skin_temp_c, temp_dev], dtype=float)


def _generate_rr_intervals(bpm: float, variability_ms: float, n_beats: int = 60) -> np.ndarray:
    mean_rr = 60000.0 / bpm
    rr = np.random.normal(loc=mean_rr, scale=variability_ms, size=n_beats)
    return np.clip(rr, 350.0, 1600.0)


def generate_synthetic_dataset(n_samples: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic labeled data based on physiological assumptions.
    label=0 normal: stable BPM, higher HRV, stable temp
    label=1 stress: elevated BPM, lower HRV, larger temp deviation
    """
    if n_samples < 50:
        raise ValueError("Use at least 50 samples for meaningful training.")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    n0 = n_samples // 2
    n1 = n_samples - n0

    X_rows: List[np.ndarray] = []
    y: List[int] = []

    # Normal class
    for _ in range(n0):
        bpm = float(rng.normal(72, 6))
        bpm = float(np.clip(bpm, 55, 90))
        rr = _generate_rr_intervals(bpm=bpm, variability_ms=float(rng.uniform(35, 75)))

        baseline_temp = float(rng.normal(36.2, 0.25))
        skin_temp = float(rng.normal(baseline_temp, 0.12))

        sample = SessionSample(
            bpm=bpm,
            rr_intervals_ms=rr.tolist(),
            skin_temp_c=skin_temp,
            baseline_temp_c=baseline_temp,
        )
        X_rows.append(build_feature_vector(sample))
        y.append(0)

    # Stress / suspicious class
    for _ in range(n1):
        bpm = float(rng.normal(98, 10))
        bpm = float(np.clip(bpm, 78, 135))
        rr = _generate_rr_intervals(bpm=bpm, variability_ms=float(rng.uniform(8, 28)))

        baseline_temp = float(rng.normal(36.2, 0.25))
        # Peripheral temp can dip or fluctuate during stress.
        skin_temp = float(rng.normal(baseline_temp - rng.uniform(0.2, 1.0), 0.25))

        sample = SessionSample(
            bpm=bpm,
            rr_intervals_ms=rr.tolist(),
            skin_temp_c=skin_temp,
            baseline_temp_c=baseline_temp,
        )
        X_rows.append(build_feature_vector(sample))
        y.append(1)

    X = np.vstack(X_rows)
    y_arr = np.array(y, dtype=int)

    # Shuffle
    idx = rng.permutation(len(y_arr))
    return X[idx], y_arr[idx]


class PhysiologicalRiskModel:
    feature_names = ["bpm", "rmssd", "sdnn", "skin_temp_c", "temp_dev_c"]

    def __init__(self, use_xgboost: bool = False, random_state: int = 42):
        self.random_state = random_state
        self.use_xgboost = use_xgboost and XGBClassifier is not None

        self.models: Dict[str, object] = {}
        self.best_model_name: Optional[str] = None

    def _build_models(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "random_forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "logistic_regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
        }

        if self.use_xgboost:
            models["xgboost"] = XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
            )

        return models

    def fit(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.25) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        self.models = self._build_models()
        auc_scores: Dict[str, float] = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)[:, 1]
            auc_scores[name] = float(roc_auc_score(y_val, proba))

        self.best_model_name = max(auc_scores, key=auc_scores.get)
        return auc_scores

    @property
    def best_model(self):
        if not self.best_model_name:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.models[self.best_model_name]

    def predict_probability(self, sample: SessionSample) -> float:
        x = build_feature_vector(sample).reshape(1, -1)
        p = float(self.best_model.predict_proba(x)[0, 1])
        return p

    def predict_label(self, sample: SessionSample, threshold: float = 0.5) -> int:
        return int(self.predict_probability(sample) >= threshold)

    def risk_score(self, sample: SessionSample) -> float:
        p = self.predict_probability(sample)
        return round(p * 100.0, 2)


if __name__ == "__main__":
    # 1) Train on synthetic data assumptions
    X, y = generate_synthetic_dataset(n_samples=320, seed=7)

    risk_model = PhysiologicalRiskModel(use_xgboost=True, random_state=7)
    scores = risk_model.fit(X, y)

    print("Validation ROC-AUC by model:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    print(f"Selected model: {risk_model.best_model_name}")

    # 2) Example real-time session inference
    example_session = SessionSample(
        bpm=104,
        rr_intervals_ms=[590, 582, 605, 575, 588, 579, 584, 592, 577, 586, 581, 590],
        skin_temp_c=35.4,
        baseline_temp_c=36.2,
    )

    probability = risk_model.predict_probability(example_session)
    risk = risk_model.risk_score(example_session)
    label = risk_model.predict_label(example_session, threshold=0.7)

    print(f"Predicted suspicious probability: {probability:.3f}")
    print(f"Risk score (0-100): {risk:.2f}")
    print(f"Predicted label: {label} (0=normal, 1=suspicious)")
