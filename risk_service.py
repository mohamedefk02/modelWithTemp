from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None


FEATURE_COLUMNS = [
    "bpm",
    "skin_temperature",
    "temperature_delta",
]
LABEL_COLUMN = "label"


@dataclass
class TrainingResult:
    accuracy: float
    confusion_matrix: List[List[int]]
    selected_model: str
    total_samples: int
    precision: float
    recall: float
    f1_score: float
    specificity: float
    balanced_accuracy: float


@dataclass
class BenchmarkResult:
    model: str
    folds: int
    mean_accuracy: float
    std_accuracy: float
    fold_accuracies: List[float]


@dataclass
class TestResult:
    accuracy: float
    confusion_matrix: List[List[int]]
    total_samples: int
    precision: float
    recall: float
    f1_score: float
    specificity: float
    balanced_accuracy: float


@dataclass
class StrictEvaluationResult:
    selected_model: str
    cv_folds: int
    train_samples: int
    test_samples: int
    train_class_distribution: Dict[str, int]
    test_class_distribution: Dict[str, int]
    shared_exact_feature_rows: int
    shared_exact_feature_label_rows: int
    shared_subjects: int
    warnings: List[str]
    accuracy: float
    confusion_matrix: List[List[int]]
    precision: float
    recall: float
    f1_score: float
    specificity: float
    balanced_accuracy: float
    cleaned_test_samples: int
    cleaned_dropped_rows: int
    cleaned_accuracy: Optional[float]
    cleaned_confusion_matrix: Optional[List[List[int]]]
    cleaned_precision: Optional[float]
    cleaned_recall: Optional[float]
    cleaned_f1_score: Optional[float]
    cleaned_specificity: Optional[float]
    cleaned_balanced_accuracy: Optional[float]
    decision_threshold: float
    threshold_tuned_for_recall: bool
    target_recall: Optional[float]


class RiskScoringService:
    def __init__(self, model_path: str = "models/risk_model.joblib", use_xgboost: bool = False):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_xgboost = use_xgboost and XGBClassifier is not None

        self.pipeline: Pipeline | None = None
        self.selected_model_name: str | None = None
        self.model_features: List[str] | None = None

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[("num", numeric_pipe, FEATURE_COLUMNS)],
            remainder="drop",
        )

    def _candidate_models(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                random_state=42,
            ),
        }
        if self.use_xgboost:
            models["xgboost"] = XGBClassifier(
                n_estimators=240,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
        return models

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        required = set(FEATURE_COLUMNS + [LABEL_COLUMN])
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in FEATURE_COLUMNS + [LABEL_COLUMN]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[LABEL_COLUMN]).copy()
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
        return df

    def train(self, dataset_path: str) -> TrainingResult:
        df = pd.read_csv(dataset_path)
        self._validate_dataframe(df)

        df = self._clean_dataframe(df)

        X = df[FEATURE_COLUMNS]
        y = df[LABEL_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None,
        )

        best_name = None
        best_pipe = None
        best_acc = -1.0
        best_cm = None
        best_total = 0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_specificity = 0.0
        best_balanced_acc = 0.0

        preprocessor = self._build_preprocessor()
        for model_name, model in self._candidate_models().items():
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            acc = float(accuracy_score(y_test, preds))
            cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
            tn, fp = cm[0]
            fn, tp = cm[1]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            balanced_acc = (recall + specificity) / 2.0

            if acc > best_acc:
                best_acc = acc
                best_cm = cm
                best_name = model_name
                best_pipe = pipe
                best_total = int(len(y_test))
                best_precision = float(precision)
                best_recall = float(recall)
                best_f1 = float(f1)
                best_specificity = float(specificity)
                best_balanced_acc = float(balanced_acc)

        if best_pipe is None or best_name is None or best_cm is None:
            raise RuntimeError("Failed to train any model.")

        self.pipeline = best_pipe
        self.selected_model_name = best_name
        self._save()

        return TrainingResult(
            accuracy=round(best_acc, 4),
            confusion_matrix=best_cm,
            selected_model=best_name,
            total_samples=best_total,
            precision=round(best_precision, 4),
            recall=round(best_recall, 4),
            f1_score=round(best_f1, 4),
            specificity=round(best_specificity, 4),
            balanced_accuracy=round(best_balanced_acc, 4),
        )

    def _save(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("No trained pipeline to save.")
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "selected_model": self.selected_model_name,
                "features": FEATURE_COLUMNS,
            },
            self.model_path,
        )

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        payload = joblib.load(self.model_path)
        payload_features = payload.get("features")
        if payload_features is not None and list(payload_features) != FEATURE_COLUMNS:
            raise RuntimeError(
                "Saved model features do not match current API features. "
                f"saved={list(payload_features)} current={FEATURE_COLUMNS}. "
                "Call /train again to rebuild the model."
            )
        self.pipeline = payload["pipeline"]
        self.selected_model_name = payload.get("selected_model", "unknown")
        self.model_features = payload_features if payload_features is not None else FEATURE_COLUMNS

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            self.load()

    def predict_one(self, features: Dict[str, float], threshold: float = 0.5) -> Dict[str, float | int]:
        self._ensure_loaded()
        assert self.pipeline is not None

        row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        proba = float(self.pipeline.predict_proba(row)[0, 1])
        label = int(proba >= threshold)
        risk_score = round(proba * 100.0, 2)

        return {"label": label, "risk_score": risk_score}

    def predict_batch(self, rows: List[Dict[str, float]], threshold: float = 0.5) -> List[Dict[str, float | int]]:
        self._ensure_loaded()
        assert self.pipeline is not None

        batch_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        probas = self.pipeline.predict_proba(batch_df)[:, 1]

        results: List[Dict[str, float | int]] = []
        for p in probas:
            p_val = float(p)
            results.append(
                {
                    "label": int(p_val >= threshold),
                    "risk_score": round(p_val * 100.0, 2),
                }
            )

        return results

    def benchmark_accuracy(self, dataset_path: str, folds: int = 5) -> BenchmarkResult:
        if folds < 2:
            raise ValueError("folds must be >= 2")

        df = pd.read_csv(dataset_path)
        self._validate_dataframe(df)

        for col in FEATURE_COLUMNS + [LABEL_COLUMN]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[LABEL_COLUMN]).copy()
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

        X = df[FEATURE_COLUMNS]
        y = df[LABEL_COLUMN]

        if len(y.unique()) < 2:
            raise ValueError("Benchmark requires at least 2 classes in label column.")

        min_class = int(y.value_counts().min())
        if folds > min_class:
            raise ValueError(
                f"folds={folds} is too high for class counts. Use folds <= {min_class}."
            )

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        preprocessor = self._build_preprocessor()
        best_name = None
        best_scores = None
        best_mean = -1.0

        for model_name, model in self._candidate_models().items():
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=None)
            mean_score = float(np.mean(scores))
            if mean_score > best_mean:
                best_mean = mean_score
                best_scores = scores
                best_name = model_name

        if best_name is None or best_scores is None:
            raise RuntimeError("Failed to benchmark candidate models.")

        return BenchmarkResult(
            model=best_name,
            folds=folds,
            mean_accuracy=round(float(np.mean(best_scores)), 4),
            std_accuracy=round(float(np.std(best_scores, ddof=1)), 4) if len(best_scores) > 1 else 0.0,
            fold_accuracies=[round(float(s), 4) for s in best_scores.tolist()],
        )

    def evaluate_test_set(self, dataset_path: str) -> TestResult:
        self._ensure_loaded()
        assert self.pipeline is not None

        df = pd.read_csv(dataset_path)
        self._validate_dataframe(df)

        for col in FEATURE_COLUMNS + [LABEL_COLUMN]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[LABEL_COLUMN]).copy()
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

        X = df[FEATURE_COLUMNS]
        y = df[LABEL_COLUMN]
        if len(y) == 0:
            raise ValueError("Test dataset is empty after cleaning.")

        preds = self.pipeline.predict(X)
        acc = float(accuracy_score(y, preds))
        cm = confusion_matrix(y, preds, labels=[0, 1]).tolist()
        tn, fp = cm[0]
        fn, tp = cm[1]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        balanced_acc = (recall + specificity) / 2.0

        return TestResult(
            accuracy=round(acc, 4),
            confusion_matrix=cm,
            total_samples=int(len(y)),
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1_score=round(float(f1), 4),
            specificity=round(float(specificity), 4),
            balanced_accuracy=round(float(balanced_acc), 4),
        )

    def strict_evaluate(
        self,
        train_dataset_path: str,
        test_dataset_path: str,
        cv_folds: int = 5,
        subject_column: Optional[str] = None,
        drop_overlap: bool = False,
        tune_threshold_for_recall: bool = False,
        target_recall: float = 0.9,
    ) -> StrictEvaluationResult:
        if cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if not (0.0 < target_recall <= 1.0):
            raise ValueError("target_recall must be in (0, 1].")

        train_df = pd.read_csv(train_dataset_path)
        test_df = pd.read_csv(test_dataset_path)
        self._validate_dataframe(train_df)
        self._validate_dataframe(test_df)
        train_df = self._clean_dataframe(train_df)
        test_df = self._clean_dataframe(test_df)

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Train/test dataset is empty after cleaning.")
        if len(train_df[LABEL_COLUMN].unique()) < 2:
            raise ValueError("Train dataset needs at least 2 classes.")
        if len(test_df[LABEL_COLUMN].unique()) < 2:
            raise ValueError("Test dataset needs at least 2 classes.")

        train_features_only = set(map(tuple, train_df[FEATURE_COLUMNS].round(6).to_numpy().tolist()))
        test_features_only = set(map(tuple, test_df[FEATURE_COLUMNS].round(6).to_numpy().tolist()))
        shared_feature_rows = len(train_features_only.intersection(test_features_only))

        train_feat_lbl = set(
            map(tuple, train_df[FEATURE_COLUMNS + [LABEL_COLUMN]].round(6).to_numpy().tolist())
        )
        test_feat_lbl = set(
            map(tuple, test_df[FEATURE_COLUMNS + [LABEL_COLUMN]].round(6).to_numpy().tolist())
        )
        shared_feature_label_rows = len(train_feat_lbl.intersection(test_feat_lbl))

        shared_subjects = 0
        if subject_column and subject_column in train_df.columns and subject_column in test_df.columns:
            train_subjects = set(train_df[subject_column].dropna().astype(str).tolist())
            test_subjects = set(test_df[subject_column].dropna().astype(str).tolist())
            shared_subjects = len(train_subjects.intersection(test_subjects))

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[LABEL_COLUMN]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[LABEL_COLUMN]

        min_class = int(y_train.value_counts().min())
        if cv_folds > min_class:
            raise ValueError(f"cv_folds={cv_folds} is too high. Use cv_folds <= {min_class}.")

        use_subject_groups = bool(
            subject_column and subject_column in train_df.columns and subject_column in test_df.columns
        )
        groups = train_df[subject_column].astype(str).values if use_subject_groups else None

        if use_subject_groups:
            unique_groups = np.unique(groups)
            if len(unique_groups) < cv_folds:
                raise ValueError(
                    f"Not enough unique groups for subject-aware CV: {len(unique_groups)} groups, "
                    f"cv_folds={cv_folds}. Reduce cv_folds."
                )
            cv = GroupKFold(n_splits=cv_folds)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        preprocessor = self._build_preprocessor()
        best_name = None
        best_score = -1.0
        best_pipe = None

        for model_name, model in self._candidate_models().items():
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
            scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring="balanced_accuracy",
                groups=groups,
                n_jobs=None,
            )
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_name = model_name
                best_pipe = pipe

        if best_pipe is None or best_name is None:
            raise RuntimeError("Failed to select a model.")

        decision_threshold = 0.45
        threshold_tuned = False
        threshold_tuning_warning: Optional[str] = None
        if tune_threshold_for_recall:
            if use_subject_groups:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                split = next(gss.split(X_train, y_train, groups=groups))
                train_idx, val_idx = split
                X_subtrain, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_subtrain, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            else:
                X_subtrain, X_val, y_subtrain, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_train,
                )

            if len(np.unique(y_val)) < 2:
                threshold_tuning_warning = "Validation split has <2 classes; threshold tuning skipped."
            else:
                tune_pipe = Pipeline(steps=[("prep", preprocessor), ("model", self._candidate_models()[best_name])])
                tune_pipe.fit(X_subtrain, y_subtrain)
                val_proba = tune_pipe.predict_proba(X_val)[:, 1]

                best_thr = 0.5
                for thr in np.unique(np.round(val_proba, 6))[::-1]:
                    preds_thr = (val_proba >= thr).astype(int)
                    cm_thr = confusion_matrix(y_val, preds_thr, labels=[0, 1]).tolist()
                    fn_thr, tp_thr = cm_thr[1]
                    recall_thr = tp_thr / (tp_thr + fn_thr) if (tp_thr + fn_thr) > 0 else 0.0
                    if recall_thr >= target_recall:
                        best_thr = float(thr)
                        break
                decision_threshold = best_thr
                threshold_tuned = True

        best_pipe.fit(X_train, y_train)
        test_proba = best_pipe.predict_proba(X_test)[:, 1]
        preds = (test_proba >= decision_threshold).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
        tn, fp = cm[0]
        fn, tp = cm[1]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        balanced_acc = (recall + specificity) / 2.0
        acc = float(accuracy_score(y_test, preds))

        warnings: List[str] = []
        if shared_feature_rows > 0:
            warnings.append(f"Train/test share {shared_feature_rows} exact feature rows.")
        if shared_feature_label_rows > 0:
            warnings.append(f"Train/test share {shared_feature_label_rows} exact feature+label rows.")
        if subject_column and (subject_column not in train_df.columns or subject_column not in test_df.columns):
            warnings.append(f"subject_column '{subject_column}' was requested but not present in both datasets.")
        if shared_subjects > 0:
            warnings.append(f"Train/test share {shared_subjects} subject IDs.")
        if threshold_tuning_warning:
            warnings.append(threshold_tuning_warning)
        if threshold_tuned:
            warnings.append(
                f"Decision threshold tuned for recall target {target_recall:.2f}: threshold={decision_threshold:.4f}"
            )

        cleaned_test_samples = int(len(test_df))
        cleaned_dropped_rows = 0
        cleaned_accuracy: Optional[float] = None
        cleaned_cm: Optional[List[List[int]]] = None
        cleaned_precision: Optional[float] = None
        cleaned_recall: Optional[float] = None
        cleaned_f1: Optional[float] = None
        cleaned_specificity: Optional[float] = None
        cleaned_balanced_acc: Optional[float] = None

        if drop_overlap:
            train_feature_index = pd.MultiIndex.from_frame(train_df[FEATURE_COLUMNS].round(6))
            test_feature_index = pd.MultiIndex.from_frame(test_df[FEATURE_COLUMNS].round(6))
            keep_mask = ~test_feature_index.isin(train_feature_index)
            cleaned_df = test_df.loc[keep_mask].copy()
            cleaned_dropped_rows = int((~keep_mask).sum())
            cleaned_test_samples = int(len(cleaned_df))

            if cleaned_test_samples > 0 and len(cleaned_df[LABEL_COLUMN].unique()) >= 2:
                X_test_clean = cleaned_df[FEATURE_COLUMNS]
                y_test_clean = cleaned_df[LABEL_COLUMN]
                proba_clean = best_pipe.predict_proba(X_test_clean)[:, 1]
                preds_clean = (proba_clean >= decision_threshold).astype(int)
                cm_clean = confusion_matrix(y_test_clean, preds_clean, labels=[0, 1]).tolist()
                tn_c, fp_c = cm_clean[0]
                fn_c, tp_c = cm_clean[1]

                precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                specificity_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
                f1_c = (
                    (2 * precision_c * recall_c / (precision_c + recall_c))
                    if (precision_c + recall_c) > 0
                    else 0.0
                )
                balanced_acc_c = (recall_c + specificity_c) / 2.0

                cleaned_accuracy = round(float(accuracy_score(y_test_clean, preds_clean)), 4)
                cleaned_cm = cm_clean
                cleaned_precision = round(float(precision_c), 4)
                cleaned_recall = round(float(recall_c), 4)
                cleaned_f1 = round(float(f1_c), 4)
                cleaned_specificity = round(float(specificity_c), 4)
                cleaned_balanced_acc = round(float(balanced_acc_c), 4)
            elif cleaned_test_samples == 0:
                warnings.append("All test rows were removed by overlap cleaning.")
            else:
                warnings.append("Cleaned test set has <2 classes; cleaned metrics not computed.")

        return StrictEvaluationResult(
            selected_model=best_name,
            cv_folds=cv_folds,
            train_samples=int(len(train_df)),
            test_samples=int(len(test_df)),
            train_class_distribution={
                "0": int((y_train == 0).sum()),
                "1": int((y_train == 1).sum()),
            },
            test_class_distribution={
                "0": int((y_test == 0).sum()),
                "1": int((y_test == 1).sum()),
            },
            shared_exact_feature_rows=shared_feature_rows,
            shared_exact_feature_label_rows=shared_feature_label_rows,
            shared_subjects=shared_subjects,
            warnings=warnings,
            accuracy=round(acc, 4),
            confusion_matrix=cm,
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1_score=round(float(f1), 4),
            specificity=round(float(specificity), 4),
            balanced_accuracy=round(float(balanced_acc), 4),
            cleaned_test_samples=cleaned_test_samples,
            cleaned_dropped_rows=cleaned_dropped_rows,
            cleaned_accuracy=cleaned_accuracy,
            cleaned_confusion_matrix=cleaned_cm,
            cleaned_precision=cleaned_precision,
            cleaned_recall=cleaned_recall,
            cleaned_f1_score=cleaned_f1,
            cleaned_specificity=cleaned_specificity,
            cleaned_balanced_accuracy=cleaned_balanced_acc,
            decision_threshold=round(float(decision_threshold), 6),
            threshold_tuned_for_recall=threshold_tuned,
            target_recall=target_recall if threshold_tuned else None,
        )
