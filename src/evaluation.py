"""Cross-validated evaluation of classification pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             make_scorer, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate

# Threshold-free metrics (roc_auc, pr_auc) use predicted scores; the others use
# hard 0/1 predictions at the model's default threshold (0.5 for probabilities).
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
HARD_ONLY_SCORING = {k: v for k, v in SCORING.items() if k not in ("roc_auc", "pr_auc")}
METRIC_LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
                 "precision": "Precision", "recall": "Recall", "f1": "F1",
                 "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


@dataclass
class CVResult:
    folds: pd.DataFrame        # one row per fold: test_*, train_*, fit_time, score_time
    oof_scores: np.ndarray     # out-of-fold positive-class score for every training row
    oof_pred: np.ndarray       # out-of-fold 0/1 prediction at the default threshold
    has_proba: bool            # True if oof_scores are probabilities (else decision values)


def positive_scores(estimator, X) -> tuple[np.ndarray, bool]:
    """Positive-class probability if the model has one, else its decision value."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1], True
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X), False
    return estimator.predict(X).astype(float), False


def evaluate_cv(pipe, X: pd.DataFrame, y: pd.Series, folds, n_jobs: int | None = None,
                train_score: bool = True) -> CVResult:
    """Fit and score `pipe` on fixed folds with scikit-learn's cross_validate,
    then rebuild out-of-fold predictions from the fitted fold models."""
    has_scores = hasattr(pipe, "predict_proba") or hasattr(pipe, "decision_function")
    res = cross_validate(pipe, X, y, cv=folds, scoring=SCORING if has_scores else HARD_ONLY_SCORING,
                         return_train_score=train_score, return_estimator=True,
                         return_indices=True, n_jobs=n_jobs)
    oof_scores = np.full(len(y), np.nan)
    oof_pred = np.zeros(len(y), dtype=int)
    has_proba = False
    for est, test_idx in zip(res["estimator"], res["indices"]["test"]):
        Xt = X.iloc[test_idx]
        oof_scores[test_idx], has_proba = positive_scores(est, Xt)
        oof_pred[test_idx] = est.predict(Xt)
    folds_df = pd.DataFrame({k: v for k, v in res.items() if k not in ("estimator", "indices")})
    return CVResult(folds_df, oof_scores, oof_pred, has_proba)


def summarize(results: dict[str, CVResult], metrics=tuple(SCORING)) -> pd.DataFrame:
    """Mean and standard deviation across folds, one row per model."""
    rows = {}
    for name, r in results.items():
        row = {}
        for m in metrics:
            col = f"test_{m}"
            if col in r.folds:
                row[m] = r.folds[col].mean()
                row[f"{m}_std"] = r.folds[col].std()
            tcol = f"train_{m}"
            if tcol in r.folds:
                row[f"train_{m}"] = r.folds[tcol].mean()
        row["fit_time_s"] = r.folds["fit_time"].mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def metrics_at_threshold(y_true, scores, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return {"threshold": threshold,
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, scores),
            "pr_auc": average_precision_score(y_true, scores)}


def best_f1_threshold(y_true, scores) -> float:
    """Score threshold that maximises F1 on the given (out-of-fold) predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def pick_best_variant(table: pd.DataFrame) -> str:
    """Choose one variant from a `summarize` table with a "one standard error" rule:

    1. Variants whose ROC-AUC is within one fold-standard-deviation of the best
       ROC-AUC count as tied (their difference is within cross-validation noise).
    2. Among tied variants, prefer the highest F1 (rounded to 2 decimals): the one
       that actually finds high-cost people at the default threshold.
    3. If still tied, prefer the fastest to train.
    """
    scored = table.dropna(subset=["roc_auc"])
    top = scored["roc_auc"].idxmax()
    tied = scored[scored["roc_auc"] >= scored.loc[top, "roc_auc"] - scored.loc[top, "roc_auc_std"]]
    f1 = tied["f1"].round(2)
    tied = tied[f1 == f1.max()]
    return str(tied["fit_time_s"].idxmin())
