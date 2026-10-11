"""Train/test split, target creation and the scikit-learn preprocessing pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from .data_loader import PROCESSED_DIR, PROJECT_ROOT, TARGET_COL
from .feature_engineering import feature_groups

RANDOM_STATE = 42
THRESHOLD_QUANTILE = 0.80
GROUP_COL = "DUID"          # household: members always stay on the same side of a split
LABEL_COL = "HIGH_COST"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "reports" / "results" / "selected_features.json"


def grouped_train_test_split(df: pd.DataFrame, test_size: float = 0.20,
                             random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) positions.

    - Grouped by household (DUID): people who live together share family-level
      features, so splitting them up would let the model "recognise" a household
      in the test set.
    - Stratified by spending decile: both sides get the same mix of low, medium
      and very high spenders. Deciles (not the final 0/1 label) are used because
      the label's threshold must be learned from the training rows only.
    """
    n_splits = round(1 / test_size)
    strata = pd.qcut(df[TARGET_COL].rank(method="first"), 10, labels=False)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, strata, groups=df[GROUP_COL]))
    return train_idx, test_idx


def high_cost_threshold(train_spend: pd.Series, q: float = THRESHOLD_QUANTILE) -> float:
    """P80 of total expenditure, computed on training rows only."""
    return float(train_spend.quantile(q))


def make_label(spend: pd.Series, threshold: float) -> pd.Series:
    return (spend >= threshold).astype(int).rename(LABEL_COL)


def cv_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5,
             random_state: int = RANDOM_STATE) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed, household-grouped, stratified CV folds. Every model is scored on
    exactly these folds so the comparison is fair."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(X, y, groups=groups))


def build_preprocessor(selected: list[str], min_category_frequency: int = 50) -> ColumnTransformer:
    """ColumnTransformer that is fitted on training data only (inside a Pipeline).

    numeric     -> median imputation -> standard scaling
    skewed      -> median imputation -> Yeo-Johnson power transform (+ scaling)
    ordinal     -> median imputation (+ missing indicator) -> standard scaling
    categorical -> most-frequent imputation -> one-hot encoding
                   (categories rarer than `min_category_frequency` rows are pooled
                   into one "infrequent" column; unseen categories map there too)
    """
    g = feature_groups(selected)
    transformers = []
    if g["numeric"]:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), g["numeric"]))
    if g["skewed"]:
        transformers.append(("skew", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson")),
        ]), g["skewed"]))
    if g["ordinal"]:
        transformers.append(("ord", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), g["ordinal"]))
    if g["categorical"]:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", drop="if_binary",
                                     min_frequency=min_category_frequency, sparse_output=False)),
        ]), g["categorical"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def save_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)


def load_selected_features() -> list[str]:
    return json.loads(SELECTED_FEATURES_PATH.read_text())["selected_features"]
