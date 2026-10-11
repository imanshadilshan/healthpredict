"""Feature definitions, reserved-code cleaning and engineered features for the
high-cost classification task.

Every choice here comes from a documented EDA decision (notebooks/02_eda.ipynb,
reports/results/eda_decisions.json). All functions are stateless row-wise
operations, so they can safely run before the train/test split: nothing is
learned from the data here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- raw candidate features
NUMERIC = ["AGELAST", "FAMSZE22"]
SKEWED_NUMERIC = ["POVLEV22"]                 # long right tail, can be negative -> Yeo-Johnson
ORDINAL = ["RTHLTH53", "MNHLTH53", "POVCAT22"]  # ordered scales (1 = best ... 5 = worst / poorest)
NOMINAL = ["SEX", "RACETHX", "REGION22", "MARRY22X", "HIDEG", "INSCOV22",
           "EMPST53", "HAVEUS42", "BORNUSA", "OFTSMK53"]
CONDITION_FLAGS = ["HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "EMPHDX",
                   "CHOLDX", "CANCERDX", "DIABDX_M18", "ARTHDX", "ASTHDX"]
LIMITATION_FLAGS = ["ACTLIM31", "WLKLIM31", "COGLIM31", "ADLHLP31", "IADLHP31"]
INSURANCE_FLAGS = ["MCARE22", "MCAID22", "UNINS22"]
FLAGS = CONDITION_FLAGS + LIMITATION_FLAGS + INSURANCE_FLAGS   # 1 = yes, 2 = no, -1 = not asked

# ---------------------------------------------------------------- engineered features
ENGINEERED = {
    "N_CONDITIONS": "Number of the 12 chronic conditions the person reports (0-12)",
    "N_LIMITATIONS": "Number of the 5 functional limitations the person reports (0-5)",
    "ANY_LIMITATION": "1 if the person has at least one functional limitation",
    "OUT_OF_SCOPE_1231": "1 if the person was not in the survey population on 31 Dec 2022 "
                         "(died or institutionalized), detected by REGION22 = -1",
    "AGE_BAND": "Age group 0-17, 18-34, 35-49, 50-64, 65-79, 80+ coded 0-5",
}
ENGINEERED_NUMERIC = list(ENGINEERED)

AGE_BINS = [-1, 17, 34, 49, 64, 79, 120]

# Reserved codes (documentation Table 2)
MISSING_CODES = [-7, -8, -15]   # asked but no usable answer -> NaN, imputed later
NOT_ASKED = -1                  # skipped by design -> kept as its own category where informative

ALL_CANDIDATES = NUMERIC + SKEWED_NUMERIC + ORDINAL + NOMINAL + FLAGS + ENGINEERED_NUMERIC


def clean_reserved_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Turn MEPS reserved codes into values a model can use.

    - Nominal and yes/no columns: -7/-8/-15 become NaN (imputed later); -1 stays,
      because "not asked" (e.g. a child skipped for adult questions) is information.
    - Ordinal scales: -1/-7/-8 become NaN. A "not asked" level has no place on a
      1-5 scale; a missing-indicator column added by the imputer keeps that signal.
    - FAMSZE22: -1 (person out of scope at year end) becomes NaN.
    - POVLEV22: unchanged. Its negative values are real (family income losses).
    """
    out = df.copy()
    for col in NOMINAL + FLAGS:
        out[col] = out[col].where(~out[col].isin(MISSING_CODES), np.nan)
    for col in ORDINAL:
        out[col] = out[col].where(out[col] > 0, np.nan)
    out["FAMSZE22"] = out["FAMSZE22"].where(out["FAMSZE22"] > 0, np.nan)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the documented engineered features (see ENGINEERED)."""
    out = df.copy()
    out["N_CONDITIONS"] = (out[CONDITION_FLAGS] == 1).sum(axis=1)
    out["N_LIMITATIONS"] = (out[LIMITATION_FLAGS] == 1).sum(axis=1)
    out["ANY_LIMITATION"] = (out["N_LIMITATIONS"] > 0).astype(int)
    out["OUT_OF_SCOPE_1231"] = (out["REGION22"] == NOT_ASKED).astype(int)
    out["AGE_BAND"] = pd.cut(out["AGELAST"], AGE_BINS, labels=False).astype(int)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw analysis-population rows -> candidate feature table (engineering first,
    so the counts see the original 1/2/-1 codes, then code cleaning)."""
    return clean_reserved_codes(add_engineered_features(df))[ALL_CANDIDATES]


def feature_groups(selected: list[str]) -> dict[str, list[str]]:
    """Split a list of selected features into the preprocessing groups."""
    numeric = set(NUMERIC + ENGINEERED_NUMERIC)
    return {
        "numeric": [c for c in selected if c in numeric],
        "skewed": [c for c in selected if c in SKEWED_NUMERIC],
        "ordinal": [c for c in selected if c in ORDINAL],
        "categorical": [c for c in selected if c in NOMINAL + FLAGS],
    }
