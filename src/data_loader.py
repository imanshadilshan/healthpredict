"""Loading and basic helpers for the MEPS HC-243 (2022 Full Year Consolidated) dataset.

The raw Excel file is never modified. The first load reads the xlsx (~2.5 min) and
writes an identical Parquet copy to data/processed/ so later loads take ~1 second.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_XLSX = RAW_DIR / "h243.xlsx"
CACHE_PARQUET = PROCESSED_DIR / "h243.parquet"

TARGET_COL = "TOTEXP22"
ID_COL = "DUPERSID"
WEIGHT_COL = "PERWT22F"

# Reserved codes from the HC-243 documentation, Section 2.2, Table 2.
RESERVED_CODES = {
    -1: "Inapplicable",
    -2: "Determined in previous round",
    -7: "Refused",
    -8: "Don't know",
    -10: "Hourly wage top-coded",
    -13: "Initial wage imputed",
    -15: "Cannot be computed",
}
# Codes meaning "asked but no usable answer" (true missing), as opposed to -1 (skipped by design).
MISSING_CODES = [-7, -8, -15]

# Variables where negative values are genuine amounts (losses), not reserved codes.
SIGNED_INCOME_COLS = ["TTLP22X", "FAMINC22", "POVLEV22", "BUSNP22X", "SALEP22X", "TRSTP22X"]

# First column of each variable group, in file order (documentation Section 2.5).
VARIABLE_GROUPS = {
    "Survey admin / IDs": "DUID",
    "Demographics": "AGE31X",
    "Priority conditions": "HIBPDX",
    "Health status": "RTHLTH31",
    "Disability days": "DDNWRK22",
    "Access to care": "ACCELI42",
    "Employment": "EMPST31",
    "Income & tax": "FILEDR22",
    "Health insurance": "TRIJA22X",
    "Utilization & expenditure": "TOTTCH22",
    "Weights & variance": "PERWT22F",
}

# Same-year healthcare utilization counts. Strongly tied to TOTEXP22 (they count the
# events whose payments make up the target), so excluded from the main classifier.
UTILIZATION_COLS = [
    "OBTOTV22", "OBDRV22", "OPTOTV22", "OPDRV22", "ERTOT22", "IPDIS22",
    "IPNGTD22", "DVTOT22", "HHTOTD22", "HHAGD22", "HHINDD22", "HHINFD22", "RXTOT22",
]

WEIGHT_DESIGN_COLS = ["PERWT22F", "FAMWT22F", "FAMWT22C", "SAQWT22F", "DIABW22F", "VARSTR", "VARPSU"]

_PAYER = r"(EXP|SLF|MCR|MCD|PRV|VA|TRI|OFD|STL|WCP|OSR|PTR|OTH|TCH)"
_EXPENDITURE_RE = re.compile(rf"^[A-Z]{{2,6}}{_PAYER}22$")


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Return the full HC-243 table (22,431 x 1,420), unmodified."""
    if use_cache and CACHE_PARQUET.exists():
        return pd.read_parquet(CACHE_PARQUET)
    df = pd.read_excel(RAW_XLSX, sheet_name="H243")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PARQUET, index=False)
    return df


def analysis_population(df: pd.DataFrame) -> pd.DataFrame:
    """Persons with a positive person-level weight (the documented analytic universe)."""
    return df.loc[df[WEIGHT_COL] > 0].copy()


def expenditure_columns(df: pd.DataFrame) -> list[str]:
    """All payment/source-of-payment/charge columns. TOTEXP22 is the sum of the
    service-level *EXP22 columns, so every one of these leaks the target."""
    return [c for c in df.columns if _EXPENDITURE_RE.match(c)]


def variable_group_map(columns: list[str]) -> dict[str, str]:
    """Map each column to its documentation variable group, based on file position."""
    starts = sorted((columns.index(first), group) for group, first in VARIABLE_GROUPS.items())
    mapping = {}
    for i, (start, group) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(columns)
        for col in columns[start:end]:
            mapping[col] = group
    return mapping


def reserved_code_share(series: pd.Series) -> dict[int, float]:
    """Share of rows holding each reserved code. Not meaningful for SIGNED_INCOME_COLS,
    where negative values can be real amounts."""
    if not pd.api.types.is_numeric_dtype(series):
        return {}
    counts = series[series.isin(list(RESERVED_CODES))].value_counts()
    return {int(code): n / len(series) for code, n in counts.items()}
