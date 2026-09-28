# Data

## Source

**MEPS HC-243: 2022 Full Year Consolidated Data File**, Agency for Healthcare Research and Quality (AHRQ), Medical Expenditure Panel Survey.
Download: https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-243

## Layout

| Path | Contents | In git |
|---|---|---|
| `raw/h243.xlsx` | The dataset: 22,431 persons x 1,420 variables, one sheet `H243` | No (93 MB, re-download) |
| `raw/h243doc.pdf` | Official documentation (variable descriptions, reserved codes, weights) | Yes |
| `processed/h243.parquet` | Exact copy of the raw sheet in Parquet for fast loading (created by `src/data_loader.py`) | No |

The raw files are never modified. Loading the xlsx takes about 2.5 minutes; the Parquet cache loads in about 1 second.

## Key facts

- One row = one person. `DUPERSID` is the unique person ID.
- There are no NaNs. Missing or not-applicable answers use **reserved negative codes**:

| Code | Meaning |
|---|---|
| -1 | Inapplicable (question not asked because of a skip pattern) |
| -2 | Determined in a previous round |
| -7 | Refused |
| -8 | Don't know / not ascertained |
| -10 | Hourly wage top-coded |
| -13 | Initial wage imputed |
| -15 | Cannot be computed |

- Some income variables (`TTLP22X`, `FAMINC22`, `POVLEV22`, `BUSNP22X`, `SALEP22X`, `TRSTP22X`) contain genuine negative values (losses), so not every negative number is a code.
- 684 persons have `PERWT22F = 0` (no person-level weight). The analysis population is the 21,747 persons with `PERWT22F > 0`.

## Data use

Under the AHRQ data use agreement, the data may be used only for statistical reporting and analysis. Do not attempt to identify any individual. Cite AHRQ and MEPS as the data source.
