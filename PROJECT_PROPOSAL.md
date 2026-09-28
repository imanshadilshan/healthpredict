# HealthPredict ML Project

## Project Status

**Current stage:** Dataset downloaded. VS Code setup and EDA are next.

**Primary dataset:** MEPS HC-243, 2022 Full Year Consolidated Data File

**Main objective:** Build a complete traditional Machine Learning project using real healthcare data, covering the full ML lifecycle from problem definition and EDA to model training, evaluation, tuning, explainability, and deployment.

---

# 1. Project Overview

## Project Title

**HealthPredict: An Explainable Traditional Machine Learning Platform for Healthcare Risk Assessment, Cost Prediction, and Patient Segmentation**

## Purpose

This is an educational and portfolio project designed to learn and demonstrate traditional Machine Learning concepts using a real healthcare dataset.

The project must not jump directly into model training.

The first principle is:

> Understand the business or analytical problem first, determine the type of Machine Learning problem, understand the data through EDA, and only then select suitable algorithms.

---

# 2. Dataset

## Selected Dataset

**MEPS HC-243: 2022 Full Year Consolidated Data File**

The dataset is person-level healthcare data containing information related to:

- Demographics
- Health conditions
- Health status
- Healthcare utilization
- Insurance
- Employment
- Income
- Healthcare expenditure

Known dataset information:

- Year: 2022
- Records: approximately 22,431 people
- Variables: approximately 1,420
- Type: structured/tabular healthcare data
- Source: U.S. Agency for Healthcare Research and Quality, Medical Expenditure Panel Survey

## Dataset Files

The downloaded ZIP may contain multiple files.

Identify:

1. Actual HC-243 data file
2. HC-243 codebook/documentation
3. Supporting files required to understand variable codes

The codebook is documentation, not the ML dataset.

The actual data file will be loaded with Pandas.

Do not modify the raw dataset directly.

---

# 3. Project Structure

```text
HealthPredict/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_classification_baseline.ipynb
│   ├── 05_classification_algorithms.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_clustering.ipynb
│   ├── 09_regression.ipynb
│   ├── 10_pca.ipynb
│   └── 11_explainability.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── visualization.py
│
├── models/
├── reports/
│   ├── figures/
│   └── results/
│
├── app/
├── requirements.txt
├── README.md
└── PROJECT_PROPOSAL.md
```

---

# 4. Core Learning Goals

The project should cover:

## Data Understanding

- Dataset structure
- Data types
- Missing values
- Duplicate detection
- Outliers
- Distributions
- Correlations
- Categorical analysis

## Data Preprocessing

- Missing value handling
- MEPS special codes
- Categorical encoding
- Numerical scaling
- Outlier handling
- Data leakage prevention

## Feature Engineering

- Creating meaningful features
- Transformations
- Feature selection
- Statistical selection
- RFE
- Embedded feature selection

## Classification

- Logistic Regression
- K Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- Naive Bayes
- Gradient Boosting
- XGBoost
- Voting Classifier

## Regression

- Linear Regression
- Ridge
- Lasso
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Unsupervised Learning

- K Means
- Hierarchical Clustering
- DBSCAN
- Elbow Method
- Silhouette Score

## Dimensionality Reduction

- PCA
- Optional t-SNE for visualization

## Model Evaluation

Classification:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- ROC AUC
- Cross Validation

Regression:

- MAE
- MSE
- RMSE
- R2

Clustering:

- Silhouette Score
- Inertia
- Cluster visualization

## Optimization

- Cross Validation
- Grid Search
- Randomized Search
- Hyperparameter tuning

## Explainability

- Feature importance
- Permutation importance
- SHAP
- Individual prediction explanations

---

# 5. Critical Project Principle

Before selecting an algorithm, always answer:

> What type of Machine Learning problem are we solving?

Do not start with:

> Let's use Random Forest.

Start with:

```text
What is the problem?
        ↓
What is the target?
        ↓
What type of target is it?
        ↓
What type of ML problem is this?
        ↓
What algorithms are appropriate?
```

---

# 6. Possible ML Problems in HC-243

## Problem A: Healthcare Expenditure Prediction

Target:

`TOTEXP22`

If kept as a continuous numerical value:

```text
Input Features
      ↓
Healthcare Expenditure
```

This is **REGRESSION**.

---

## Problem B: High vs Low Healthcare Expenditure

Create a classification target from total expenditure:

```text
TOTEXP22
    ↓
Statistically justified threshold
    ↓
Low Expenditure
High Expenditure
```

This becomes **BINARY CLASSIFICATION**.

Important:

Do not arbitrarily choose the threshold before EDA. The threshold must be justified using the target distribution and project objective.

---

## Problem C: Patient or Individual Segmentation

No target is required.

```text
Healthcare Features
        ↓
Clustering
        ↓
Groups of similar individuals
```

This is **UNSUPERVISED LEARNING**.

---

## Problem D: Dimensionality Reduction

The dataset has many variables.

PCA can be used to:

- Reduce dimensionality
- Preserve important variance
- Visualize high dimensional data
- Support clustering

This is **DIMENSIONALITY REDUCTION**.

---

# 7. Recommended Main ML Problem

For the first supervised learning stage:

## Binary Classification

Proposed question:

> Can we classify individuals into lower and higher healthcare expenditure groups using demographic, health, insurance, utilization, and socioeconomic features?

Potential target:

`TOTEXP22`

However, the exact classification target must be finalized after EDA.

---

# 8. Avoid Data Leakage

If the target is derived from `TOTEXP22`, do not use variables that directly contain the same expenditure information as predictors.

For example, if total expenditure is calculated from component expenditure variables, those component variables should generally not be used as predictors for the expenditure classification target.

The feature selection stage must explicitly check for leakage.

---

# 9. Training Lifecycle

```text
1. Define Problem
        ↓
2. Identify ML Problem Type
        ↓
3. Understand Dataset
        ↓
4. EDA
        ↓
5. Define Target
        ↓
6. Select Candidate Features
        ↓
7. Split Data
        ↓
8. Preprocess Training Data
        ↓
9. Transform Validation/Test Data
        ↓
10. Train Baseline Model
        ↓
11. Evaluate
        ↓
12. Compare Algorithms
        ↓
13. Cross Validation
        ↓
14. Hyperparameter Tuning
        ↓
15. Final Evaluation
        ↓
16. Explain Model
        ↓
17. Save Model
        ↓
18. Deploy
        ↓
19. Monitor
```

---

# 10. Train, Validation, and Test Data

Initial approach:

```text
Raw Dataset
    ↓
Train Set
    ↓
Cross Validation

Test Set
    ↓
Final evaluation only
```

An 80/20 train/test split can be used initially.

For classification, use stratification where appropriate.

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Do not repeatedly use the test set for model selection.

---

# 11. EDA Plan

EDA must happen before final feature selection.

## Dataset Overview

Inspect:

```python
df.head()
df.shape
df.info()
df.describe()
```

Then identify:

- Numerical variables
- Categorical variables
- Identifier variables
- Potential targets

## Missing Values

Calculate:

- Missing count
- Missing percentage

MEPS may contain special codes such as `-1`, `-7`, and `-8`. Their meanings must be verified from the HC-243 codebook before preprocessing.

Do not automatically treat every negative value as missing.

## Numerical Analysis

Investigate:

- Mean
- Median
- Standard deviation
- Minimum
- Maximum
- Quartiles
- Skewness
- Outliers
- Correlations

## Categorical Analysis

Investigate:

- Number of unique values
- Frequency distribution
- Rare categories
- Dominant categories

## Target Analysis

If `TOTEXP22` is selected:

- Distribution
- Mean
- Median
- Standard deviation
- Minimum
- Maximum
- Quartiles
- Skewness
- Zero expenditure percentage
- Potential classification threshold

---

# 12. EDA Visualizations

At minimum:

1. Missing value percentage chart
2. Target distribution
3. Target boxplot
4. Important numerical feature distributions
5. Categorical feature count plots
6. Correlation heatmap
7. Important feature versus target plots
8. Outlier visualizations

EDA should lead to decisions, not just produce many charts.

---

# 13. Classification Algorithms

## Logistic Regression

Learn:

- Sigmoid function
- Probability
- Decision boundary
- Binary classification
- Multiclass extension
- L1 regularization
- L2 regularization

Variants:

```text
Logistic Regression
├── Binary
├── Multinomial
├── L1
└── L2
```

## K Nearest Neighbors

Learn:

- Distance
- K value
- Decision boundaries
- Feature scaling
- Weighted KNN

Distances:

- Euclidean
- Manhattan
- Minkowski

## Decision Tree

Learn:

- Root node
- Splitting
- Gini impurity
- Entropy
- Information Gain
- Max depth
- Overfitting

## Random Forest

Learn:

- Bagging
- Bootstrap samples
- Random feature selection
- Multiple decision trees
- Voting
- Feature importance

## Support Vector Machine

Learn:

- Hyperplane
- Margin
- Support vectors
- Kernel trick

Variants:

- Linear
- Polynomial
- RBF
- Sigmoid

## Naive Bayes

Learn:

- Bayes theorem
- Conditional probability
- Independence assumption

Variants:

- Gaussian NB
- Multinomial NB
- Bernoulli NB

For this numerical healthcare dataset, Gaussian Naive Bayes is the most relevant initial variant.

## Gradient Boosting

Learn:

- Weak learners
- Sequential learning
- Boosting
- Learning rate
- Number of estimators
- Tree depth

## XGBoost

Learn:

- Gradient boosting
- Regularization
- Tree boosting
- Learning rate
- Number of estimators
- Maximum depth

Use XGBoost after understanding simpler algorithms.

---

# 14. Classification Model Comparison

Create a table:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | |
| KNN | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| SVM | | | | | |
| Naive Bayes | | | | | |
| Gradient Boosting | | | | | |
| XGBoost | | | | | |

Do not select a final model using accuracy alone.

For imbalanced data, pay attention to:

- Precision
- Recall
- F1
- ROC AUC
- Confusion Matrix

---

# 15. Imbalanced Classification

First measure class distribution.

If classes are strongly imbalanced, investigate:

- Class weights
- Random oversampling
- Random undersampling
- SMOTE

Do not apply SMOTE automatically.

SMOTE must be applied only to training data.

```text
Train Data
    ↓
Preprocessing
    ↓
SMOTE
    ↓
Model Training

Test Data
    ↓
Preprocessing only
    ↓
Final Evaluation
```

---

# 16. Feature Selection

Because HC-243 contains approximately 1,420 variables, feature selection is a major part of the project.

## Filter Methods

- Correlation
- Chi Square
- ANOVA
- Mutual Information

## Wrapper Methods

- RFE

## Embedded Methods

- L1 Logistic Regression
- Decision Tree importance
- Random Forest importance
- Gradient Boosting importance

The final feature set must be justified.

---

# 17. Feature Engineering

Possible areas:

- Age groups
- Healthcare utilization indicators
- Insurance categories
- Employment categories
- Income transformations
- Healthcare utilization scores

Only create features with a clear interpretation.

Document every engineered feature.

---

# 18. Regression Stage

Target:

`TOTEXP22`

Problem:

```text
Predict continuous healthcare expenditure
```

Models:

- Linear Regression
- Ridge
- Lasso
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Evaluate using:

- MAE
- MSE
- RMSE
- R2

---

# 19. Clustering Stage

Use relevant healthcare and demographic features.

Algorithms:

- K Means
- Hierarchical Clustering
- DBSCAN

Evaluate using:

- Elbow Method
- Silhouette Score

Interpret clusters using original features.

Do not assign medical meanings to clusters unless supported by the data.

---

# 20. PCA Stage

Use PCA after appropriate preprocessing and scaling.

Objectives:

- Reduce dimensionality
- Analyze explained variance
- Visualize high dimensional data
- Support clustering visualization

Report:

- Explained variance ratio
- Cumulative explained variance
- Number of components

---

# 21. Explainability

## Global Explanation

Which features are important overall?

Methods:

- Tree feature importance
- Permutation importance
- SHAP

## Local Explanation

Why did the model classify one individual into a particular class?

Use SHAP or another suitable explanation method.

Important:

Model explanations describe model behavior. They do not prove medical causation.

---

# 22. Technology Stack

## Language

Python

## Data

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn
- Plotly

## Machine Learning

- Scikit learn
- XGBoost
- Imbalanced learn

## Explainability

- SHAP

## Development

- VS Code
- Jupyter Notebook

## Version Control

- Git
- GitHub

## Future Application

- FastAPI
- React

## Deployment

- Docker

---

# 23. Environment Setup

Recommended:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter scipy xgboost imbalanced-learn shap
```

Save dependencies:

```bash
pip freeze > requirements.txt
```

---

# 24. Coding Rules

1. Never modify the raw dataset.
2. Keep raw files inside `data/raw/`.
3. Save processed data inside `data/processed/`.
4. Use fixed random seeds where reproducibility matters.
5. Document preprocessing decisions.
6. Never fit transformations on the test set.
7. Never apply SMOTE to the test set.
8. Check target leakage before training.
9. Do not use test data repeatedly for model selection.
10. Do not claim causation from feature importance.
11. Do not call this a medical diagnostic system.
12. Keep notebooks focused on experiments and analysis.
13. Move reusable functions into `src/`.
14. Store model results in `reports/results/`.
15. Store plots in `reports/figures/`.

---

# 25. Development Roadmap

## Phase 1: Environment Setup

- Create VS Code project
- Create virtual environment
- Install dependencies
- Create folder structure
- Add `.gitignore`
- Put HC-243 files into `data/raw/`

**Status: NEXT**

## Phase 2: Dataset Understanding

Create:

`notebooks/01_dataset_understanding.ipynb`

Tasks:

- Extract ZIP
- Identify actual data file
- Load data
- Inspect shape
- Inspect columns
- Inspect data types
- Inspect missing values
- Read codebook
- Identify identifiers
- Identify potential targets

## Phase 3: EDA

Create:

`notebooks/02_eda.ipynb`

Tasks:

- Missing value analysis
- Special code analysis
- Numerical analysis
- Categorical analysis
- Target analysis
- Correlation analysis
- Outlier analysis
- Visualizations

## Phase 4: Define Classification Problem

Do not finalize the classification target before EDA.

After EDA:

1. Analyze `TOTEXP22`
2. Decide whether classification is appropriate
3. Define a defensible target
4. Check class balance
5. Identify leakage variables
6. Select candidate features

## Phase 5: Preprocessing

Create:

`notebooks/03_data_preprocessing.ipynb`

Tasks:

- Missing values
- Special MEPS codes
- Encoding
- Scaling
- Feature selection
- Train/test split
- Pipeline creation

## Phase 6: Classification

Start in this order:

```text
Logistic Regression
        ↓
KNN
        ↓
Decision Tree
        ↓
Random Forest
        ↓
SVM
        ↓
Naive Bayes
        ↓
Gradient Boosting
        ↓
XGBoost
```

## Phase 7: Model Comparison

Create:

`notebooks/06_model_comparison.ipynb`

Compare:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- Confusion matrices

## Phase 8: Hyperparameter Tuning

Create:

`notebooks/07_hyperparameter_tuning.ipynb`

Use:

- GridSearchCV
- RandomizedSearchCV
- Cross validation

## Phase 9: Clustering

Create:

`notebooks/08_clustering.ipynb`

Implement:

- K Means
- Hierarchical Clustering
- DBSCAN

## Phase 10: Regression

Create:

`notebooks/09_regression.ipynb`

Predict:

`TOTEXP22`

## Phase 11: PCA

Create:

`notebooks/10_pca.ipynb`

## Phase 12: Explainability

Create:

`notebooks/11_explainability.ipynb`

## Phase 13: Application

Future architecture:

```text
React
   ↓
FastAPI
   ↓
Saved ML Models
```

Possible application features:

- Prediction
- Model comparison
- Feature importance
- Patient segmentation
- Dataset insights

---

# 26. First Task in VS Code

Do not train models yet.

Start with:

```text
Extract ZIP
    ↓
Identify actual data file
    ↓
Place it in data/raw/
    ↓
Create virtual environment
    ↓
Install dependencies
    ↓
Create 01_dataset_understanding.ipynb
    ↓
Load dataset
    ↓
Inspect actual structure
    ↓
Start EDA
```

Initial notebook code:

```python
import pandas as pd
import numpy as np

DATA_PATH = "../data/raw/"

# Load the actual HC-243 file after identifying its filename
# df = pd.read_excel(DATA_PATH + "actual_filename.xlsx")

print(df.shape)
display(df.head())
df.info()
```

Then investigate:

```python
df.shape
df.columns.tolist()
df.dtypes
df.describe(include="all")
df.isnull().sum()
df.nunique()
```

Do not invent column names before inspecting the actual file.

---

# 27. Git Workflow

Use meaningful commits:

```text
chore: initialize ML project structure
chore: add HC-243 raw dataset files
feat: load and inspect HC-243 dataset
feat: complete initial EDA
feat: add preprocessing pipeline
feat: implement baseline classification models
feat: compare classification models
feat: add hyperparameter tuning
feat: implement clustering
feat: implement regression
feat: add PCA analysis
feat: add model explainability
feat: add prediction API
feat: add React dashboard
```

Do not commit:

- `.venv/`
- secrets
- API keys
- temporary files
- unnecessary generated files

---

# 28. Mentor Session Preparation

## What type of ML problem are we solving?

Answer:

> We are initially investigating a supervised binary classification problem based on healthcare expenditure. The exact target definition will be finalized after EDA. The same dataset can also support regression and unsupervised clustering.

## Why classification?

Answer:

> We want to investigate whether healthcare related demographic, health, insurance, utilization, and socioeconomic features can be used to classify individuals into defined healthcare expenditure categories.

## Why not directly train a model?

Answer:

> Because the first step in Machine Learning is understanding the problem and data. EDA is required to identify the target, data quality issues, feature distributions, class balance, relationships, and possible data leakage before selecting algorithms.

## What algorithms will you use?

Answer:

> Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting, and XGBoost. They will be compared using appropriate classification metrics.

## What is the main outcome?

Answer:

> An explainable healthcare ML platform that demonstrates the complete traditional ML lifecycle and can classify healthcare expenditure categories, predict healthcare expenditure, and identify groups of individuals with similar healthcare characteristics.

---

# 29. Future AI Assistant Instructions

When continuing this project with an AI coding assistant in VS Code:

1. Read this `PROJECT_PROPOSAL.md` before making project changes.
2. Do not skip problem definition.
3. Do not invent dataset columns.
4. Inspect the actual HC-243 dataset before writing feature-specific code.
5. Use the official HC-243 codebook to interpret variables.
6. Do not assume every negative value means missing data.
7. Do not select the classification threshold before analyzing `TOTEXP22`.
8. Check target leakage before feature selection.
9. Keep the raw dataset unchanged.
10. Prefer reproducible pipelines.
11. Explain why each ML technique is being used.
12. Compare algorithms rather than blindly selecting one.
13. Record experiment results.
14. Do not claim clinical validity.
15. If the actual dataset structure differs from this proposal, update the implementation based on the real dataset while preserving the learning objectives.

---

# 30. Current Status

## Completed

- Project idea defined
- Healthcare domain selected
- MEPS HC-243 selected
- HC-243 ZIP downloaded
- Overall ML roadmap defined

## Current Task

**Set up the VS Code project and inspect the downloaded HC-243 files.**

## Next Action

```text
Extract ZIP
    ↓
Identify actual data file
    ↓
Place it in data/raw/
    ↓
Create virtual environment
    ↓
Install dependencies
    ↓
Create 01_dataset_understanding.ipynb
    ↓
Load dataset
    ↓
Inspect actual structure
    ↓
Start EDA
```

**Do not train classification models until initial EDA and target definition are completed.**

---

# 31. Project Success Criteria

The completed project should demonstrate:

```text
Problem Definition
        +
EDA
        +
Data Preprocessing
        +
Feature Engineering
        +
Classification
        +
Regression
        +
Clustering
        +
PCA
        +
Model Evaluation
        +
Hyperparameter Tuning
        +
Explainability
        +
Deployment
```

The primary goal is not simply to achieve the highest accuracy.

The primary goal is to demonstrate a **correct, explainable, reproducible, end to end traditional Machine Learning workflow** using the HC-243 healthcare dataset.
