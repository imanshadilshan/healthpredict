# HealthPredict - Theory Notes

These notes explain the machine-learning ideas behind the HealthPredict notebooks in plain English, with the key formulas. Each section ends with **"In this project"**, which links the idea to what we actually did with the MEPS HC-243 data.

| Notebook | Sections that explain it |
|---|---|
| `01_dataset_understanding`, `02_eda` | 1, 2, 3, 6.1 |
| `03_data_preprocessing` | 3, 4, 6, 7, 8 |
| `04_classification_baseline` | 5, 9, 11, 12 |
| `05_classification_algorithms` | 10, 11 |
| `06_model_comparison` | 11, 12, 13 |
| `07`-`11` (coming next) | 14, 15, 16 |

## Contents

1. [The machine learning lifecycle](#1-the-machine-learning-lifecycle)
2. [Types of machine learning problems](#2-types-of-machine-learning-problems)
3. [Data leakage](#3-data-leakage)
4. [Splitting data: train, validation and test](#4-splitting-data-train-validation-and-test)
5. [Cross-validation](#5-cross-validation)
6. [Data preprocessing](#6-data-preprocessing)
7. [Feature engineering](#7-feature-engineering)
8. [Feature selection](#8-feature-selection)
9. [Imbalanced classification](#9-imbalanced-classification)
10. [Classification algorithms](#10-classification-algorithms)
11. [Overfitting, underfitting and the bias-variance trade-off](#11-overfitting-underfitting-and-the-bias-variance-trade-off)
12. [Evaluation metrics for classification](#12-evaluation-metrics-for-classification)
13. [Comparing and selecting models](#13-comparing-and-selecting-models)
14. [Hyperparameter tuning (next part)](#14-hyperparameter-tuning-next-part)
15. [Explainability (later part)](#15-explainability-later-part)
16. [Related later parts: regression, clustering and PCA](#16-related-later-parts-regression-clustering-and-pca)
17. [Glossary](#17-glossary)

---

## 1. The machine learning lifecycle

Machine learning is not "pick an algorithm and train it". It is a sequence of decisions, and each one depends on the one before:

```text
Problem -> Problem type -> Understand data (EDA) -> Define target -> Select features
   -> Split -> Preprocess (fit on train only) -> Baseline -> Compare algorithms
   -> Cross-validate -> Tune -> Final test evaluation -> Explain -> Save -> Deploy -> Monitor
```

The most common mistake is to start in the middle ("let's use Random Forest") before knowing what the target is, what the data looks like, or how success will be measured.

**In this project:** notebooks 01-02 covered problem → EDA, notebook 03 covered target → preprocessing, and notebooks 04-06 covered baseline → comparison. Tuning, final evaluation and explanation follow in 07-11.

---

## 2. Types of machine learning problems

| Type | Has a target? | Target type | Example in this project |
|---|---|---|---|
| **Regression** | Yes | A number | Predict `TOTEXP22` in dollars |
| **Binary classification** | Yes | One of two classes | High-cost (1) vs low-cost (0) |
| **Multiclass classification** | Yes | One of 3+ classes | Low / medium / high cost |
| **Clustering** | No | Groups found by the algorithm | Segments of similar people |
| **Dimensionality reduction** | No | Fewer, combined features | PCA on the health features |

The target decides the problem type, and the problem type decides which algorithms and metrics make sense.

**In this project:** the main problem is **binary classification**. `HIGH_COST = 1` if 2022 spending is at least the 80th percentile ($8,913). The EDA justified this cut-off: the top 20% of people account for about 80% of all spending.

---

## 3. Data leakage

**Leakage** happens when information that would not be available at prediction time, or that comes from the test data, slips into training. The model then looks excellent during development and fails in real use.

Two kinds matter most:

1. **Target leakage:** a feature that *contains* the answer.
   - In this project, the 279 expenditure columns add up exactly to `TOTEXP22`, so they are excluded.
   - Same-year **utilisation counts** (doctor visits, prescriptions, hospital stays) are the *events whose bills make up the target*. Using "number of doctor visits" to predict "money spent on doctor visits" is circular, so they are excluded from the main model.
2. **Train-test contamination:** any statistic (a mean, a median, a category list, a threshold, a selected feature set) computed using the test rows.
   - The fix is to **fit on training data only, then apply to test data**.

Warning sign: a score that looks too good. For this problem, a ROC-AUC above about 0.95 would point to leakage, not genius.

---

## 4. Splitting data: train, validation and test

| Part | Used for | Touched how often |
|---|---|---|
| **Training set** | Learning model parameters | Every fit |
| **Validation data** | Choosing between models and settings | Many times (via cross-validation) |
| **Test set** | One final, honest estimate of performance on new data | **Once**, at the very end |

If the test set is used to make choices, for example "model B scored higher on test, so use B", it stops being a fair measure. It has become part of training.

**Stratification** keeps the class mix (20% high-cost) the same in each part.

**Grouping** keeps related rows together. When rows are not independent (people in the same household, repeated visits from the same patient, the same customer), splitting a group across train and test lets the model "recognise" it. Group-aware splitting gives an honest estimate for *new* groups.

**In this project:** an 80/20 split with `StratifiedGroupKFold`, grouped by household (`DUID`) and stratified by spending decile. The result is 17,398 training and 4,349 test people, with zero households on both sides. A plain random split would have broken up 2,493 households. The high-cost threshold was computed *after* the split, from training data only.

---

## 5. Cross-validation

**k-fold cross-validation** cuts the training data into *k* equal parts (folds). The model is trained *k* times, each time on *k − 1* folds, and scored on the one fold left out:

```text
Fold:      1     2     3     4     5
Run 1:   [VAL] train train train train
Run 2:   train [VAL] train train train
...
Run 5:   train train train train [VAL]
```

- The **mean** of the *k* scores estimates performance; the **standard deviation** shows how much it moves by chance.
- **Stratified** k-fold keeps the class balance in every fold.
- **Group** k-fold keeps each group inside a single fold.
- **Out-of-fold predictions:** every row gets exactly one prediction, from the model that didn't see it. They are ideal for drawing ROC curves and confusion matrices without touching the test set.
- Preprocessing must happen **inside** each fold, which is why everything sits in a `Pipeline`. Fitting the scaler on all the training data before cross-validation leaks validation information into training.

**In this project:** 5 folds, stratified and grouped by household. Exactly the same folds are reused for every model (`src/preprocessing.py: cv_folds`), so comparisons are fair.

---

## 6. Data preprocessing

### 6.1 Missing values

| Kind | Meaning | Typical handling |
|---|---|---|
| **MCAR** (missing completely at random) | Missingness unrelated to anything | Any imputation works |
| **MAR** (missing at random) | Missingness explained by *other* columns | Imputation using other columns |
| **MNAR** (missing not at random) | Missingness related to the missing value itself | Add a "missing" indicator, model it explicitly |
| **Structural** | The question didn't apply, by design | Keep as its own category |

Common imputation methods:
- **Mean/median** for numbers. The median is robust to outliers.
- **Most frequent** (mode) for categories.
- **Constant / "missing" category.**
- **Model-based** imputation (KNN imputer, iterative imputer).

A **missing indicator** column (1 = was missing) lets the model learn whether missingness itself carries meaning.

**In this project:** MEPS uses reserved codes instead of blanks.
- `-1` ("inapplicable", structural): kept as its own category for nominal and yes/no features. For example, children are not asked about adult conditions.
- `-7/-8/-15` (refused / don't know / cannot compute): set to NaN, then imputed inside the pipeline with the training median or mode.
- On ordinal 1-5 scales, `-1` became NaN plus a missing-indicator column.
- Negative income values are real (business losses) and were left untouched.

### 6.2 Encoding categorical features

Models need numbers, but category codes are labels, not quantities. "Region 4" is not twice "Region 2".

| Encoding | How | When |
|---|---|---|
| **One-hot** | One 0/1 column per category | Nominal categories (no order). Works with every model |
| **Ordinal** | Map categories to 1, 2, 3, ... in their natural order | Ordered scales (poor → excellent) |
| **Target / frequency encoding** | Replace a category by a statistic | Very many categories (risk of leakage, so it must be fitted inside CV) |

The **dummy-variable trap:** with one-hot encoding, the last column can always be predicted from the others. Linear models without regularisation can suffer from this, so one column per feature is often dropped (`drop="if_binary"` or `drop="first"`).

**Rare categories:** pooling very small categories into an "infrequent" column (`min_frequency`) avoids tiny, noisy columns and handles unseen categories at prediction time.

**In this project:** `OneHotEncoder(handle_unknown="infrequent_if_exist", drop="if_binary", min_frequency=50)`. The 37 selected features become 110 encoded columns.

### 6.3 Feature scaling

| Scaler | Formula | Result |
|---|---|---|
| **Standardisation** (`StandardScaler`) | $z = (x - \mu) / \sigma$ | Mean 0, standard deviation 1 |
| **Min-max** (`MinMaxScaler`) | $x' = (x - \min) / (\max - \min)$ | Range 0 to 1 |
| **Robust** (`RobustScaler`) | $(x - \text{median}) / \text{IQR}$ | Less affected by outliers |

**Who needs scaling?**
- **Needs it:** distance-based models (KNN, SVM, clustering), gradient-based models (logistic regression, neural networks), regularised models, and PCA.
- **Doesn't need it:** tree-based models (decision trees, random forests, boosting). They only compare a feature with a threshold, so the units don't matter.

**In this project:** KNN with scaling scored much better than without, because unscaled `POVLEV22` (range −2,190 to 4,239) swamped age (0-85) and the number of conditions (0-12) in the distance calculation. The numbers are in notebook 05.

### 6.4 Transformations for skewed data

Long right tails (income, spending) give a few extreme values a huge influence on linear models.

| Transform | Formula | Notes |
|---|---|---|
| **Log** | $\log(1 + x)$ | Only for $x \ge 0$ |
| **Box-Cox** | $(x^\lambda - 1)/\lambda$ | Only for $x > 0$; $\lambda$ is learned |
| **Yeo-Johnson** | Box-Cox-like, defined for all real numbers | Handles zeros and negatives; $\lambda$ is learned |

**In this project:** `POVLEV22` (skew 2.49, with some negative values) was transformed with Yeo-Johnson, leaving skew −0.47. For the later regression task, `log1p(TOTEXP22)` will be the natural target.

### 6.5 Outliers

An outlier is not automatically an error. First ask: is the value **impossible** (data error, so fix or remove it) or **rare but real** (keep it, and perhaps soften its influence)?

- **Detection:** the IQR rule (below $Q_1 - 1.5\,\text{IQR}$ or above $Q_3 + 1.5\,\text{IQR}$), z-scores, or domain knowledge.
- **Softening:** log or power transforms, capping (winsorising), or robust models (trees).

**In this project:** 12% of people are "outliers" in `TOTEXP22` by the IQR rule. They are the genuine high spenders the project is trying to find, so they were kept.

### 6.6 Pipelines and the fit/transform rule

Every preprocessing step that **learns** something follows the same rule:

```python
step.fit(X_train)            # learn medians, means, categories... from TRAINING data only
step.transform(X_train)      # apply them to training data
step.transform(X_test)       # apply the SAME learned values to test data (never fit on test)
```

scikit-learn's **`Pipeline`** chains steps, and **`ColumnTransformer`** applies different steps to different column groups. Putting preprocessing *and* the model in one pipeline means:
- cross-validation re-fits the preprocessing inside every fold automatically, so there is no leakage;
- the same object can be saved and used in the app, so training and prediction can never drift apart.

imbalanced-learn's **`Pipeline`** adds sampler steps (SMOTE and others) that run only during `fit`, never on validation or test data.

---

## 7. Feature engineering

Feature engineering creates new inputs that make a pattern easier to learn. Good engineered features:
- have a **clear meaning** (you can explain them in one sentence);
- use only information **available at prediction time** (no leakage);
- are **documented**;
- must **earn their place**. If an engineered feature duplicates an existing one, drop it.

Common techniques:
- **Counts and aggregates:** the number of conditions.
- **Ratios:** income per family member.
- **Binning:** age groups.
- **Interactions:** age × conditions.
- **Date parts.**
- **Domain flags:** out of scope at year end.

**In this project:** five features were engineered (`N_CONDITIONS`, `N_LIMITATIONS`, `ANY_LIMITATION`, `OUT_OF_SCOPE_1231`, `AGE_BAND`).
- `N_CONDITIONS` became the single most informative feature.
- Three of the five were later dropped as redundant: `AGE_BAND` duplicates `AGELAST`, `ANY_LIMITATION` duplicates `N_LIMITATIONS`, and `OUT_OF_SCOPE_1231` duplicates `REGION22 = -1`.

---

## 8. Feature selection

The goal is to keep features that carry **useful, non-duplicated** information, which gives simpler, faster, more explainable and sometimes better models.

### Filter methods (score each feature alone, before any model)

| Method | Measures | Feature types |
|---|---|---|
| **Correlation** (Pearson / Spearman) | Linear / rank relationship. Mostly used to find **redundant pairs** | Numeric |
| **ANOVA F-test** (`f_classif`) | Do the class means differ? $F = \frac{\text{between-class variance}}{\text{within-class variance}}$ | Numeric vs class |
| **Chi-square** (`chi2`) | Do category counts differ from what independence predicts? $\chi^2 = \sum \frac{(O-E)^2}{E}$ | Categorical (non-negative) vs class |
| **Mutual information** | How much knowing the feature reduces uncertainty about the target. Catches non-linear patterns | Any |

Cautions:
- With large samples almost every p-value is tiny, so rank by **effect size**, not p-value.
- The continuous mutual-information estimator is distorted by many identical values. In notebook 03, family members share the same income value, so continuous columns were binned into deciles first.

### Wrapper methods (search over feature subsets using a model)

- **Recursive Feature Elimination (RFE):** fit a model, remove the weakest feature(s), and repeat.
- **RFECV** picks the number of features by cross-validation.
- Wrappers are accurate but expensive.

### Embedded methods (selection happens during training)

- **L1 (Lasso) regularisation** drives the weights of unhelpful features to exactly zero. With overlapping features, it tends to keep one and drop the rest.
- **Tree importance** (impurity decrease) is fast, but biased towards features with many distinct values. Permutation importance is fairer.

**In this project:**
- All three families were used, and a feature was dropped only if it was **redundant** or **weak on every method**.
- 41 candidates became 37 features, and the cross-validated ROC-AUC was unchanged (0.8228 → 0.8222 for logistic regression).
- Selection used the whole training set, so later CV scores are very slightly optimistic. The test set is untouched.

---

## 9. Imbalanced classification

With 80% negatives, a model that always predicts "negative" is 80% accurate and completely useless. Imbalance affects both **training** (the model can ignore the minority class) and **evaluation** (accuracy misleads).

| Method | Idea | Pros / cons |
|---|---|---|
| **Class weights** | Errors on the minority class cost more in the loss function, e.g. `class_weight="balanced"`: $w_c = \frac{n}{k \cdot n_c}$ | No data changes; fast. Inflates predicted probabilities |
| **Random oversampling** | Duplicate minority rows | Simple; can overfit duplicates |
| **Random undersampling** | Delete majority rows | Fast; throws information away |
| **SMOTE** | Create synthetic minority rows between neighbours: $x_{new} = x_i + \lambda (x_{nn} - x_i)$ | Smoother than duplication; unrealistic with one-hot data (use SMOTENC) |
| **Threshold moving** | Keep the model and lower the decision threshold | No retraining; needs a threshold chosen on validation data |

**Golden rule:** resampling is applied **only to the training folds**, never to validation or test data, which must keep the real class mix.

**In this project:**
- All four fixes gave the same recall (≈ 0.72) and F1 (≈ 0.54) with logistic regression, and none changed ROC-AUC (0.82).
- They all effectively move the threshold. Class weights were chosen as the simplest.
- Weighted models need **calibration** before their probabilities are shown as "risk %" (see §12).

---

## 10. Classification algorithms

Quick map:

| Algorithm | Decision boundary | Needs scaling | Handles interactions | Speed | Interpretability |
|---|---|---|---|---|---|
| Logistic Regression | Straight (linear) | Yes | Only if added by hand | Very fast | High |
| KNN | Very flexible, local | **Yes** | Yes | Slow to predict | Low |
| Decision Tree | Axis-aligned steps | No | Yes | Fast | High (if small) |
| Random Forest | Smooth-ish steps | No | Yes | Medium | Medium |
| SVM | Linear or curved (kernel) | **Yes** | With kernels | Slow on large data | Low |
| Naive Bayes | Curved (quadratic for Gaussian) | No (Gaussian) | No (independence) | Very fast | Medium |
| Gradient Boosting | Steps | No | Yes | Medium | Medium |
| XGBoost | Steps | No | Yes | Fast | Medium |
| Voting ensemble | Mix | Depends | Yes | Sum of members | Low |

### 10.1 Logistic Regression

- **Linear score:** $z = w_0 + w_1 x_1 + \dots + w_p x_p$
- **Sigmoid:** $p = \sigma(z) = \frac{1}{1 + e^{-z}}$, which squeezes any score into (0, 1).
- **Log-odds:** $\log\frac{p}{1-p} = z$. Each weight is the change in log-odds per unit of its feature; $e^{w}$ is the **odds ratio**.
- **Training** minimises **log-loss** (cross-entropy): $-\frac{1}{n}\sum [y\log p + (1-y)\log(1-p)]$.
- **Regularisation** adds a penalty that controls overfitting:
  - **L2 (Ridge):** $\lambda \sum w_j^2$ shrinks all weights.
  - **L1 (Lasso):** $\lambda \sum |w_j|$ sets some weights to exactly 0 (feature selection).
  - **ElasticNet** mixes both.
  - In scikit-learn, **C = 1/λ**, so a *smaller* C means a *stronger* penalty. Since version 1.8, L1/L2/ElasticNet is chosen with `l1_ratio` (0 = L2, 1 = L1).
- **Multiclass:** multinomial (softmax) or one-vs-rest.
- **Strengths:** fast, stable, well-calibrated (without class weights), explainable. **Weaknesses:** linear boundary, so interactions must be added by hand.

### 10.2 K-Nearest Neighbors

- To predict a point, find the **k closest training points** and take a vote (classification) or an average (regression).
- **Distances:**
  - Euclidean: $\sqrt{\sum (a_i - b_i)^2}$
  - Manhattan: $\sum |a_i - b_i|$
  - Minkowski: $(\sum |a_i - b_i|^p)^{1/p}$, which is Manhattan for p = 1 and Euclidean for p = 2.
- **k:** a small k gives a jagged boundary and overfits; a large k gives a smooth boundary and underfits.
- **Weighted KNN** (`weights="distance"`): closer neighbours count more.
- It is a **"lazy learner"**: no training at all, and all the work happens at prediction time.
- It needs **scaling**, and suffers from the **curse of dimensionality**: in many dimensions, all points become almost equally far apart.

### 10.3 Decision Tree

- It splits the data with yes/no questions on one feature at a time, choosing at each node the split that makes the children **purest**:
  - **Gini impurity:** $G = 1 - \sum_c p_c^2$, where 0 means pure and 0.5 is the worst case for two classes.
  - **Entropy:** $H = -\sum_c p_c \log_2 p_c$
  - **Information gain** = parent impurity − weighted average child impurity.
- **Root node:** the first split. **Leaves:** the final groups.
- **Overfitting control:**
  - pre-pruning: `max_depth`, `min_samples_leaf`, `min_samples_split`;
  - post-pruning: cost-complexity, `ccp_alpha`.
- **Strengths:** readable, no scaling, captures interactions. **Weaknesses:** unstable (small data changes give a different tree) and overfits easily.

### 10.4 Random Forest

- **Bagging (bootstrap aggregating):** each tree trains on a **bootstrap sample** (n rows drawn *with replacement*; about 63% of unique rows appear).
- **Random feature selection:** each split considers only a random subset of features (`max_features`, often $\sqrt{p}$). This **decorrelates** the trees.
- **Prediction** is the average of the trees' probabilities (or a majority vote).
- **Why it works:** averaging many low-bias, high-variance trees keeps the low bias and cuts the variance.
- **Out-of-bag (OOB) score:** each tree can be tested on the ~37% of rows it never saw, which gives a free validation estimate.
- **Extra Trees:** also randomises the split thresholds, adding more randomness and speed.
- **Feature importance** is the mean impurity decrease (biased towards high-cardinality features) or permutation importance.

### 10.5 Support Vector Machine

- It finds the **hyperplane** $w \cdot x + b = 0$ with the **largest margin** between the classes. The margin width is $2 / \lVert w \rVert$.
- **Support vectors** are the training points on or inside the margin; they alone define the boundary.
- **Soft margin:** **C** trades margin width against training errors (hinge loss $\max(0, 1 - y f(x))$). A large C means fewer errors allowed, a narrower margin, and a higher risk of overfitting.
- **Kernel trick:** replace dot products with a kernel $K(a, b)$, which implicitly maps the data into a higher-dimensional space:
  - linear: $a \cdot b$
  - polynomial: $(\gamma\, a \cdot b + r)^d$
  - **RBF:** $\exp(-\gamma \lVert a - b \rVert^2)$
  - sigmoid: $\tanh(\gamma\, a \cdot b + r)$
- **Practical notes:** needs scaling; training time grows roughly with $n^2$ (slow beyond ~20k rows); outputs a decision value, not a probability (`probability=True` adds Platt scaling at extra cost).

### 10.6 Naive Bayes

- **Bayes' theorem:** $P(y \mid x) = \frac{P(x \mid y)\,P(y)}{P(x)}$
- **Naive independence assumption:** $P(x \mid y) = \prod_j P(x_j \mid y)$, i.e. features are independent within each class.
- **Variants:**
  - **Gaussian NB:** each $P(x_j \mid y)$ is a normal distribution (continuous features).
  - **Multinomial NB:** word counts / frequencies (text).
  - **Bernoulli NB:** binary features.
  - **Categorical NB:** categorical features coded 0, 1, 2, ...
- **Prior** $P(y)$: learned from the class frequencies, or set by hand (equal priors act like class weighting).
- **Strengths:** extremely fast; works with little data. **Weaknesses:** the independence assumption is usually false (here, age and conditions are strongly related), which makes probabilities over-confident and costs accuracy.

### 10.7 Gradient Boosting

- **Additive model:** $F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$, where each $h_m$ is a small tree (a **weak learner**).
- **Sequential learning:** each new tree fits the **negative gradient** of the loss for the current model. For squared error that is simply the residuals; for log-loss it is $y - p$.
- **Learning rate** $\eta$ shrinks each tree's contribution. A smaller $\eta$ needs more trees but generalises better.
- **Main hyperparameters:** number of trees, learning rate, tree depth, subsample, min samples per leaf.
- **Too many trees overfit.** Early stopping on a validation set finds a good number.
- **`HistGradientBoostingClassifier`** bins each feature into at most 255 buckets, which makes it much faster. It also handles missing values natively and supports class weights (similar to LightGBM).

### 10.8 XGBoost

XGBoost is gradient boosting with engineering and regularisation improvements:
- **Regularised objective:** $\text{loss} + \sum_{trees} \left(\gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2 + \alpha \lVert w \rVert_1\right)$, which penalises the number of leaves $T$ and large leaf values $w$.
- **Second-order optimisation:** uses both the gradient and the Hessian (curvature) to choose splits and leaf values.
- **Subsampling:** rows (`subsample`) and columns (`colsample_bytree`).
- **Speed:** parallel split finding, histogram method, sparsity-aware handling of missing values.
- **Imbalance:** `scale_pos_weight` ≈ negatives / positives.
- **Key hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`.

### 10.9 Ensembles: Voting and Stacking

- **Hard voting:** majority of the predicted classes.
- **Soft voting:** average of the predicted probabilities. Usually better, and gives scores for ROC-AUC.
- **Stacking:** a "meta-model" (often logistic regression) learns how to combine the base models' out-of-fold predictions.
- **Bagging** (Random Forest) and **boosting** (Gradient Boosting) are also ensembles.
- An ensemble helps most when its members are **accurate and make different mistakes**. Combining models that already agree adds little.

---

## 11. Overfitting, underfitting and the bias-variance trade-off

| | Training score | Validation score | Cause | Fix |
|---|---|---|---|---|
| **Underfitting (high bias)** | Low | Low | Model too simple | More flexible model, better features |
| **Good fit** | Good | Close to training | | |
| **Overfitting (high variance)** | Very high | Much lower | Model memorises noise | Regularisation, simpler model, more data |

**Expected error = bias² + variance + irreducible noise.**
- Flexible models (deep trees, small k) have low bias and high variance.
- Rigid models (linear, large k) have high bias and low variance.
- The best model balances the two.
- The **irreducible noise** is the part no model can predict. For healthcare cost it is large: accidents and sudden illnesses can't be seen in survey answers.

**In this project:**
- An unlimited decision tree scored 1.000 on training data and 0.611 on validation.
- KNN with k = 5 scored 0.904 vs 0.736.
- XGBoost's validation score peaked after about 140 boosting rounds and then fell, while its training score kept rising to 0.96.
- Logistic regression's gap was only 0.006.

**Diagnostic tools:**
- **Train vs validation gap:** a large gap means overfitting.
- **Validation curve:** score vs one hyperparameter (k in KNN, depth in trees). It shows the sweet spot between under- and over-fitting.
- **Learning curve:** score vs training-set size.
  - If the validation score is still rising, more data helps.
  - If both curves are flat and close, the model has hit its limit with these features.
- **Boosting curve:** validation score vs number of trees. It peaks, then flattens or drops.

---

## 12. Evaluation metrics for classification

### The confusion matrix

| | Predicted negative | Predicted positive |
|---|---|---|
| **Actual negative** | TN (true negative) | FP (false positive, a "false alarm") |
| **Actual positive** | FN (false negative, a "miss") | TP (true positive, a "hit") |

### Metrics from the confusion matrix

| Metric | Formula | Question it answers |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | How often is the model right? (Misleading with imbalance) |
| **Precision** | $\frac{TP}{TP + FP}$ | Of those flagged, how many are truly positive? |
| **Recall** (sensitivity, TPR) | $\frac{TP}{TP + FN}$ | Of all true positives, how many did we catch? |
| **Specificity** (TNR) | $\frac{TN}{TN + FP}$ | Of all true negatives, how many did we leave alone? |
| **False Positive Rate** | $\frac{FP}{FP + TN} = 1 -$ specificity | How many negatives got a false alarm? |
| **F1 score** | $2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$ | A balance of precision and recall (harmonic mean) |
| **F-beta** | $(1+\beta^2)\frac{P \cdot R}{\beta^2 P + R}$ | β > 1 weights recall more; β < 1 weights precision more |
| **Balanced accuracy** | $\frac{\text{recall} + \text{specificity}}{2}$ | Accuracy that isn't fooled by imbalance |

### Threshold-free metrics (use predicted scores)

- **ROC curve:** TPR (y) against FPR (x) at every threshold.
- **ROC-AUC:** the area under the ROC curve. It equals the probability that a random positive is scored higher than a random negative.
  - 0.5 = random
  - 0.7-0.8 = acceptable
  - 0.8-0.9 = good
  - more than 0.9 = excellent (or suspicious)
- **Precision-Recall curve:** precision (y) against recall (x) at every threshold. More informative than ROC when positives are rare.
- **PR-AUC / Average Precision (AP):** the area under the PR curve. **No-skill = the positive rate** (0.20 here), not 0.5.
- **Log-loss:** penalises confident wrong probabilities.
- **Brier score:** $\frac{1}{n}\sum (p_i - y_i)^2$, the mean squared error of the probabilities. Lower is better.

### Thresholds and calibration

- A classifier outputs a **score**, and the **threshold** (default 0.5) turns it into a decision.
  - Lowering it raises recall and lowers precision.
  - Pick it from the costs of misses and false alarms, using validation data. **Never use the test set to choose it.**
- **Calibration:** does a predicted "30%" really mean 30%?
  - Check it with a **calibration curve** (reliability diagram).
  - Fix it with **Platt scaling** (sigmoid) or **isotonic regression** (`CalibratedClassifierCV`).
  - Class weights and resampling distort calibration but not ranking.

**In this project** (logistic regression, out-of-fold):
- TN = 10,632, FP = 3,286, FN = 961, TP = 2,519.
- Accuracy 0.756, precision 0.434, recall 0.724, F1 0.543, ROC-AUC 0.822, PR-AUC 0.566.
- The "always low cost" dummy has accuracy 0.80 and recall 0, which is why accuracy is not used to choose models.

---

## 13. Comparing and selecting models

1. **Same conditions for everyone:** same folds, same preprocessing, same metrics.
2. **Pick a primary metric that fits the problem.** Here that is ROC-AUC (ranking), with PR-AUC and F1/recall as secondary metrics. Not accuracy.
3. **Look at variability, not just the mean.** A 0.002 difference with a fold standard deviation of 0.004 is noise.
4. **Use paired comparisons.** Compare models fold by fold, since they were tested on the same rows.
   - Ordinary t-tests are over-optimistic because the CV training sets overlap.
   - The **Nadeau-Bengio corrected resampled t-test** inflates the variance by $\left(\frac{1}{k} + \frac{n_{val}}{n_{train}}\right)$ instead of $\frac{1}{k}$.
5. **Weigh other factors:** training and prediction time, interpretability, calibration, maintenance.
6. **Occam's razor:** if two models are statistically tied, prefer the simpler or more explainable one. The **one-standard-error rule** puts this into practice: treat every option within one standard deviation of the best score as tied, then pick the simplest, fastest or most useful of them.
7. **Statistical vs practical significance.** A difference can be statistically "real" (p < 0.05) and still too small to matter in practice.
8. **Shortlist, then tune.** Only the shortlist goes on to hyperparameter tuning. The test set is used once, for the final chosen model.

**In this project (notebooks 05-06):**
- **Choosing variants (notebook 05):** among the variants of each algorithm, every one within one fold standard deviation of the best ROC-AUC counted as tied. The tie was broken by F1 (rounded to 2 decimals), then by training time.
- **Comparison (notebook 06), 5-fold CV ROC-AUC:**

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Soft voting (LR + RF + XGB) | 0.826 | 0.568 | 0.552 |
| XGBoost | 0.824 | 0.565 | 0.546 |
| Logistic regression | 0.822 | 0.566 | 0.543 |
| HistGradientBoosting / linear SVM / random forest | 0.821 | 0.553-0.562 | 0.542-0.550 |
| KNN | 0.811 | 0.539 | 0.359 |
| Naive Bayes | 0.800 | 0.531 | 0.503 |
| Decision tree | 0.792 | 0.487 | 0.509 |

- XGBoost was *not distinguishable* from voting (corrected p = 0.16). Logistic regression was statistically but not practically worse (−0.004, p = 0.04).
- **Shortlist:** Voting, XGBoost and Logistic Regression.

---

## 14. Hyperparameter tuning (next part)

- **Parameters** are learned from data (logistic-regression weights, tree splits). **Hyperparameters** are set before training (C, k, max_depth, learning rate).
- **Grid search** (`GridSearchCV`) tries every combination in a grid. It is thorough but expensive, and grows exponentially with the number of hyperparameters.
- **Random search** (`RandomizedSearchCV`) samples combinations at random. It is usually as good with far fewer fits, because only a few hyperparameters really matter.
- **Bayesian optimisation** (e.g. Optuna) uses past results to choose the next combination to try.
- **Nested cross-validation:** an inner CV loop tunes, and an outer CV loop estimates performance. It avoids the optimism of reporting the best tuned CV score.
- **Early stopping** (boosting) is a cheap way to tune the number of trees.
- **Always tune on training data with CV.** The test set is opened once, afterwards.

---

## 15. Explainability (later part)

- **Global explanations** answer "which features matter overall?":
  - model coefficients (linear models);
  - tree impurity importance (fast but biased);
  - **permutation importance:** how much the score drops when one feature's values are shuffled;
  - mean |SHAP|.
- **Local explanations** answer "why this prediction for this person?":
  - **SHAP** (SHapley Additive exPlanations) splits one prediction into additive contributions from each feature, based on game theory (Shapley values). The contributions sum to the prediction minus the average prediction.
- **Important:** explanations describe **what the model learned**, not cause and effect. "Uninsured people are rarely high-cost" reflects less access to care, not that being uninsured is healthy. HealthPredict is not a diagnostic tool.

---

## 16. Related later parts: regression, clustering and PCA

### Regression (predicting `TOTEXP22` in dollars)

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert y - \hat{y} \rvert$ | Average size of the error, in dollars |
| **MSE** | $\frac{1}{n}\sum (y - \hat{y})^2$ | Punishes big errors heavily |
| **RMSE** | $\sqrt{MSE}$ | Same units as the target |
| **R²** | $1 - \frac{\sum (y-\hat{y})^2}{\sum (y-\bar{y})^2}$ | Share of variance explained (1 = perfect, 0 = no better than the mean) |

- Models: Linear Regression, Ridge (L2), Lasso (L1), Polynomial Regression (adds $x^2$ and $x_i x_j$ terms), and tree / forest / boosting regressors.
- Spending is very skewed, so predicting $\log(1 + \text{TOTEXP22})$ and converting back is usually much better.

### Clustering (patient segmentation, no target)

- **K-Means:** places k centres and assigns each point to the nearest one, repeating until stable. It minimises **inertia** (within-cluster sum of squares).
- **Elbow method:** plot inertia against k and look for the bend.
- **Silhouette score:** $s = \frac{b - a}{\max(a, b)}$, where *a* is the mean distance to the point's own cluster and *b* the mean distance to the nearest other cluster. It ranges from −1 to 1, and higher is better.
- **Hierarchical clustering:** builds a tree of merges (dendrogram), using linkage rules such as Ward, complete or average.
- **DBSCAN:** density-based. It finds clusters of any shape and marks outliers as noise, using `eps` (radius) and `min_samples`.
- Clusters must be **interpreted with the original features**. Don't give them medical meanings the data doesn't support.

### PCA (dimensionality reduction)

- PCA finds new axes (**principal components**) that capture the most variance, each uncorrelated with the others.
- **Explained variance ratio:** the share of total variance each component keeps. Choose the number of components from the **cumulative** explained variance (e.g. 90%).
- **Requires scaling first**; otherwise the largest-unit feature dominates.
- Uses: visualisation (2-D plots), noise reduction, speeding up models, and supporting clustering. **t-SNE** and **UMAP** are non-linear alternatives for visualisation only.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **Baseline** | The simplest reference model a real model must beat |
| **Bootstrap sample** | A random sample of rows drawn *with replacement* |
| **Class weight** | A multiplier making errors on one class count more |
| **Cross-validation** | Repeated train/validate splits of the training data |
| **Data leakage** | Information from the target or test set sneaking into training |
| **Decision threshold** | The score above which a case is predicted positive |
| **Estimator** | scikit-learn's name for any model or transformer with `fit` |
| **Feature** | An input column |
| **Fold** | One of the k parts in k-fold cross-validation |
| **Hyperparameter** | A setting chosen before training (C, k, depth, ...) |
| **Imputation** | Filling in missing values |
| **One-hot encoding** | One 0/1 column per category |
| **Out-of-fold prediction** | A prediction for a row made by the model that did not train on it |
| **Overfitting** | Memorising training data, which gives poor performance on new data |
| **Pipeline** | A chain of preprocessing steps plus a model, fitted as one object |
| **Regularisation** | A penalty that keeps a model simple to reduce overfitting |
| **Stratification** | Keeping class proportions equal across splits |
| **Target (label)** | What the model predicts (`HIGH_COST`) |
| **Test set** | Held-out data used once, for the final evaluation |
| **Validation set** | Data used to compare models and tune settings |

---

*Data source: Agency for Healthcare Research and Quality (AHRQ), Medical Expenditure Panel Survey, HC-243 2022 Full Year Consolidated Data File. Model explanations describe model behaviour and do not prove medical causation; HealthPredict is an educational project, not a diagnostic system.*
