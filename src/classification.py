"""Classification models and their variants.

VARIANTS maps algorithm -> variant name -> factory. A factory takes the
positive-class weight (negatives / positives in the training data) and returns
(estimator, sampler). The sampler is an imbalanced-learn resampler or None.
Notebook 05 scores every variant; notebook 06 rebuilds the best one per
algorithm by name from reports/results/best_variants.json.
"""

from __future__ import annotations

import json

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data_loader import PROJECT_ROOT

RS = 42
BEST_VARIANTS_PATH = PROJECT_ROOT / "reports" / "results" / "best_variants.json"

# Fixed display order (also fixes each algorithm's chart color).
ALGORITHMS = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM",
              "Naive Bayes", "Gradient Boosting", "XGBoost", "Voting"]


def make_pipeline(preprocessor, model, sampler=None) -> ImbPipeline:
    """preprocess -> (optional resampling) -> model.

    imbalanced-learn's Pipeline applies the sampler during fit only, so
    resampling happens on the training folds and never on validation/test rows.
    """
    steps = [("prep", clone(preprocessor))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps)


def _lr(**kw):
    return LogisticRegression(max_iter=5000, **kw)


def _xgb(pos_weight, **kw):
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=RS)
    params.update(kw)
    return XGBClassifier(**params)


def _soft_voters(pw):
    return [("lr", _lr(class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                          class_weight="balanced_subsample", n_jobs=-1, random_state=RS)),
            ("xgb", _xgb(pw, scale_pos_weight=pw))]


VARIANTS = {
    "Logistic Regression": {
        "L2, no class weight": lambda pw: (_lr(), None),
        "L2 + balanced": lambda pw: (_lr(class_weight="balanced"), None),
        "L1 + balanced": lambda pw: (_lr(l1_ratio=1.0, solver="liblinear", class_weight="balanced"), None),
        "ElasticNet + balanced": lambda pw: (_lr(l1_ratio=0.5, solver="saga", class_weight="balanced"), None),
    },
    "KNN": {
        "k=15, uniform, euclidean": lambda pw: (KNeighborsClassifier(n_neighbors=15), None),
        "k=75, uniform, euclidean": lambda pw: (KNeighborsClassifier(n_neighbors=75), None),
        "k=75, distance-weighted": lambda pw: (KNeighborsClassifier(n_neighbors=75, weights="distance"), None),
        "k=75, manhattan": lambda pw: (KNeighborsClassifier(n_neighbors=75, p=1), None),
        "k=75 + SMOTE": lambda pw: (KNeighborsClassifier(n_neighbors=75), SMOTE(random_state=RS)),
    },
    "Decision Tree": {
        "Gini, no depth limit": lambda pw: (DecisionTreeClassifier(class_weight="balanced", random_state=RS), None),
        "Gini, max_depth=6": lambda pw: (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RS), None),
        "Entropy, max_depth=6": lambda pw: (DecisionTreeClassifier(criterion="entropy", max_depth=6,
                                                                   class_weight="balanced", random_state=RS), None),
        "Gini, min_samples_leaf=100": lambda pw: (DecisionTreeClassifier(min_samples_leaf=100, class_weight="balanced",
                                                                         random_state=RS), None),
    },
    "Random Forest": {
        "300 trees, fully grown": lambda pw: (RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                                     n_jobs=-1, random_state=RS), None),
        "300 trees, min_samples_leaf=5": lambda pw: (RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                            class_weight="balanced_subsample",
                                                                            n_jobs=-1, random_state=RS), None),
        "300 trees, leaf=5, max_features=0.3": lambda pw: (RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                                  max_features=0.3,
                                                                                  class_weight="balanced_subsample",
                                                                                  n_jobs=-1, random_state=RS), None),
        "Extra Trees, leaf=5": lambda pw: (ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                                                class_weight="balanced_subsample",
                                                                n_jobs=-1, random_state=RS), None),
    },
    "SVM": {
        "Linear kernel": lambda pw: (SVC(kernel="linear", class_weight="balanced"), None),
        "Polynomial kernel (degree 3)": lambda pw: (SVC(kernel="poly", degree=3, class_weight="balanced"), None),
        "RBF kernel": lambda pw: (SVC(kernel="rbf", class_weight="balanced"), None),
        "Sigmoid kernel": lambda pw: (SVC(kernel="sigmoid", class_weight="balanced"), None),
    },
    "Naive Bayes": {
        "Gaussian NB, learned priors": lambda pw: (GaussianNB(), None),
        "Gaussian NB, equal priors": lambda pw: (GaussianNB(priors=[0.5, 0.5]), None),
        "Bernoulli NB, equal priors": lambda pw: (BernoulliNB(binarize=0.0, class_prior=[0.5, 0.5]), None),
    },
    "Gradient Boosting": {
        "sklearn GradientBoosting (unweighted)": lambda pw: (GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                                                        random_state=RS), None),
        "HistGradientBoosting (unweighted)": lambda pw: (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                                        random_state=RS), None),
        "HistGradientBoosting + balanced": lambda pw: (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                                      class_weight="balanced",
                                                                                      random_state=RS), None),
    },
    "XGBoost": {
        "No class weighting": lambda pw: (_xgb(pw), None),
        "scale_pos_weight": lambda pw: (_xgb(pw, scale_pos_weight=pw), None),
        "scale_pos_weight, depth 3, 600 trees": lambda pw: (_xgb(pw, scale_pos_weight=pw, max_depth=3,
                                                                 n_estimators=600), None),
        "scale_pos_weight + stronger L1/L2": lambda pw: (_xgb(pw, scale_pos_weight=pw, reg_alpha=1.0,
                                                              reg_lambda=5.0), None),
    },
    "Voting": {
        "Hard voting (LR + RF + XGB)": lambda pw: (VotingClassifier(_soft_voters(pw), voting="hard"), None),
        "Soft voting (LR + RF + XGB)": lambda pw: (VotingClassifier(_soft_voters(pw), voting="soft"), None),
    },
}

BASELINES = {
    "Dummy (most frequent)": lambda pw: (DummyClassifier(strategy="most_frequent"), None),
    "Dummy (stratified random)": lambda pw: (DummyClassifier(strategy="stratified", random_state=RS), None),
}


def build(algorithm: str, variant: str, pos_weight: float):
    """Return (estimator, sampler) for one named variant."""
    table = BASELINES if algorithm == "Baseline" else VARIANTS[algorithm]
    return table[variant](pos_weight)


def save_best_variants(best: dict[str, str]) -> None:
    BEST_VARIANTS_PATH.write_text(json.dumps(best, indent=2))


def load_best_variants() -> dict[str, str]:
    return json.loads(BEST_VARIANTS_PATH.read_text())
