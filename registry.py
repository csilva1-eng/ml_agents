"""Algorithm registry + the hyperparameter WHITELIST.

The model agent can only touch what is listed in PARAM_SPECS, within the ranges given.
Anything else is rejected deterministically by guardrails.py. To support a new algorithm,
add it to ESTIMATORS and give it an entry in PARAM_SPECS.
"""
from __future__ import annotations

from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

ESTIMATORS = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "mlp": MLPClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    },
    "regression": {
        "ridge": Ridge,
        "knn": KNeighborsRegressor,
        "random_forest": RandomForestRegressor,
        "svm": SVR,
        "mlp": MLPRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    },
}

ALGORITHM_ALIASES = {
    "nn": "mlp", "neural_network": "mlp", "neuralnet": "mlp", "ann": "mlp",
    "logreg": "logistic_regression", "logistic": "logistic_regression",
    "k_nn": "knn", "kneighbors": "knn",
    "svc": "svm", "svr": "svm",
    "rf": "random_forest", "forest": "random_forest",
    "gbm": "gradient_boosting", "gbt": "gradient_boosting",
    "linear_regression": "ridge", "linear": "ridge",
}


def P(type_, only=None, **kw):
    d = {"type": type_, **kw}
    if only:
        d["only"] = only  # parameter only valid for this problem type
    return d


_CW = P("choice", only="classification", choices=[None, "balanced"])

PARAM_SPECS = {
    "logistic_regression": {
        "C": P("float", min=1e-4, max=1e3),
        "max_iter": P("int", min=100, max=5000),
        "class_weight": _CW,
    },
    "ridge": {"alpha": P("float", min=1e-4, max=1e3)},
    "knn": {
        "n_neighbors": P("int", min=1, max=100),
        "weights": P("choice", choices=["uniform", "distance"]),
        "p": P("choice", choices=[1, 2]),
    },
    "random_forest": {
        "n_estimators": P("int", min=10, max=500),
        "max_depth": P("int", min=2, max=50, nullable=True),
        "min_samples_split": P("int", min=2, max=50),
        "min_samples_leaf": P("int", min=1, max=50),
        "max_features": P("choice", choices=["sqrt", "log2", None]),
        "class_weight": _CW,
    },
    "svm": {
        "C": P("float", min=1e-3, max=1e3),
        "kernel": P("choice", choices=["linear", "rbf", "poly"]),
        "gamma": P("choice", choices=["scale", "auto"]),
        "degree": P("int", min=2, max=4),
        "class_weight": _CW,
    },
    "mlp": {
        "hidden_layer_sizes": P("int_list", min=4, max=256, min_len=1, max_len=4),
        "activation": P("choice", choices=["relu", "tanh", "logistic"]),
        "alpha": P("float", min=1e-7, max=1.0),
        "learning_rate_init": P("float", min=1e-4, max=0.1),
        "max_iter": P("int", min=50, max=1000),
        "early_stopping": P("bool"),
    },
    "gradient_boosting": {
        "n_estimators": P("int", min=20, max=500),
        "learning_rate": P("float", min=0.005, max=1.0),
        "max_depth": P("int", min=1, max=8),
        "subsample": P("float", min=0.3, max=1.0),
        "min_samples_leaf": P("int", min=1, max=50),
    },
}


def normalize_algorithm(name: str) -> str:
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    return ALGORITHM_ALIASES.get(n, n)
