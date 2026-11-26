"""Turns validated specs into a leak-proof sklearn Pipeline and evaluates it.

Leak-proofing: EVERY learned transform (imputer, scaler, encoder, selector, ...) lives inside
the Pipeline, and the Pipeline is evaluated with cross_validate, so each transform is fit on
the training folds only.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, PolynomialFeatures, RobustScaler, StandardScaler)

from registry import ESTIMATORS
from utils import make_cv


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


class ClampedSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest that clamps k to the number of available features at fit time."""
    def __init__(self, k=10, problem_type="classification"):
        self.k = k
        self.problem_type = problem_type

    def fit(self, X, y=None):
        f = f_classif if self.problem_type == "classification" else f_regression
        self.selector_ = SelectKBest(f, k=min(self.k, X.shape[1])).fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def build_estimator(model_spec: dict, req, cfg):
    cls = ESTIMATORS[req.problem_type][req.algorithm]
    params = dict(model_spec["hyperparameters"])
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    if "random_state" in cls().get_params():
        params["random_state"] = cfg.seed  # seeds are set by us, never by an agent
    return cls(**params)


def build_pipeline(fs: dict, ms: dict, profile: dict, req, cfg) -> Pipeline:
    excluded = set(profile["suspects"]) | set(fs["drop_columns"])
    num = [c for c in profile["numeric_columns"] if c not in excluded]
    cat = [c for c in profile["categorical_columns"] if c not in excluded]
    if not num and not cat:
        raise ValueError("No usable features left")
    log_cols = [c for c in fs["log_transform_columns"] if c in num]
    plain = [c for c in num if c not in log_cols]

    def imp():
        return SimpleImputer(strategy=fs["numeric_impute"])

    blocks = []
    if num:
        prep = []
        if plain:
            prep.append(("plain", imp(), plain))
        if log_cols:
            prep.append(("log", Pipeline([("imp", imp()), ("log", FunctionTransformer(signed_log1p))]), log_cols))
        steps = [("prep", ColumnTransformer(prep, sparse_threshold=0))]
        if fs["polynomial_degree"] > 1:
            steps.append(("poly", PolynomialFeatures(degree=fs["polynomial_degree"],
                                                     interaction_only=fs["interaction_only"], include_bias=False)))
        if fs["numeric_scaler"] != "none":
            steps.append(("scale", _SCALERS[fs["numeric_scaler"]]()))
        blocks.append(("num", Pipeline(steps), num))
    if cat:
        enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)
               if fs["categorical_encoding"] == "onehot"
               else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        blocks.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]), cat))

    steps = [("columns", ColumnTransformer(blocks, sparse_threshold=0)),
             ("variance", VarianceThreshold(fs["variance_threshold"]))]
    if fs["select_k_best"]:
        steps.append(("kbest", ClampedSelectKBest(fs["select_k_best"], req.problem_type)))
    steps.append(("model", build_estimator(ms, req, cfg)))
    return Pipeline(steps)


def run_cv(pipe, X, y, req, cfg, shuffle_target: bool = False) -> dict:
    y_used = y.to_numpy()
    if shuffle_target:  # leakage canary: destroy the X->y relationship
        y_used = np.random.RandomState(cfg.seed + 1).permutation(y_used)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cross_validate(pipe, X, y_used, cv=make_cv(req.problem_type, cfg.cv_folds, cfg.seed),
                             scoring=req.metric, return_train_score=True, n_jobs=cfg.n_jobs,
                             error_score="raise")
    val, tr = res["test_score"], res["train_score"]
    return {"val_score": float(val.mean()), "val_std": float(val.std()),
            "train_score": float(tr.mean()), "gap": float(tr.mean() - val.mean()),
            "fit_time": round(time.time() - t0, 2)}


def canary_check(pipe, X, y, req, cfg, baseline: float):
    """Fit the SAME pipeline on shuffled targets. A leak-free pipeline must score ~ baseline."""
    score = run_cv(pipe, X, y, req, cfg, shuffle_target=True)["val_score"]
    return score, bool(score > baseline + cfg.canary_margin)
