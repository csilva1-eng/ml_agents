"""Data handling. The test set lives inside DataVault and can only be scored ONCE, by final_test().

Nothing in the agent loop ever receives X_test / y_test: agents see a profile of the
development split only, and the graph state never contains dataframes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils import make_cv, to_native


def _clean_features(X: pd.DataFrame):
    """Pure dtype normalisation (no statistics are learned here, so no leakage possible)."""
    out, unsupported = {}, []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("float64")
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype("float64").replace([np.inf, -np.inf], np.nan)
        elif pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            unsupported.append(col)
        else:
            out[col] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return pd.DataFrame(out, index=X.index), unsupported


def scan_suspect_columns(X, y, problem_type, threshold, seed, max_rows=5000) -> dict:
    """Heuristic leak scan on DEV data only: a column that alone (depth-4 tree, CV) nearly
    perfectly predicts the target, or an identifier-like text column, is excluded."""
    if len(X) > max_rows:
        idx = X.sample(max_rows, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    cls = problem_type == "classification"
    scoring = "accuracy" if cls else "r2"
    baseline = y.value_counts(normalize=True).iloc[0] if cls else 0.0
    suspects = {}
    for col in X.columns:
        s = X[col]
        if s.dtype == object:
            if s.nunique() > 0.95 * len(s) and len(s) > 20:
                suspects[col] = "identifier-like (almost every value unique)"
                continue
            v = pd.factorize(s)[0].astype(float)
        else:
            med = s.median()
            v = s.fillna(0.0 if pd.isna(med) else med).to_numpy(dtype=float)
        tree = (DecisionTreeClassifier if cls else DecisionTreeRegressor)(max_depth=4, random_state=seed)
        try:
            score = cross_val_score(tree, v.reshape(-1, 1), y, cv=3, scoring=scoring).mean()
        except Exception:
            continue
        if score >= threshold and score > baseline + 0.02:
            suspects[col] = f"single-feature {scoring}={score:.3f} (looks target-derived)"
    return suspects


class DataVault:
    def __init__(self, df: pd.DataFrame, req, cfg):
        if req.target not in df.columns:
            raise ValueError(f"Target column '{req.target}' not found in data")
        df = df.dropna(subset=[req.target]).reset_index(drop=True)
        y = df[req.target]
        cls = req.problem_type == "classification"
        if cls and req.metric == "roc_auc" and y.nunique() != 2:
            raise ValueError("roc_auc needs a binary target; pick another metric")
        X, unsupported = _clean_features(df.drop(columns=[req.target]))
        stratify = y if cls and y.value_counts().min() >= 2 else None
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify)
        self.X_dev, self.y_dev = X_dev.reset_index(drop=True), y_dev.reset_index(drop=True)
        self.__X_test, self.__y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
        self.__used = False
        self._req, self._cfg = req, cfg
        self._profile = self._build_profile(unsupported)

    # -- profile (dev data only) -------------------------------------------------
    def _build_profile(self, unsupported):
        req, cfg, X, y = self._req, self._cfg, self.X_dev, self.y_dev
        suspects = scan_suspect_columns(X, y, req.problem_type, cfg.leak_threshold, cfg.seed)
        usable = [c for c in X.columns if c not in suspects]
        numeric = [c for c in usable if X[c].dtype != object]
        categorical = [c for c in usable if X[c].dtype == object]
        stats = {c: {"kind": "numeric" if c in numeric else "categorical",
                     "missing_frac": round(float(X[c].isna().mean()), 4),
                     "nunique": int(X[c].nunique())} for c in usable}
        cls = req.problem_type == "classification"
        if cls:
            target = {"classes": {str(k): int(v) for k, v in y.value_counts().items()}}
        else:
            target = {"mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()), "max": float(y.max())}
        dummy = (DummyClassifier(strategy="prior") if cls else DummyRegressor())
        baseline = cross_val_score(dummy, np.zeros((len(y), 1)), y, scoring=req.metric,
                                   cv=make_cv(req.problem_type, cfg.cv_folds, cfg.seed)).mean()
        return to_native({
            "target": req.target, "n_dev_rows": len(X), "columns": usable,
            "numeric_columns": numeric, "categorical_columns": categorical,
            "column_stats": stats, "suspects": suspects, "unsupported": unsupported,
            "target_summary": target, "baseline_score": float(baseline),
        })

    def profile(self) -> dict:
        return self._profile

    # -- the ONE place the test set is touched ---------------------------------
    def final_test(self, unfitted_pipeline, metric: str):
        if self.__used:
            raise RuntimeError("The locked test set has already been used. It can only be scored once.")
        self.__used = True
        unfitted_pipeline.fit(self.X_dev, self.y_dev)
        score = get_scorer(metric)(unfitted_pipeline, self.__X_test, self.__y_test)
        return float(score), unfitted_pipeline
