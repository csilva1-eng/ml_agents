"""Small shared helpers."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def to_native(o):
    """Convert numpy scalars etc. to plain Python so graph state stays serialisable."""
    if isinstance(o, dict):
        return {str(k): to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def make_cv(problem_type: str, n_splits: int, seed: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
