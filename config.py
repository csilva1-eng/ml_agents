"""User-facing requirements and run settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from registry import ESTIMATORS, normalize_algorithm

DEFAULT_METRIC = {"classification": "accuracy", "regression": "r2"}
VALID_METRICS = {
    "classification": {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc"},
    "regression": {"r2"},
}


@dataclass
class Requirements:
    """What the USER tells the orchestrator."""
    problem_type: str            # "classification" | "regression"
    target: str                  # target column name
    algorithm: str               # "nn"/"mlp", "knn", "logistic_regression", "random_forest", "svm", ...
    goal: float                  # desired validation score (CV mean)
    max_tries: int               # max number of orchestrator rounds
    metric: Optional[str] = None # defaults: accuracy (classification) / r2 (regression)
    notes: str = ""              # free-text hints for the agents (domain knowledge etc.)

    def __post_init__(self):
        self.problem_type = self.problem_type.strip().lower()
        if self.problem_type not in DEFAULT_METRIC:
            raise ValueError("problem_type must be 'classification' or 'regression'")
        self.algorithm = normalize_algorithm(self.algorithm)
        if self.algorithm not in ESTIMATORS[self.problem_type]:
            raise ValueError(f"Algorithm '{self.algorithm}' not supported for {self.problem_type}. "
                             f"Options: {sorted(ESTIMATORS[self.problem_type])}")
        self.metric = self.metric or DEFAULT_METRIC[self.problem_type]
        if self.metric not in VALID_METRICS[self.problem_type]:
            raise ValueError(f"metric must be one of {sorted(VALID_METRICS[self.problem_type])}")
        if self.max_tries < 1:
            raise ValueError("max_tries must be >= 1")


@dataclass
class RunConfig:
    """Engineering knobs (not something the end user normally changes)."""
    seed: int = 42
    test_size: float = 0.2          # locked away until final_eval
    cv_folds: int = 5
    n_jobs: int = 1                 # keep 1 on a laptop unless you know better
    max_agent_retries: int = 2      # how many times the middleman bounces a specialist per round
    use_llm_review: bool = True     # middleman LLM sanity review on top of deterministic checks
    leak_threshold: float = 0.99    # single-feature score that marks a column as a likely leak
    canary_margin: float = 0.10     # shuffled-target score allowed above the dummy baseline
    overfit_gap_flag: float = 0.20  # train-val gap that gets reported as possible overfitting
    max_total_features: int = 2000  # guard against polynomial / one-hot explosions
    history_in_prompt: int = 3      # best-N and last-N runs shown to agents
    output_dir: str = "runs/latest"
