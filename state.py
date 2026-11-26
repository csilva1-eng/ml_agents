from __future__ import annotations
from typing import Optional, TypedDict


class GraphState(TypedDict, total=False):
    # set once by orchestrator_brief
    briefs: dict
    profile: dict
    # round bookkeeping
    attempt: int                  # rounds completed
    pending: list                 # which specialists still owe a (re)proposal: "feature" / "model"
    retries: dict                 # {"feature": n, "model": n} bounces this round
    feature_spec: Optional[dict]  # raw proposal (untrusted)
    model_spec: Optional[dict]
    feature_code: Optional[str]     # descriptive LLM output; never executed
    model_code: Optional[str]
    feature_feedback: str         # middleman -> specialist
    model_feedback: str
    accepted: dict                # validated specs {"feature": ..., "model": ...}  <- only these can run
    rationales: dict
    round_notes: list
    guidance: str                 # orchestrator -> specialists
    # results
    history: list
    best: Optional[dict]
    report: dict                  # middleman -> orchestrator
    decision: str                 # "continue" | "stop"
    stop_reason: str
    final_result: dict
