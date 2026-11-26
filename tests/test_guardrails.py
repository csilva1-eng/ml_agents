import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd, pytest
from sklearn.datasets import load_breast_cancer

from config import Requirements, RunConfig
from data import DataVault
from graph import build_graph
from guardrails import validate_feature_spec, validate_model_spec
from llm import ROLES, DummyLLM
from pipeline import build_pipeline, canary_check, run_cv

REQ = Requirements("classification", "target", "knn", 0.99, 3)
CFG = RunConfig()


@pytest.fixture(scope="module")
def vault():
    return DataVault(load_breast_cancer(as_frame=True).frame, REQ, CFG)


def test_feature_guardrails(vault):
    prof = vault.profile()
    assert validate_feature_spec({"numeric_scaler": "standard"}, prof, CFG).ok
    assert validate_feature_spec({"numeric_scaler": "standard",
                                  "architecture_code": "feature_spec = {'numeric_scaler': 'standard'}"},
                                 prof, CFG).ok
    assert not validate_feature_spec({"target_encoding": True}, prof, CFG).ok          # forbidden key
    assert not validate_feature_spec({"drop_columns": ["target"]}, prof, CFG).ok       # touches target
    assert not validate_feature_spec({"polynomial_degree": 7}, prof, CFG).ok           # out of range
    assert not validate_feature_spec({"polynomial_degree": 3}, prof, CFG).ok           # 30 numeric cols -> explosion
    assert not validate_feature_spec("import os", prof, CFG).ok                        # not JSON object


def test_model_guardrails():
    assert validate_model_spec({"hyperparameters": {"n_neighbors": 5}}, REQ).ok
    assert validate_model_spec({"hyperparameters": {"n_neighbors": 5},
                                "architecture_code": "model_config = {'n_neighbors': 5}"}, REQ).ok
    assert validate_model_spec({"n_neighbors": 5}, REQ).ok                              # lenient flat form
    assert not validate_model_spec({"hyperparameters": {"n_neighbors": 9999}}, REQ).ok
    assert not validate_model_spec({"hyperparameters": {"random_state": 1}}, REQ).ok    # agent can't set seeds
    assert not validate_model_spec({"algorithm": "svm", "hyperparameters": {}}, REQ).ok


def test_leaky_column_autoexcluded():
    df = load_breast_cancer(as_frame=True).frame
    df["leak"] = df["target"] * 3.0 + 1
    v = DataVault(df, REQ, CFG)
    assert "leak" in v.profile()["suspects"]
    assert "leak" not in v.profile()["columns"]


def test_test_set_single_use(vault):
    from guardrails import default_feature_spec, default_model_spec
    pipe = build_pipeline(default_feature_spec(), default_model_spec("knn"), vault.profile(), REQ, CFG)
    vault.final_test(pipe, "accuracy")
    with pytest.raises(RuntimeError):
        vault.final_test(pipe, "accuracy")


def test_canary_passes_on_clean_pipeline():
    v = DataVault(load_breast_cancer(as_frame=True).frame, REQ, CFG)
    from guardrails import default_feature_spec, default_model_spec
    pipe = build_pipeline(default_feature_spec(), default_model_spec("knn"), v.profile(), REQ, CFG)
    _, leaked = canary_check(pipe, v.X_dev, v.y_dev, REQ, CFG, v.profile()["baseline_score"])
    assert not leaked


def test_run_cannot_bypass_review():
    g = build_graph({r: DummyLLM() for r in ROLES}, DataVault(load_breast_cancer(as_frame=True).frame, REQ, CFG), REQ, CFG)
    edges = [(e.source, e.target) for e in g.get_graph().edges]
    assert [s for s, t in edges if t == "middleman_run"] == ["middleman_review"]


def test_agent_architecture_code_is_saved_with_each_run(vault):
    import json
    import re
    from types import SimpleNamespace
    from langgraph.checkpoint.memory import MemorySaver

    class ArchitectureLLM:
        def invoke(self, messages, **kw):
            role = re.search(r"ROLE=(\w+)", messages[0][1]).group(1)
            responses = {
                "orchestrator_brief": {"feature_agent": "", "model_agent": "", "middleman": ""},
                "feature_agent": {"numeric_scaler": "standard",
                                  "architecture_code": "feature_spec = {'numeric_scaler': 'standard'}"},
                "model_agent": {"hyperparameters": {"n_neighbors": 5},
                                "architecture_code": "model_config = {'algorithm': 'knn', 'n_neighbors': 5}"},
                "middleman_review": {"approve": True, "feature_feedback": "", "model_feedback": ""},
            }
            return SimpleNamespace(content=json.dumps(responses[role]))

    req = Requirements("classification", "target", "knn", 2.0, 1)
    cfg = RunConfig(cv_folds=2)
    llm = ArchitectureLLM()
    graph = build_graph({role: llm for role in ROLES}, vault, req, cfg, MemorySaver())
    out = graph.invoke({}, {"configurable": {"thread_id": "architecture"}, "recursion_limit": 50})

    assert len(out["history"]) == 1
    assert out["history"][0]["architecture_code"] == {
        "feature": "feature_spec = {'numeric_scaler': 'standard'}",
        "model": "model_config = {'algorithm': 'knn', 'n_neighbors': 5}",
    }


@pytest.mark.parametrize("alg,prob", [("knn", "classification"), ("logistic_regression", "classification"),
                                      ("random_forest", "classification"), ("svm", "classification"),
                                      ("mlp", "classification"), ("gradient_boosting", "classification"),
                                      ("ridge", "regression"), ("mlp", "regression"), ("svm", "regression")])
def test_every_algorithm_end_to_end(alg, prob):
    from sklearn.datasets import load_diabetes
    from langgraph.checkpoint.memory import MemorySaver
    df = (load_breast_cancer if prob == "classification" else load_diabetes)(as_frame=True).frame
    req = Requirements(prob, "target", alg, 2.0, 3)  # unreachable goal -> must stop on max_tries
    cfg = RunConfig(output_dir=f"/tmp/runs_{alg}_{prob}")
    g = build_graph({r: DummyLLM(1) for r in ROLES}, DataVault(df, req, cfg), req, cfg, MemorySaver())
    out = g.invoke({}, {"configurable": {"thread_id": "t"}, "recursion_limit": 200})
    assert out["attempt"] == 3 and "max tries" in out["final_result"]["stop_reason"]
    assert out["final_result"]["final_test_score"] is not None
