"""LLM plumbing.

>>> PLUG YOUR MODEL IN `get_llm()` <<<
Return any LangChain chat model (anything with `.invoke(messages)` returning an object with `.content`).
You can return a different model per role, e.g. a stronger one for the orchestrator.

Because the target model may have limited reasoning, nothing here relies on native
structured-output / tool-calling: agents are asked for plain JSON, which `call_json` extracts
and retries. Anything unparseable is simply treated as a rejected proposal by the middleman.
"""
from __future__ import annotations

import json
import math
import random
import re
from types import SimpleNamespace
import os 

from guardrails import FEATURE_SCHEMA
from registry import PARAM_SPECS
from langchain_ollama import ChatOllama
from navigator import make_llm
from dotenv import load_dotenv
load_dotenv()  

ROLES = ("orchestrator", "feature_agent", "model_agent", "middleman")

MODELS = [
    {"name": "llama",    "model": "llama-3.3-70b-instruct",     "weight": 0.3, "max_tokens": 4000},
    {"name": "gpt",      "model": "gpt-oss-120b",               "weight": 0.4, "max_tokens": 4000},
    {"name": "nemotron", "model": "nemotron-3-super-120b-a12b", "weight": 0.3, "max_tokens": 4000},
]

def get_llm(role: str):
    """role is one of ROLES. Replace the body with your model."""
    if role == 'orchestrator' or role == 'middleman':
        return make_llm(model=MODELS[1]["model"], temperature=0.2, max_tokens=4000)
    return make_llm(model=MODELS[0]["model"], temperature=0.2, max_tokens=4000)

# ----------------------------------------------------------------------------
def _text(resp) -> str:
    c = getattr(resp, "content", resp)
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


def extract_json(text: str):
    """Pull the first JSON object out of an LLM reply (handles ``` fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break
        start = text.find("{", start + 1)
    return None


def call_json(llm, system: str, user: str, retries: int = 1):
    """Invoke the LLM, return a dict or None if it never produced valid JSON."""
    for attempt in range(retries + 1):
        extra = "" if attempt == 0 else "\n\nYour previous reply was not valid JSON. Reply with ONLY one JSON object."
        out = extract_json(_text(llm.invoke([("system", system), ("human", user + extra)])))
        if out is not None:
            return out
    return None


# ----------------------------------------------------------------------------
class DummyLLM:
    """Offline stand-in that returns random-but-plausible JSON for each role, including
    occasional INVALID proposals so the middleman's reject/retry path gets exercised.
    Use `--fake-llm` to run the whole graph without any model."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def invoke(self, messages, **kw):
        system, user = messages[0][1], messages[1][1]
        role = re.search(r"ROLE=(\w+)", system).group(1)
        return SimpleNamespace(content=json.dumps(getattr(self, f"_{role}")(user)))

    # -- samplers
    def _sample(self, rule):
        r, t = self.rng, rule["type"]
        if t == "int":
            if rule.get("nullable") and r.random() < 0.5:
                return None
            return r.randint(rule["min"], rule["min"] + max(1, (rule["max"] - rule["min"]) // 3))
        if t == "float":
            lo, hi = rule["min"], rule["max"]
            if lo > 0 and hi / lo > 100:
                return math.exp(r.uniform(math.log(lo), math.log(hi)))
            return r.uniform(lo, hi)
        if t == "bool":
            return r.random() < 0.5
        if t == "choice":
            return r.choice(rule["choices"])
        if t == "int_list":
            return [r.randint(rule["min"], 64) for _ in range(r.randint(1, 2))]
        return []

    # -- roles
    def _orchestrator_brief(self, user):
        return {"feature_agent": "Start simple; then explore scaling and polynomial features.",
                "model_agent": "Tune the main capacity/regularisation parameter first.",
                "middleman": "Be strict about leakage."}

    def _feature_agent(self, user):
        if "MIDDLEMAN FEEDBACK: NONE" in user and self.rng.random() < 0.3:
            return {"numeric_scaler": "standard", "target_encoding": True}  # forbidden key
        spec = {k: self._sample(v) for k, v in FEATURE_SCHEMA.items()
                if v["type"] != "columns" and self.rng.random() < 0.5}
        if spec.get("polynomial_degree", 1) == 3:
            spec["polynomial_degree"] = 2
        spec["rationale"] = "random exploration (dummy LLM)"
        spec["architecture_code"] = "feature_spec = " + repr({k: v for k, v in spec.items() if k != "rationale"})
        return spec

    def _model_agent(self, user):
        alg = re.search(r"ALGORITHM: (\w+)", user).group(1)
        ptype = re.search(r"PROBLEM_TYPE: (\w+)", user).group(1)
        if "MIDDLEMAN FEEDBACK: NONE" in user and self.rng.random() < 0.2:
            return {"hyperparameters": {"bogus_param": 1}}
        rules = {k: v for k, v in PARAM_SPECS[alg].items() if v.get("only") in (None, ptype)}
        keys = self.rng.sample(sorted(rules), k=min(len(rules), self.rng.randint(1, 3)))
        hyperparameters = {k: self._sample(rules[k]) for k in keys}
        return {"hyperparameters": hyperparameters,
            "rationale": "random exploration (dummy LLM)",
            "architecture_code": "model_config = " + repr({"algorithm": alg,
                                    "hyperparameters": hyperparameters})}

    def _middleman_review(self, user):
        if self.rng.random() < 0.15:
            return {"approve": False, "feature_feedback": "Too aggressive; simplify.", "model_feedback": ""}
        return {"approve": True, "feature_feedback": "", "model_feedback": ""}

    def _orchestrator_decide(self, user):
        return {"assessment": "Not at goal yet.", "guidance": "Try a different capacity/regularisation setting."}
