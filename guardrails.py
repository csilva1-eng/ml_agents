"""DETERMINISTIC guardrails. No LLM is involved in anything in this file.

Agents output JSON *specs*. Specs are validated against whitelists and ranges here,
and only validated specs can ever reach pipeline construction. Free-form code from an
agent is never executed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Optional

from registry import PARAM_SPECS, normalize_algorithm

# ----------------------------------------------------------------------------
# Feature / preprocessing whitelist
# (target encoding & friends are deliberately absent: they are leakage-prone)
# ----------------------------------------------------------------------------
FEATURE_SCHEMA = {
    "numeric_impute": {"type": "choice", "choices": ["mean", "median", "most_frequent"], "default": "median"},
    "numeric_scaler": {"type": "choice", "choices": ["none", "standard", "minmax", "robust"], "default": "standard"},
    "categorical_encoding": {"type": "choice", "choices": ["onehot", "ordinal"], "default": "onehot"},
    "polynomial_degree": {"type": "int", "min": 1, "max": 3, "default": 1},
    "interaction_only": {"type": "bool", "default": False},
    "log_transform_columns": {"type": "columns", "default": []},
    "drop_columns": {"type": "columns", "default": []},
    "variance_threshold": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.0},
    "select_k_best": {"type": "int", "min": 1, "max": 500, "nullable": True, "default": None},
}
IGNORED_KEYS = {"rationale", "architecture_code"}  # descriptive metadata only; never used as config


@dataclass
class Validation:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    cleaned: Optional[dict] = None


def default_feature_spec() -> dict:
    return {k: copy.deepcopy(v.get("default")) for k, v in FEATURE_SCHEMA.items()}


def default_model_spec(algorithm: str) -> dict:
    return {"algorithm": algorithm, "hyperparameters": {}}


# ----------------------------------------------------------------------------
# generic value checking
# ----------------------------------------------------------------------------
def _check(name, rule, value):
    """Return (cleaned_value, error_or_None)."""
    t = rule["type"]
    if value is None:
        if rule.get("nullable") or (t == "choice" and None in rule["choices"]):
            return None, None
        return None, f"'{name}' may not be null"
    if t == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                (isinstance(value, float) and not value.is_integer()):
            return None, f"'{name}' must be an integer"
        v = int(value)
        if not rule["min"] <= v <= rule["max"]:
            return None, f"'{name}'={v} is outside the allowed range [{rule['min']}, {rule['max']}]"
        return v, None
    if t == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None, f"'{name}' must be a finite number"
        v = float(value)
        if not rule["min"] <= v <= rule["max"]:
            return None, f"'{name}'={v} is outside the allowed range [{rule['min']}, {rule['max']}]"
        return v, None
    if t == "bool":
        if not isinstance(value, bool):
            return None, f"'{name}' must be true or false"
        return value, None
    if t == "choice":
        if isinstance(value, str) and value.lower() in ("none", "null") and None in rule["choices"]:
            return None, None
        for c in rule["choices"]:
            if c is None:
                continue
            if isinstance(c, str) and isinstance(value, str) and c == value:
                return c, None
            if (isinstance(c, (int, float)) and not isinstance(c, bool) and isinstance(value, (int, float))
                    and not isinstance(value, bool) and c == value):
                return c, None
        return None, f"'{name}'={value!r} is not allowed; choose one of {rule['choices']}"
    if t == "int_list":
        if not isinstance(value, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            return None, f"'{name}' must be a list of integers"
        if not rule["min_len"] <= len(value) <= rule["max_len"]:
            return None, f"'{name}' must have {rule['min_len']}-{rule['max_len']} entries"
        if any(not rule["min"] <= x <= rule["max"] for x in value):
            return None, f"every entry of '{name}' must be in [{rule['min']}, {rule['max']}]"
        return list(value), None
    if t == "columns":
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            return None, f"'{name}' must be a list of column-name strings"
        return list(dict.fromkeys(value)), None
    return None, f"unknown rule type for '{name}'"


# ----------------------------------------------------------------------------
# feature spec validation
# ----------------------------------------------------------------------------
def _count_poly(n: int, degree: int, interaction_only: bool) -> int:
    if degree <= 1 or n == 0:
        return n
    if interaction_only:
        return sum(math.comb(n, k) for k in range(1, min(degree, n) + 1))
    return math.comb(n + degree, degree) - 1


def validate_feature_spec(raw, profile: dict, cfg) -> Validation:
    if not isinstance(raw, dict):
        return Validation(False, ["Output was not a valid JSON object"])
    raw = {k: v for k, v in raw.items() if k not in IGNORED_KEYS}
    errors, warnings = [], []

    unknown = sorted(set(raw) - set(FEATURE_SCHEMA))
    if unknown:
        errors.append(f"Unknown / forbidden keys {unknown}. Allowed keys: {sorted(FEATURE_SCHEMA)}")

    cleaned = {}
    for name, rule in FEATURE_SCHEMA.items():
        if name not in raw:
            cleaned[name] = copy.deepcopy(rule.get("default"))
            continue
        val, err = _check(name, rule, raw[name])
        if err:
            errors.append(err)
        else:
            cleaned[name] = val

    target = profile["target"]
    usable = set(profile["columns"])
    numeric = set(profile["numeric_columns"])
    known = usable | set(profile.get("suspects", {})) | set(profile.get("unsupported", []))
    for name in ("drop_columns", "log_transform_columns"):
        for col in cleaned.get(name) or []:
            if col == target:
                errors.append(f"'{name}' references the target column '{target}'. The target must never be touched.")
            elif col not in known:
                errors.append(f"'{name}' references unknown column '{col}'")
            elif name == "log_transform_columns" and col not in numeric:
                errors.append(f"'log_transform_columns': '{col}' is not a usable numeric column")

    if not errors:
        dropped = set(cleaned["drop_columns"])
        for col in set(cleaned["log_transform_columns"]) & dropped:
            warnings.append(f"'{col}' is both dropped and log-transformed; it will be dropped")
        remaining = usable - dropped
        if not remaining:
            errors.append("drop_columns would remove every feature")
        else:
            n_num = len(numeric - dropped)
            est = _count_poly(n_num, cleaned["polynomial_degree"], cleaned["interaction_only"])
            for c in remaining - numeric:
                est += min(profile["column_stats"][c]["nunique"], 50) if cleaned["categorical_encoding"] == "onehot" else 1
            if est > cfg.max_total_features:
                errors.append(f"Spec would create ~{est} features (limit {cfg.max_total_features}); "
                              "lower polynomial_degree, use interaction_only, or drop columns")
    return Validation(not errors, errors, warnings, cleaned if not errors else None)


# ----------------------------------------------------------------------------
# model spec validation
# ----------------------------------------------------------------------------
def allowed_params(algorithm: str, problem_type: str) -> dict:
    return {k: v for k, v in PARAM_SPECS[algorithm].items() if v.get("only") in (None, problem_type)}


def validate_model_spec(raw, req) -> Validation:
    if not isinstance(raw, dict):
        return Validation(False, ["Output was not a valid JSON object"])
    raw = {k: v for k, v in raw.items() if k not in IGNORED_KEYS}
    rules = allowed_params(req.algorithm, req.problem_type)
    errors = []

    # leniency for small LLMs: accept flat {"C": 1.0} as {"hyperparameters": {"C": 1.0}}
    if "hyperparameters" not in raw and raw and set(raw) <= (set(rules) | {"algorithm"}):
        raw = {"algorithm": raw.get("algorithm", req.algorithm),
               "hyperparameters": {k: v for k, v in raw.items() if k != "algorithm"}}

    extra = sorted(set(raw) - {"algorithm", "hyperparameters"})
    if extra:
        errors.append(f"Unknown keys {extra}. Use only 'hyperparameters' (and optional 'rationale')")
    if "algorithm" in raw and normalize_algorithm(str(raw["algorithm"])) != req.algorithm:
        errors.append(f"The algorithm is fixed to '{req.algorithm}' and may not be changed")

    hp = raw.get("hyperparameters", {})
    if hp is None:
        hp = {}
    if not isinstance(hp, dict):
        return Validation(False, errors + ["'hyperparameters' must be an object"])

    bad = sorted(set(hp) - set(rules))
    if bad:
        errors.append(f"Hyperparameters {bad} are not allowed for {req.algorithm}/{req.problem_type}. "
                      f"Allowed: {sorted(rules)}")
    cleaned_hp = {}
    for k, v in hp.items():
        if k in rules:
            val, err = _check(k, rules[k], v)
            if err:
                errors.append(err)
            else:
                cleaned_hp[k] = val
    cleaned = {"algorithm": req.algorithm, "hyperparameters": cleaned_hp}
    return Validation(not errors, errors, [], cleaned if not errors else None)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fingerprint(feature_spec: dict, model_spec: dict) -> str:
    blob = json.dumps({"f": feature_spec, "m": model_spec}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def diff_specs(prev: dict, new: dict) -> dict:
    prev, new = prev or {}, new or {}
    out = {}
    for k in sorted(set(prev) | set(new)):
        if prev.get(k) != new.get(k):
            out[k] = [prev.get(k), new.get(k)]
    return out


def describe_rules(schema: dict) -> str:
    lines = []
    for name, r in schema.items():
        t = r["type"]
        if t in ("int", "float"):
            d = f"{t} in [{r['min']}, {r['max']}]" + (" or null" if r.get("nullable") else "")
        elif t == "choice":
            d = "one of " + json.dumps(r["choices"])
        elif t == "bool":
            d = "true/false"
        elif t == "int_list":
            d = f"list of {r['min_len']}-{r['max_len']} ints, each in [{r['min']}, {r['max']}]"
        else:
            d = "list of column names"
        lines.append(f"- {name}: {d}")
    return "\n".join(lines)
