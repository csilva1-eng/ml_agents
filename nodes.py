"""Graph nodes. `make_nodes` closes over the LLMs, the data vault and the user requirements,
so none of those (and no dataframes) ever need to live in the serialisable graph state."""
from __future__ import annotations

import json
import os

import joblib

import prompts
from guardrails import (FEATURE_SCHEMA, default_feature_spec, default_model_spec, describe_rules,
                        diff_specs, fingerprint, allowed_params, validate_feature_spec,
                        validate_model_spec)
from llm import call_json
from pipeline import build_pipeline, canary_check, run_cv
from utils import to_native


def _j(o) -> str:
    return json.dumps(o, default=str, separators=(",", ":"))


def _architecture_code(proposal) -> str | None:
    if not isinstance(proposal, dict):
        return None
    code = proposal.get("architecture_code")
    return code.strip()[:4000] if isinstance(code, str) and code.strip() else None


def make_nodes(llms: dict, vault, req, cfg):
    profile = vault.profile()

    # ------------------------------------------------------------------ helpers
    def profile_text() -> str:
        cols = {c: f"{s['kind'][:3]},miss={s['missing_frac']},uniq={s['nunique']}"
                for c, s in list(profile["column_stats"].items())[:80]}
        return _j({"n_dev_rows": profile["n_dev_rows"], "columns": cols,
                   "auto_excluded_suspect_columns": profile["suspects"],
                   "target": profile["target_summary"], "baseline_score": round(profile["baseline_score"], 4)})

    def history_text(history) -> str:
        if not history:
            return "(no runs yet)"
        ok = sorted([h for h in history if h["status"] == "ok"], key=lambda h: -h["val_score"])
        picks = {h["attempt"]: h for h in ok[:cfg.history_in_prompt] + history[-cfg.history_in_prompt:]}
        lines = []
        for h in sorted(picks.values(), key=lambda h: h["attempt"]):
            score = f"val={h['val_score']:.4f} train={h['train_score']:.4f}" if "val_score" in h else h.get("error", "")
            lines.append(f"#{h['attempt']} [{h['status']}] {score} feature={_j(h['feature_spec'])} "
                         f"model={_j(h['model_spec']['hyperparameters'])}")
        return "\n".join(lines)

    def header(state, extra="") -> str:
        return (f"PROBLEM_TYPE: {req.problem_type}\nALGORITHM: {req.algorithm}\n"
                f"METRIC: {req.metric}  GOAL: {req.goal}\n"
                f"ROUND: {state['attempt'] + 1} of {req.max_tries}\n{extra}")

    def last_specs(state):
        h = state.get("history") or []
        if h:
            return h[-1]["feature_spec"], h[-1]["model_spec"]
        return default_feature_spec(), default_model_spec(req.algorithm)

    def fallback_spec(agent, state):
        best = state.get("best")
        if best:
            return best["feature_spec"] if agent == "feature" else best["model_spec"]
        return default_feature_spec() if agent == "feature" else default_model_spec(req.algorithm)

    # ------------------------------------------------------------------ nodes
    def orchestrator_brief(state):
        core = {
            "feature_agent": "Improve preprocessing/feature engineering for the fixed algorithm.",
            "model_agent": f"Tune '{req.algorithm}' hyperparameters.",
            "middleman": "Validate both proposals; send back anything leaky or unreasonable.",
        }
        user = (header({"attempt": 0}) + f"USER NOTES: {req.notes or 'none'}\nDATA PROFILE: {profile_text()}\n"
                "Write the briefs.")
        data = call_json(llms["orchestrator"], prompts.ORCH_BRIEF_SYSTEM, user) or {}
        briefs = {k: f"{v}\n{prompts.HARD_RULES}" + (f"\nStrategy notes: {data[k]}" if isinstance(data.get(k), str) else "")
                  for k, v in core.items()}
        return {"briefs": briefs, "profile": profile, "attempt": 0, "pending": ["feature", "model"],
                "retries": {"feature": 0, "model": 0}, "accepted": {}, "history": [], "best": None,
                "guidance": "", "feature_feedback": "", "model_feedback": "", "round_notes": [],
                "rationales": {}, "feature_spec": None, "model_spec": None,
                "feature_code": None, "model_code": None}

    def feature_agent(state):
        if "feature" not in state["pending"]:
            return {}
        user = (header(state) + f"YOUR BRIEF: {state['briefs']['feature_agent']}\n"
                f"ORCHESTRATOR GUIDANCE: {state.get('guidance') or 'none'}\n"
                f"DATA PROFILE: {profile_text()}\nALLOWED KEYS:\n{describe_rules(FEATURE_SCHEMA)}\n"
                f"RECENT RUNS:\n{history_text(state['history'])}\n"
                f"MIDDLEMAN FEEDBACK: {state.get('feature_feedback') or 'NONE'}\n"
                "Reply with ONLY the JSON spec.")
        proposal = call_json(llms["feature_agent"], prompts.FEATURE_SYSTEM, user)
        return {"feature_spec": proposal, "feature_code": _architecture_code(proposal)}

    def model_agent(state):
        if "model" not in state["pending"]:
            return {}
        rules = allowed_params(req.algorithm, req.problem_type)
        user = (header(state) + f"YOUR BRIEF: {state['briefs']['model_agent']}\n"
                f"ORCHESTRATOR GUIDANCE: {state.get('guidance') or 'none'}\n"
                f"ALLOWED HYPERPARAMETERS:\n{describe_rules(rules)}\n"
                f"RECENT RUNS:\n{history_text(state['history'])}\n"
                f"MIDDLEMAN FEEDBACK: {state.get('model_feedback') or 'NONE'}\n"
                'Reply with ONLY JSON like {"hyperparameters": {...}, "rationale": "..."}.')
        proposal = call_json(llms["model_agent"], prompts.MODEL_SYSTEM, user)
        return {"model_spec": proposal, "model_code": _architecture_code(proposal)}

    def middleman_review(state):
        """Gate. Deterministic validation first; optional LLM judgement second."""
        pending, accepted = list(state["pending"]), dict(state.get("accepted", {}))
        retries, notes = dict(state["retries"]), list(state.get("round_notes", []))
        rationales = dict(state.get("rationales", {}))
        code = {"feature": state.get("feature_code"), "model": state.get("model_code")}
        feedback, rejected = {"feature": "", "model": ""}, []

        for agent in pending:
            raw = state["feature_spec"] if agent == "feature" else state["model_spec"]
            v = (validate_feature_spec(raw, profile, cfg) if agent == "feature"
                 else validate_model_spec(raw, req))
            if isinstance(raw, dict):
                rationales[agent] = str(raw.get("rationale", ""))[:300]
            if v.ok:
                accepted[agent] = v.cleaned
                code[agent] = _architecture_code(raw)
                notes += [f"{agent}: {w}" for w in v.warnings]
            else:
                accepted[agent] = None
                code[agent] = None
                rejected.append(agent)
                feedback[agent] = "; ".join(v.errors)

        # LLM judgement (can only veto while retries remain, so it can never cause an endless loop)
        if not rejected and cfg.use_llm_review:
            pf, pm = last_specs(state)
            user = (header(state) + f"PREVIOUS: feature={_j(pf)} model={_j(pm['hyperparameters'])}\n"
                    f"PROPOSED: feature={_j(accepted.get('feature'))} model={_j(accepted.get('model', {}).get('hyperparameters'))}\n"
                    f"DATA PROFILE: {profile_text()}\nReview now.")
            verdict = call_json(llms["middleman"], prompts.REVIEW_SYSTEM, user)
            if verdict and verdict.get("approve") is False:
                for agent in pending:
                    msg = str(verdict.get(f"{agent}_feedback") or "").strip()
                    if msg and retries[agent] < cfg.max_agent_retries:
                        accepted[agent] = None
                        code[agent] = None
                        rejected.append(agent)
                        feedback[agent] = "Middleman review: " + msg

        for agent in list(rejected):
            retries[agent] += 1
            if retries[agent] > cfg.max_agent_retries:  # specialist keeps failing -> safe fallback
                accepted[agent] = fallback_spec(agent, state)
                code[agent] = None
                notes.append(f"{agent} agent failed validation {retries[agent]}x; fell back to "
                             f"{'best-so-far' if state.get('best') else 'defaults'}. Last errors: {feedback[agent]}")
                rejected.remove(agent)
                feedback[agent] = ""

        return {"accepted": accepted, "pending": rejected, "retries": retries, "round_notes": notes,
                "rationales": rationales, "feature_feedback": feedback["feature"],
            "model_feedback": feedback["model"], "feature_code": code["feature"],
            "model_code": code["model"]}

    def middleman_run(state):
        acc = state.get("accepted") or {}
        assert acc.get("feature") and acc.get("model"), "run reached without validated specs (graph wiring bug)"
        fs, ms = acc["feature"], acc["model"]
        history, best, attempt = list(state["history"]), state.get("best"), state["attempt"]
        pf, pm = last_specs(state)
        fp = fingerprint(fs, ms)
        entry = {"attempt": attempt + 1, "fingerprint": fp, "feature_spec": fs, "model_spec": ms,
                 "changes": {"feature": diff_specs(pf, fs), "model": diff_specs(pm["hyperparameters"], ms["hyperparameters"])},
                 "architecture_code": {"feature": state.get("feature_code"), "model": state.get("model_code")},
                 "rationales": state.get("rationales", {}), "status": "ok"}
        flags = list(state.get("round_notes", []))

        if fp in {h["fingerprint"] for h in history}:
            entry["status"] = "skipped_duplicate"
            flags.append("Identical configuration was already evaluated; no compute spent.")
        else:
            try:
                pipe = build_pipeline(fs, ms, profile, req, cfg)
                entry.update(run_cv(pipe, vault.X_dev, vault.y_dev, req, cfg))
                if best is None or entry["val_score"] > best["val_score"]:
                    c_score, leaked = canary_check(pipe, vault.X_dev, vault.y_dev, req, cfg, profile["baseline_score"])
                    entry["canary_score"] = c_score
                    if leaked:
                        entry["status"] = "rejected_leakage_suspect"
                        flags.append(f"LEAKAGE CANARY TRIPPED: shuffled-target score {c_score:.3f} vs baseline "
                                     f"{profile['baseline_score']:.3f}. Result discarded.")
                if entry["gap"] > cfg.overfit_gap_flag:
                    flags.append(f"Possible overfitting: train-val gap {entry['gap']:.3f}")
            except Exception as e:  # a crashing config is a data point, not a crash of the graph
                entry["status"], entry["error"] = "error", f"{type(e).__name__}: {str(e)[:250]}"
                flags.append(entry["error"])

        entry = to_native(entry)
        if entry["status"] == "ok" and (best is None or entry["val_score"] > best["val_score"]):
            best = entry
        history.append(entry)
        report = to_native({
            "attempt": entry["attempt"], "status": entry["status"], "val_score": entry.get("val_score"),
            "train_score": entry.get("train_score"), "best_val_score": best["val_score"] if best else None,
            "best_attempt": best["attempt"] if best else None, "goal": req.goal,
            "changes": entry["changes"], "flags": flags})
        return {"history": history, "best": best, "report": report, "attempt": attempt + 1}

    def orchestrator_decide(state):
        best, attempt, report = state.get("best"), state["attempt"], state["report"]
        if best and best["val_score"] >= req.goal:
            return {"decision": "stop", "stop_reason": f"goal reached ({best['val_score']:.4f} >= {req.goal})"}
        if attempt >= req.max_tries:
            return {"decision": "stop", "stop_reason": f"max tries reached ({req.max_tries})"}
        user = (header(state) + f"MIDDLEMAN REPORT: {_j(report)}\nRUNS:\n{history_text(state['history'])}\n"
                f"Rounds left: {req.max_tries - attempt}")
        data = call_json(llms["orchestrator"], prompts.DECIDE_SYSTEM, user) or {}
        return {"decision": "continue", "stop_reason": "",
                "guidance": str(data.get("guidance", ""))[:600],
                # fresh round
                "pending": ["feature", "model"], "retries": {"feature": 0, "model": 0}, "accepted": {},
                "feature_spec": None, "model_spec": None, "feature_feedback": "", "model_feedback": "",
                "round_notes": [], "rationales": {}, "feature_code": None, "model_code": None}

    def final_eval(state):
        best = state.get("best")
        if not best:
            return {"final_result": {"error": "No successful run; test set was never touched.",
                                     "stop_reason": state.get("stop_reason")}}
        pipe = build_pipeline(best["feature_spec"], best["model_spec"], profile, req, cfg)
        test_score, fitted = vault.final_test(pipe, req.metric)  # the ONLY test-set access
        os.makedirs(cfg.output_dir, exist_ok=True)
        joblib.dump(fitted, os.path.join(cfg.output_dir, "best_pipeline.joblib"))
        return {"final_result": to_native({
            "metric": req.metric, "goal": req.goal, "best_attempt": best["attempt"],
            "validation_score": best["val_score"], "final_test_score": test_score,
            "feature_spec": best["feature_spec"], "model_spec": best["model_spec"],
            "rounds_used": state["attempt"], "stop_reason": state.get("stop_reason")})}

    def route_after_review(state):
        return "retry" if state["pending"] else "run"

    def route_after_decide(state):
        return "stop" if state["decision"] == "stop" else "continue"

    return {"orchestrator_brief": orchestrator_brief, "feature_agent": feature_agent, "model_agent": model_agent,
            "middleman_review": middleman_review, "middleman_run": middleman_run,
            "orchestrator_decide": orchestrator_decide, "final_eval": final_eval,
            "route_after_review": route_after_review, "route_after_decide": route_after_decide}
