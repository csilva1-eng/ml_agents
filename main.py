"""CLI entry point.

Examples
  python main.py --demo --problem classification --algorithm knn --goal 0.98 --max-tries 8 --fake-llm
  python main.py --csv data.csv --target label --problem classification --algorithm nn --goal 0.92 --max-tries 15
  python main.py ... --checkpoint-db runs/ckpt.sqlite --thread-id exp1            # persistent
  python main.py ... --checkpoint-db runs/ckpt.sqlite --thread-id exp1 --resume   # resume after a crash
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3

import pandas as pd

from config import Requirements, RunConfig
from data import DataVault
from graph import build_graph, recursion_limit
from llm import ROLES, DummyLLM, get_llm


def load_df(args, problem):
    if args.demo:
        from sklearn.datasets import load_breast_cancer, load_diabetes
        ds = (load_breast_cancer if problem == "classification" else load_diabetes)(as_frame=True)
        return ds.frame, "target"
    return pd.read_csv(args.csv), args.target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv"); p.add_argument("--target"); p.add_argument("--demo", action="store_true")
    p.add_argument("--problem", required=True, choices=["classification", "regression"])
    p.add_argument("--algorithm", required=True)
    p.add_argument("--goal", type=float, required=True)
    p.add_argument("--max-tries", type=int, required=True)
    p.add_argument("--metric"); p.add_argument("--notes", default="")
    p.add_argument("--fake-llm", action="store_true", help="offline DummyLLM, no model needed")
    p.add_argument("--checkpoint-db"); p.add_argument("--thread-id", default="run-1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default="runs/latest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-graph", action="store_true")
    a = p.parse_args()
    if not a.demo and not (a.csv and a.target):
        p.error("provide --csv and --target (or --demo)")

    df, target = load_df(a, a.problem)
    req = Requirements(a.problem, target, a.algorithm, a.goal, a.max_tries, a.metric, a.notes)
    cfg = RunConfig(seed=a.seed, output_dir=a.output_dir)
    vault = DataVault(df, req, cfg)
    prof = vault.profile()
    print(f"[data] dev rows={prof['n_dev_rows']} features={len(prof['columns'])} "
          f"baseline {req.metric}={prof['baseline_score']:.3f}")
    if prof["suspects"]:
        print(f"[data] auto-excluded suspect columns: {prof['suspects']}")

    llms = {r: (DummyLLM(a.seed) if a.fake_llm else get_llm(r)) for r in ROLES}

    checkpointer = None
    if a.checkpoint_db:
        from langgraph.checkpoint.sqlite import SqliteSaver  # pip install langgraph-checkpoint-sqlite
        os.makedirs(os.path.dirname(a.checkpoint_db) or ".", exist_ok=True)
        checkpointer = SqliteSaver(sqlite3.connect(a.checkpoint_db, check_same_thread=False))
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    graph = build_graph(llms, vault, req, cfg, checkpointer)
    if a.print_graph:
        print(graph.get_graph().draw_mermaid())
    metadata = {
        key: value for key, value in {
            "agent_id": os.getenv("LANGSMITH_AGENT_ID"),
            "agent_environment": os.getenv("LANGSMITH_AGENT_ENVIRONMENT"),
        }.items() if value
    }
    config = {
        "configurable": {"thread_id": a.thread_id},
        "recursion_limit": recursion_limit(req, cfg),
        "metadata": metadata,
        "tags": ["ml-agent-graph"],
    }

    for update in graph.stream(None if a.resume else {}, config, stream_mode="updates"):
        for node, out in update.items():
            if not out:
                continue
            if node == "middleman_review" and out.get("pending"):
                print(f"  [middleman] rejected {out['pending']}: "
                      f"{out.get('feature_feedback') or ''} {out.get('model_feedback') or ''}".strip())
            elif node == "middleman_run":
                r = out["report"]
                s = f"{r['val_score']:.4f}" if r["val_score"] is not None else "n/a"
                print(f"[round {r['attempt']}/{req.max_tries}] {r['status']} val={s} "
                      f"best={r['best_val_score']} changes={json.dumps(r['changes'])}")
                for f in r["flags"]:
                    print(f"    ! {f}")
            elif node == "orchestrator_decide":
                print(f"  [orchestrator] {out['decision']} {out.get('stop_reason', '')}")

    state = graph.get_state(config).values
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_log.json"), "w") as f:
        json.dump({"requirements": req.__dict__, "history": state["history"],
                   "final_result": state["final_result"]}, f, indent=2, default=str)
    print("\n=== FINAL ===")
    print(json.dumps(state["final_result"], indent=2))
    print(f"(log + best_pipeline.joblib saved in {cfg.output_dir})")


if __name__ == "__main__":
    main()
