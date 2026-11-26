# Leak-proof multi-agent ML tuner (LangGraph)

An orchestrator + two specialist agents + a middleman iteratively improve a model's validation score
until the user's goal or max-tries is hit, then score a **locked test set exactly once**.

```
START -> orchestrator_brief -> feature_agent -> model_agent -> middleman_review
              feature_agent <-- (retry rejected agent only) ---------+   |
                                                                       | (approved)
                                                               middleman_run -> orchestrator_decide
              feature_agent <---------------- (continue) ------------------+  |
                                                                  (stop) -> final_eval -> END
```

| Node | LLM? | Job |
|---|---|---|
| `orchestrator_brief` | yes (+ deterministic core) | Briefs all agents once from the user's requirements |
| `feature_agent` / `model_agent` | yes | Propose a JSON spec (preprocessing / hyperparameters). Skipped if not pending |
| `middleman_review` | **code first**, optional LLM | Validates specs; bounces only the offending agent; falls back to best-so-far after `max_agent_retries` |
| `middleman_run` | no | Builds pipeline, runs CV, records changes, runs leakage canary, reports to orchestrator |
| `orchestrator_decide` | LLM for guidance only | Stop decisions are deterministic (goal reached / max tries) |
| `final_eval` | no | Refits best on all dev data, scores test set once, saves `best_pipeline.joblib` |

## Leakage guarantees (all deterministic)
1. **Test set is sealed** inside `DataVault`; no agent, prompt, or graph state ever contains it. `final_test()` raises on a second call.
2. **Agent code is descriptive only.** Feature/model agents return whitelisted specs plus an `architecture_code` snippet saved in each run's history. The snippet is never executed; only validated specs reach pipeline construction. Target encoding etc. are not whitelisted, and agents can't set seeds.
3. **Every learned transform sits inside an sklearn `Pipeline`** and is evaluated with `cross_validate`, so fits only see training folds.
4. **Suspect-column scan** (dev data only): near-perfect single-feature predictors and ID-like text columns are auto-excluded.
5. **Shuffled-target canary**: any run that would become the new best is re-run on permuted labels; if it scores above the dummy baseline + margin the result is discarded.
6. **Graph topology**: the only edge into `middleman_run` is from `middleman_review` (covered by a test).

## Run
```bash
pip install -r requirements.txt
python main.py --demo --problem classification --algorithm nn --goal 0.98 --max-tries 8 --fake-llm   # offline smoke test
python -m pytest tests -q
```

## LangSmith tracing
Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` to trace graph and model runs.
Set `LANGSMITH_PROJECT` to choose a project (otherwise LangSmith uses its default project).
Optional `LANGSMITH_AGENT_ID` and `LANGSMITH_AGENT_ENVIRONMENT` values are attached as run metadata;
all runs also carry the `ml-agent-graph` tag. Tracing is disabled by setting `LANGSMITH_TRACING=false`.

Traces can contain prompts, model responses, and run metadata. Do not enable tracing for sensitive
inputs unless sending that content to your LangSmith workspace is acceptable.

Plug your model into **`llm.py -> get_llm(role)`** (roles: orchestrator, feature_agent, model_agent, middleman; return a different model per role if you like), then drop `--fake-llm`:
```bash
python main.py --csv data.csv --target label --problem classification --algorithm knn --goal 0.93 --max-tries 15 \
    --notes "columns a,b are skewed" --checkpoint-db runs/ckpt.sqlite --thread-id exp1
# after a crash:  same command + --resume
```
Supported algorithms: `logistic_regression`, `knn`, `random_forest`, `svm`, `mlp` (alias `nn`), `gradient_boosting`, `ridge` (regression). Add more in `registry.py`.

## Things worth knowing
- Each proposal is a **full spec**; omitted keys mean "default", not "keep previous best".
- Each `history` entry in `run_log.json` includes `architecture_code.feature` and `architecture_code.model`, the specialists' accepted Python-style representations for that run.
- A "try" = one orchestrator round (including any middleman bounces inside it).
- Many rounds against one CV split can overfit the validation score; `final_test_score` is the honest number. Compare it with `validation_score`.
- The suspect scan and canary are strong heuristics, not proofs. Domain leakage (e.g. a feature only known after the outcome) needs a human; pass hints via `--notes`.
- Resume rebuilds the vault from the same data + seed, so the split is identical.
