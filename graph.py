"""The LangGraph wiring.

START -> orchestrator_brief -> feature_agent -> model_agent -> middleman_review
                                   ^                               |  (retry: only rejected agents re-run)
                                   +-------------------------------+
                                                                   | (run)
                                                            middleman_run -> orchestrator_decide
                                   feature_agent <---(continue)----+----(stop)---> final_eval -> END

The ONLY edge into `middleman_run` comes from `middleman_review`, so no proposal can be
evaluated without passing the deterministic leakage/sanity gate.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nodes import make_nodes
from state import GraphState


def build_graph(llms, vault, req, cfg, checkpointer=None):
    n = make_nodes(llms, vault, req, cfg)
    g = StateGraph(GraphState)
    for name in ("orchestrator_brief", "feature_agent", "model_agent", "middleman_review",
                 "middleman_run", "orchestrator_decide", "final_eval"):
        g.add_node(name, n[name])

    g.add_edge(START, "orchestrator_brief")
    g.add_edge("orchestrator_brief", "feature_agent")
    g.add_edge("feature_agent", "model_agent")
    g.add_edge("model_agent", "middleman_review")
    g.add_conditional_edges("middleman_review", n["route_after_review"],
                            {"retry": "feature_agent", "run": "middleman_run"})
    g.add_edge("middleman_run", "orchestrator_decide")
    g.add_conditional_edges("orchestrator_decide", n["route_after_decide"],
                            {"continue": "feature_agent", "stop": "final_eval"})
    g.add_edge("final_eval", END)
    return g.compile(checkpointer=checkpointer)


def recursion_limit(req, cfg) -> int:
    # worst case per round: (feature+model+review) * (retries+1) + run + decide
    return req.max_tries * (3 * (cfg.max_agent_retries + 1) + 2) + 10
