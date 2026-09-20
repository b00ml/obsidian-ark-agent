"""eval 包：Agentic RAG 与记忆的评测闭环（docs/06 §2，F14）。"""
from agentlab.eval.metrics import (  # noqa: F401
    Metrics,
    answer_relevancy,
    context_precision,
    faithfulness,
)
from agentlab.eval.eval import (  # noqa: F401
    EvalResult,
    evaluate_case,
    load_golden,
    run_eval,
)
from agentlab.eval.behavior import (  # noqa: F401
    load_behavior_scenarios,
    run_behavior_baseline,
)


def run_memory_eval(*args, **kwargs):
    """Lazy export to keep ``python -m agentlab.eval.memory_governance`` clean."""
    from agentlab.eval.memory_governance import run_memory_eval as _run
    return _run(*args, **kwargs)
