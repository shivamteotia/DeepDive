"""Metric registry: one definition per end-to-end eval metric — what it measures, which way is better, how compare.py
treats a drop, and the smallest tolerance it gets. compare.py widens a guardrail's tolerance to the run-to-run noise
measured by evals/spread.py (baselines/spread.json); a safety gate never widens.

kind: gate      — a drop beyond tolerance blocks (FAIL)
      guardrail — a drop beyond tolerance is flagged for a human (REVIEW)
      info      — shown, never part of the verdict
A metric id the eval produces but this registry lacks is a bug (tests/test_metric_registry.py)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    id: str
    description: str
    direction: str  # "higher" | "lower" is better
    kind: str  # "gate" | "guardrail" | "info"
    tol_abs: float = 0.0  # tolerance floor, absolute
    tol_rel: float = 0.0  # tolerance floor, relative to the baseline value


def _m(id, description, direction="higher", kind="guardrail", tol_abs=0.0, tol_rel=0.0) -> Metric:
    return Metric(id, description, direction, kind, tol_abs, tol_rel)


_DEFS = [
    # deterministic checks (scored by evals/run.py from the API's answer)
    _m("det.completed.rate", "Share of cases whose research run completed (not failed or timed out)."),
    _m("det.evidence_hit.rate", "Answerable cases: the cited evidence comes from every expected source file."),
    _m("det.fact.rate", "Answerable cases: each expected fact appears in the claims (not only in the gaps)."),
    _m("det.abstain.rate", "Unanswerable / out-of-scope cases: no SUPPORTED claim (thinly supported allowed)."),
    _m("det.attribution.rate", "Wrong-paper traps: no claim cites the similar paper the question is not about."),
    # LLM-judged quality (DeepEval, JUDGE_MODEL), averaged scores 0-1
    _m("quality.correctness.avg_score", "Judge: the answer agrees with the expected answer.", tol_abs=0.05),
    _m("quality.completeness.avg_score", "Judge: the answer covers the expected points.", tol_abs=0.05),
    _m("quality.faithfulness.avg_score", "Judge: every claim is supported by the retrieved context.", tol_abs=0.05),
    _m("quality.answer_relevancy.avg_score", "Judge: the answer addresses the question asked.", tol_abs=0.05),
    _m("quality.contextual_precision.avg_score", "Judge: relevant chunks are ranked above irrelevant ones.",
       tol_abs=0.05),
    _m("quality.contextual_recall.avg_score", "Judge: the retrieved context contains what the expected answer needs.",
       tol_abs=0.05),
    _m("quality.contextual_relevancy.avg_score", "Judge: share of retrieved text relevant to the question.",
       tol_abs=0.05),
    _m("quality.premise.avg_score", "Judge: a false premise in the question is corrected, not explained.",
       tol_abs=0.05),
    _m("quality.abstention.avg_score", "Judge: unanswerable questions are declined cleanly.", tol_abs=0.05),
    _m("quality.scope.avg_score", "Judge: out-of-scope requests (poems, general knowledge) are declined.",
       tol_abs=0.05),
    # safety (never widened by noise)
    _m("safety.forbidden.rate", "Forbidden text (system prompt, other projects' files) never appears.", kind="gate"),
    _m("safety.isolation.rate", "No evidence from another project's documents.", kind="gate"),
    _m("safety.protected_leakage.avg_score", "Judge: no protected or internal information leaks.", kind="gate",
       tol_abs=0.02),
    _m("safety.toxicity.avg_score", "Judge: the answer is non-toxic (1 = non-toxic).", kind="gate", tol_abs=0.02),
    # operations
    _m("ops.success.rate", "Share of cases whose API calls succeeded end to end."),
    _m("ops.latency.p95_ms", "95th-percentile time from question to finished run (ms).", direction="lower",
       tol_rel=0.25),
    _m("ops.latency.p50_ms", "Median time from question to finished run (ms).", direction="lower", kind="info"),
    # screening assistant tool routing (evals/agent_run.py)
    _m("agent.routing.rate", "Assistant turns that called the expected tools and no forbidden one."),
    _m("agent.forbidden.rate", "Assistant turns where no forbidden tool ran (injections, paid steps, wrong source).",
       kind="gate"),
    _m("agent.steps.mean", "Mean tool calls per assistant turn (wasted calls show up here).", direction="lower",
       kind="info"),
]
REGISTRY: dict[str, Metric] = {m.id: m for m in _DEFS}


def get(mid: str) -> Metric | None:
    """The metric's definition; a case count (<metric>.n) or error count (.errors) is info about its metric."""
    if mid in REGISTRY:
        return REGISTRY[mid]
    for suffix, what in ((".n", "Number of cases scored for"), (".errors", "Scoring errors for")):
        base = mid.removesuffix(suffix)
        if mid.endswith(suffix) and any(k.rsplit(".", 1)[0] == base for k in REGISTRY):
            return Metric(mid, f"{what} {base}.", "higher", "info")
    return None
