"""Shared RAGAS + DeepEval + custom G-Eval scoring, reused by Stage 8 and the
end-to-end run (Requirement 11). Calling pattern adapted from the proven,
already-debugged implementation in
../../notebooks/production_rag_chatbot_memory/eval_pipeline.py.
"""
from __future__ import annotations

import concurrent.futures
import sys
import types

import numpy as np

CALL_TIMEOUT_S = 60

# ragas 0.4.3 unconditionally imports langchain_community.chat_models.vertexai
# for a static isinstance() check that's never exercised (this app is OpenAI-only).
# That submodule no longer exists in current langchain-community releases, so the
# import crashes before ragas is usable. Stub it before ragas is ever imported —
# safe because Vertex AI is never instantiated here.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertex_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _StubChatVertexAI:  # pragma: no cover - compatibility shim only, never instantiated
        pass

    _vertex_stub.ChatVertexAI = _StubChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertex_stub

from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy as RagasAnswerRelevancy,
    ContextPrecisionWithReference as RagasContextPrecision,
    ContextRecall as RagasContextRecall,
    Faithfulness as RagasFaithfulness,
)

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval, HallucinationMetric  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402


def _call_with_timeout(fn, timeout=CALL_TIMEOUT_S):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn)
        return future.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False)


def _retry_once(fn, timeout=CALL_TIMEOUT_S):
    """One retry, each attempt time-bounded — catches the common transient judge-call
    failure without letting a single flaky call abort or hang the whole run."""
    try:
        return _call_with_timeout(fn, timeout), None
    except Exception:
        try:
            return _call_with_timeout(fn, timeout), None
        except Exception as exc2:
            return None, str(exc2)


def _avg(vals):
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def evaluate_ragas(items_with_answers: list[dict], model: str = "gpt-4o-mini",
                    embedding_model: str = "text-embedding-3-small") -> dict:
    """`items_with_answers`: [{question, answer, contexts: [str], reference: str}].
    Faithfulness/AnswerRelevancy need no ground truth; ContextPrecision/ContextRecall
    use `reference` (the cited source chunk text) as the reference answer content."""
    client = AsyncOpenAI()
    llm = llm_factory(model=model, provider="openai", client=client)
    embeddings = RagasOpenAIEmbeddings(client=client, model=embedding_model)

    faithfulness = RagasFaithfulness(llm=llm)
    answer_relevancy = RagasAnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = RagasContextPrecision(llm=llm)
    context_recall = RagasContextRecall(llm=llm)

    def _score_metric(metric, build_kwargs):
        inputs = [build_kwargs(it) for it in items_with_answers]
        batch_timeout = max(CALL_TIMEOUT_S, 15 * len(inputs))
        batch_result, _ = _retry_once(lambda: metric.batch_score(inputs), timeout=batch_timeout)
        if batch_result is not None:
            return [r.value for r in batch_result]
        values = []
        for kw in inputs:
            val, _ = _retry_once(lambda k=kw: metric.score(**k).value)
            values.append(val)
        return values

    faith_vals = _score_metric(
        faithfulness,
        lambda it: {"user_input": it["question"], "response": it["answer"], "retrieved_contexts": it["contexts"]},
    )
    ar_vals = _score_metric(answer_relevancy, lambda it: {"user_input": it["question"], "response": it["answer"]})
    cp_vals = _score_metric(
        context_precision,
        lambda it: {"user_input": it["question"], "reference": it["reference"], "retrieved_contexts": it["contexts"]},
    )
    cr_vals = _score_metric(
        context_recall,
        lambda it: {"user_input": it["question"], "retrieved_contexts": it["contexts"], "reference": it["reference"]},
    )

    return {
        "faithfulness": _avg(faith_vals), "answer_relevancy": _avg(ar_vals),
        "context_precision": _avg(cp_vals), "context_recall": _avg(cr_vals),
    }


STRICT_GROUNDING_CRITERIA = (
    "Determine whether the actual output makes any factual claim — especially any dollar amount, "
    "percentage, APR, or fee figure — that is not directly supported by the retrieval context. "
    "Penalize any claim not grounded in the provided context, even if the claim happens to be true "
    "in the real world: this assistant must only speak from the retrieved Wells Fargo documents."
)


def evaluate_deepeval(items_with_answers: list[dict], model: str = "gpt-4o-mini") -> dict:
    """Requirement 11.2: Hallucination, Answer Relevancy, Faithfulness, and a custom
    G-Eval "Strict Grounding" criterion. async_mode=True is required, not stylistic —
    FaithfulnessMetric with async_mode=False is known to hang against this deepeval
    version (see eval_pipeline.py's note); do not change this."""
    faithfulness_m = FaithfulnessMetric(model=model, async_mode=True, include_reason=False)
    answer_rel_m = AnswerRelevancyMetric(model=model, async_mode=True, include_reason=False)
    hallucination_m = HallucinationMetric(model=model, async_mode=True, include_reason=False)
    grounding_m = GEval(
        name="Strict Grounding", model=model, async_mode=True, criteria=STRICT_GROUNDING_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    )
    metrics = [
        ("faithfulness", faithfulness_m), ("answer_relevancy", answer_rel_m),
        ("hallucination", hallucination_m), ("g_eval_strict_grounding", grounding_m),
    ]

    totals = {name: [] for name, _ in metrics}
    for it in items_with_answers:
        reference = it.get("reference", "")
        test_case = LLMTestCase(
            input=it["question"], actual_output=it["answer"], retrieval_context=it["contexts"],
            expected_output=reference, context=[reference] if reference else it["contexts"][:1],
        )
        for name, metric in metrics:
            score, _ = _retry_once(lambda m=metric, t=test_case: (m.measure(t), m.score)[1])
            if score is not None:
                totals[name].append(score)

    return {name: _avg(totals[name]) for name, _ in metrics}
