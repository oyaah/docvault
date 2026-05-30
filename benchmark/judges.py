"""LLM-as-judge metrics for the DocVault benchmark.

All metrics are semantic (no substring matching against a golden answer). Each
judge returns a float in [0, 1] plus a short rationale.

Bias mitigation:
  - Use a judge model from a DIFFERENT family than the generator (default
    gpt-4o judging gpt-4o-mini answers) to avoid self-enhancement bias.
  - temperature=0 for reproducibility.
  - Faithfulness / context metrics are reference-free (judged against retrieved
    context, not a gold answer); correctness is reference-guided.
  - Rubrics penalise verbosity and ignore answer order to limit length/position bias.

Metric set (RAGAS / RAG-triad parity + DocVault-specific):
  retrieval : context_precision, context_recall
  generation: faithfulness (-> hallucination = 1 - faithfulness),
              answer_relevancy, answer_correctness
  behaviour : citation_accuracy, refusal_correctness
"""

import json

import litellm

REFUSAL_MARKERS = ("i couldn't find", "i could not find", "please check with", "i don't know")


def _judge(prompt: str, model: str, retries: int = 3) -> dict:
    for _ in range(retries):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"},
                num_retries=4,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            continue
    return {}


def _ctx_text(contexts: list[dict]) -> str:
    return "\n\n---\n\n".join(c.get("content", "") for c in contexts) or "(no context retrieved)"


def is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in REFUSAL_MARKERS)


# ── Generation metrics ──────────────────────────────────────

FAITHFULNESS = """You verify factual grounding. Break the ANSWER into atomic factual \
claims (ignore citations like [Source: ...]). For each claim decide if it is supported \
by the CONTEXT. Score = supported_claims / total_claims (1.0 if no factual claims).

Return JSON: {{"total": int, "supported": int, "score": float, "rationale": "<=25 words"}}

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEVANCY = """Rate how directly the ANSWER addresses the QUESTION, ignoring length and \
style. 1.0 = fully on-point, 0.0 = irrelevant.

Return JSON: {{"score": float, "rationale": "<=20 words"}}

QUESTION: {question}
ANSWER: {answer}"""

CORRECTNESS = """Compare the ANSWER to the REFERENCE for factual agreement (semantic, not \
wording). 1.0 = same facts, 0.5 = partially correct, 0.0 = wrong/contradictory.

Return JSON: {{"score": float, "rationale": "<=20 words"}}

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}"""


def judge_faithfulness(answer: str, contexts: list[dict], model: str) -> dict:
    r = _judge(FAITHFULNESS.format(context=_ctx_text(contexts), answer=answer), model)
    return {"score": _score(r), "rationale": r.get("rationale", "")}


def judge_answer_relevancy(question: str, answer: str, model: str) -> dict:
    r = _judge(RELEVANCY.format(question=question, answer=answer), model)
    return {"score": _score(r), "rationale": r.get("rationale", "")}


def judge_answer_correctness(question: str, answer: str, reference: str, model: str) -> dict:
    r = _judge(CORRECTNESS.format(question=question, reference=reference, answer=answer), model)
    return {"score": _score(r), "rationale": r.get("rationale", "")}


# ── Retrieval metrics ───────────────────────────────────────

CTX_PRECISION = """For each retrieved CONTEXT passage, decide if it is relevant to \
answering the QUESTION. Score = relevant_passages / total_passages.

Return JSON: {{"total": int, "relevant": int, "score": float}}

QUESTION: {question}

PASSAGES:
{passages}"""

CTX_RECALL = """Can every fact in the REFERENCE answer be found in the CONTEXT? Break the \
reference into facts and compute supported / total.

Return JSON: {{"total": int, "supported": int, "score": float}}

REFERENCE: {reference}

CONTEXT:
{context}"""


def judge_context_precision(question: str, contexts: list[dict], model: str) -> dict:
    if not contexts:
        return {"score": 0.0}  # nothing retrieved is a genuine precision miss, not a judge failure
    passages = "\n\n".join(f"[{i+1}] {c.get('content','')}" for i, c in enumerate(contexts))
    r = _judge(CTX_PRECISION.format(question=question, passages=passages), model)
    return {"score": _score(r)}


def judge_context_recall(reference: str, contexts: list[dict], model: str) -> dict:
    if not contexts:
        return {"score": 0.0}
    r = _judge(CTX_RECALL.format(reference=reference, context=_ctx_text(contexts)), model)
    return {"score": _score(r)}


# ── Behaviour metrics ───────────────────────────────────────

CITATION = """Each cited source should support the claim it is attached to. Given the \
ANSWER (with inline [Source: ...] citations) and the CONTEXT passages, score the fraction \
of citations whose claim is actually supported by the cited material. 1.0 if no citations.

Return JSON: {{"citations": int, "valid": int, "score": float}}

CONTEXT:
{context}

ANSWER:
{answer}"""


def has_citation(answer: str) -> bool:
    return "[source:" in (answer or "").lower()


def judge_citation_accuracy(answer: str, contexts: list[dict], model: str) -> dict:
    # Caller decides what a *missing* citation means (see run_benchmark); here we
    # only score the validity of citations that are actually present.
    if not has_citation(answer):
        return {"score": None, "note": "no citations"}
    r = _judge(CITATION.format(context=_ctx_text(contexts), answer=answer), model)
    return {"score": _score(r)}


def judge_refusal_correctness(answer: str, answerable: bool) -> dict:
    """Deterministic: did the system refuse exactly when it should?

    Unanswerable / false-premise questions SHOULD trigger a refusal; answerable
    ones should NOT (penalises over-refusal). No LLM call needed.
    """
    refused = is_refusal(answer)
    correct = (refused and not answerable) or (not refused and answerable)
    return {"score": 1.0 if correct else 0.0, "refused": refused}


def _score(r: dict):
    """Clamp a judge's score to [0,1], or None if the judge failed/omitted it.

    None is propagated so a failed judge call is EXCLUDED from aggregation rather
    than silently counted as a perfect (or zero) score.
    """
    if not isinstance(r, dict) or "score" not in r:
        return None
    try:
        return max(0.0, min(1.0, float(r["score"])))
    except (TypeError, ValueError):
        return None
