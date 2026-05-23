"""NLI-based hallucination verifier — checks each claim against source context."""

import re
from dataclasses import dataclass

from sentence_transformers import CrossEncoder


_nli_model: CrossEncoder | None = None
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Labels: 0=contradiction, 1=entailment, 2=neutral
LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


@dataclass
class VerificationResult:
    original_answer: str
    verified_answer: str
    claims: list[dict]  # {text, label, score, stripped}
    claims_total: int
    claims_verified: int
    claims_stripped: int


def get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(NLI_MODEL)
    return _nli_model


def verify_answer(answer: str, context_chunks: list[dict], threshold: float = 0.7) -> VerificationResult:
    """Verify each claim in the answer against the source context.

    Claims that contradict or aren't supported by the context are stripped.
    """
    claims = _extract_claims(answer)

    if not claims:
        return VerificationResult(
            original_answer=answer,
            verified_answer=answer,
            claims=[],
            claims_total=0,
            claims_verified=0,
            claims_stripped=0,
        )

    context_text = "\n".join(c["content"] for c in context_chunks)
    model = get_nli_model()

    verified_claims = []
    stripped_count = 0

    for claim_text in claims:
        # NLI: premise=context, hypothesis=claim
        scores = model.predict([(context_text, claim_text)])
        score = scores[0] if isinstance(scores[0], float) else scores[0].tolist()

        # For models that return logits per class
        if hasattr(score, "__len__") and len(score) == 3:
            label_idx = int(max(range(3), key=lambda i: score[i]))
            confidence = float(score[label_idx])
            label = LABEL_MAP[label_idx]
        else:
            # Binary: positive = entailment
            label = "entailment" if float(score) > 0.5 else "contradiction"
            confidence = abs(float(score))

        is_stripped = label in ("contradiction", "neutral") and confidence > threshold

        verified_claims.append({
            "text": claim_text,
            "label": label,
            "score": round(confidence, 3),
            "stripped": is_stripped,
        })

        if is_stripped:
            stripped_count += 1

    # Rebuild answer without stripped claims
    verified_answer = answer
    if stripped_count > 0:
        for claim_info in verified_claims:
            if claim_info["stripped"]:
                verified_answer = verified_answer.replace(claim_info["text"], "")

        # Clean up whitespace
        verified_answer = re.sub(r"\n{3,}", "\n\n", verified_answer).strip()
        verified_answer += "\n\n_[Some information was removed due to insufficient evidence in the source documents.]_"

    return VerificationResult(
        original_answer=answer,
        verified_answer=verified_answer,
        claims=verified_claims,
        claims_total=len(claims),
        claims_verified=len(claims) - stripped_count,
        claims_stripped=stripped_count,
    )


def _extract_claims(answer: str) -> list[str]:
    """Split answer into individual claims (sentences)."""
    # Remove citation markers for claim extraction
    clean = re.sub(r"\[Source:[^\]]*\]", "", answer)

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", clean)

    claims = []
    for s in sentences:
        s = s.strip()
        # Skip meta-sentences like "I couldn't find..."
        if s and len(s) > 20 and not s.lower().startswith("i couldn't find") and not s.lower().startswith("please check"):
            claims.append(s)

    return claims
