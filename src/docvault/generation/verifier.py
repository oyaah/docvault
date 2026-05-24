"""NLI-based hallucination verifier — ONNX Runtime inference (no torch needed).

Checks each claim against source context. Claims with confident
contradiction scores are stripped from the answer.

Uses pre-exported ONNX model for nli-deberta-v3-small.
Export with: python scripts/export_onnx.py
"""

import re
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from docvault.config import settings

logger = logging.getLogger(__name__)

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
ONNX_FILE = "verifier.onnx"

# Labels: 0=contradiction, 1=entailment, 2=neutral
LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}

_session: ort.InferenceSession | None = None
_tokenizer = None


@dataclass
class VerificationResult:
    original_answer: str
    verified_answer: str
    claims: list[dict]  # {text, label, score, stripped}
    claims_total: int
    claims_verified: int
    claims_stripped: int


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        model_path = settings.onnx_models_dir / ONNX_FILE
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX verifier model not found at {model_path}. "
                f"Run: python scripts/export_onnx.py"
            )
        _session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"Loaded ONNX verifier from {model_path}")
    return _session


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    return _tokenizer


def verify_answer(answer: str, context_chunks: list[dict], threshold: float = 0.7) -> VerificationResult:
    """Verify each claim in the answer against the source context.

    Claims that contradict the context are stripped.
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
    session = _get_session()
    tokenizer = _get_tokenizer()

    verified_claims = []
    stripped_count = 0

    for claim_text in claims:
        # NLI: premise=context, hypothesis=claim
        inputs = tokenizer(
            [context_text], [claim_text],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_names = {inp.name for inp in session.get_inputs()}
        feed = {k: v for k, v in inputs.items() if k in input_names}
        outputs = session.run(None, feed)
        logits = outputs[0][0]

        # Convert raw logits to probabilities via softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()

        label_idx = int(np.argmax(probs))
        label = LABEL_MAP[label_idx]
        entailment_prob = float(probs[1])
        contradiction_prob = float(probs[0])

        # Strip only when contradiction is confident — neutral means uncertain, not wrong
        is_stripped = contradiction_prob > threshold

        verified_claims.append({
            "text": claim_text,
            "label": label,
            "score": round(entailment_prob, 3),
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


def warmup():
    """Pre-load model and tokenizer."""
    _get_session()
    _get_tokenizer()
    logger.info("ONNX verifier warmed up")


def _extract_claims(answer: str) -> list[str]:
    """Split answer into individual claims (sentences)."""
    clean = re.sub(r"\[Source:[^\]]*\]", "", answer)
    sentences = re.split(r"(?<=[.!?])\s+", clean)

    claims = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 20 and not s.lower().startswith("i couldn't find") and not s.lower().startswith("please check"):
            claims.append(s)

    return claims
