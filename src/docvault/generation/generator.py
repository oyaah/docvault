"""LLM generation with constrained prompting and citation extraction."""

import json
import litellm

from docvault.config import settings
from docvault.generation.prompts import SYSTEM_PROMPT, QUERY_TEMPLATE, EXPANSION_PROMPT


def generate_answer(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
) -> dict:
    """Generate a cited answer from retrieved context.

    Returns:
        {answer: str, citations: list[dict], model: str, usage: dict}
    """
    # Format context
    context_str = _format_context(context_chunks)
    history_str = _format_history(history or [])

    prompt = QUERY_TEMPLATE.format(
        context=context_str,
        history=history_str,
        question=query,
    )

    response = litellm.completion(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,  # low temp for factual answers
        max_tokens=1024,
        num_retries=4,  # ride out transient API/TLS errors
    )

    answer = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

    # Extract citations from the answer text
    citations = _extract_citations(answer, context_chunks)

    return {
        "answer": answer,
        "citations": citations,
        "model": settings.llm_model,
        "usage": usage,
    }


def expand_query(query: str) -> list[str]:
    """Generate alternative query phrasings for short queries."""
    if len(query.split()) > 5:
        return [query]

    try:
        response = litellm.completion(
            model=settings.llm_model,
            messages=[
                {"role": "user", "content": EXPANSION_PROMPT.format(query=query)},
            ],
            temperature=0.7,
            max_tokens=100,
        )
        alternatives = response.choices[0].message.content.strip().split("\n")
        return [query] + [a.strip().lstrip("- ").lstrip("0123456789.").strip() for a in alternatives if a.strip()]
    except Exception:
        return [query]


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_title = chunk.get("doc_title", "Unknown Document")
        section = chunk.get("section_path", "")
        version = chunk.get("doc_version", "")
        header = f"[Document {i}: {doc_title}"
        if section:
            header += f", {section}"
        if version:
            header += f", v{version}"
        header += "]"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    parts = []
    for turn in history[-5:]:  # last 5 turns
        parts.append(f"User: {turn.get('question', '')}")
        parts.append(f"Assistant: {turn.get('answer', '')}")
    return "\n".join(parts)


def _extract_citations(answer: str, context_chunks: list[dict]) -> list[dict]:
    """Extract [Source: ...] citations from answer text."""
    import re

    citations = []
    pattern = r"\[Source:\s*([^\]]+)\]"
    matches = re.findall(pattern, answer)

    for match in matches:
        parts = [p.strip() for p in match.split(",")]
        citation = {"text": match}
        if parts:
            citation["document"] = parts[0]
        if len(parts) > 1:
            citation["section"] = parts[1]
        citations.append(citation)

    return citations
