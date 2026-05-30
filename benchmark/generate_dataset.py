"""Generate a benchmark question set grounded in the INGESTED corpus.

No golden/exact-match targets and no dataset-specific hardcoding: questions and
their semantic reference answers are synthesised from the chunks actually stored
in DocVault, so the set tracks whatever corpus you ingested.

Categories (the stress axes from the benchmark plan):
    factoid        — single-fact lookup from one chunk
    multi_hop      — needs two chunks from the same document
    comparative    — compare two facts/sections
    paraphrase     — factoid phrased without corpus vocabulary (tests dense recall)
    table_lookup   — answer lives inside a markdown table
    unanswerable   — plausible but out-of-corpus; correct behaviour is to refuse
    false_premise  — embeds an assumption the corpus contradicts

Output: JSONL, one object per line:
    {id, category, question, reference_answer, answerable,
     grounded_doc_title, grounded_section, source_chunk_ids}

Reference answers are used ONLY by the LLM judge (semantic), never substring-matched.
"""

import argparse
import json
import random
import re
from pathlib import Path

import litellm

from docvault.config import settings
from docvault.storage.documents import DocumentStore

# Share of total questions per category (must sum to ~1.0)
CATEGORY_MIX = {
    "factoid": 0.26,
    "multi_hop": 0.13,
    "comparative": 0.11,
    "paraphrase": 0.16,
    "table_lookup": 0.12,
    "unanswerable": 0.12,
    "false_premise": 0.10,
}

GROUNDED_PROMPT = """You write evaluation questions for a company-policy Q&A system.

Below are excerpt(s) from real policy documents. Write ONE realistic "{category}" \
question an employee might ask, answerable SOLELY from the excerpt(s), plus a \
concise reference answer grounded only in them.

The question MUST be self-contained: name the specific subject (act, policy, or \
topic) explicitly. NEVER refer to "the excerpt", "the passage", "the document", \
"above", or "mentioned here" — the system answering it cannot see this excerpt and \
must be able to retrieve the right document from the question alone.

Rules for category "{category}": {rule}

Return STRICT JSON: {{"question": "...", "reference_answer": "..."}}

EXCERPT(S):
{passages}"""

UNANSWERABLE_PROMPT = """You write evaluation questions for a company-policy Q&A system \
whose corpus covers these documents:
{titles}

Write ONE realistic question that is plausibly in-domain but CANNOT be answered \
from such a corpus (asks for a fact that would not appear in these policies). \
The correct system behaviour is to refuse / say it doesn't know.

Return STRICT JSON: {{"question": "..."}}"""

FALSE_PREMISE_PROMPT = """Below is an excerpt from a policy document. Write ONE question \
that embeds a FALSE assumption contradicting the excerpt (e.g. states a wrong number \
or rule as if true, then asks a follow-up). A correct system should reject the premise.

Return STRICT JSON: {{"question": "...", "false_premise": "...", "correction": "..."}}

EXCERPT:
{passages}"""

RULES = {
    "factoid": "ask for one specific fact (a number, date, name, or rule).",
    "multi_hop": "require combining facts from BOTH excerpts to answer.",
    "comparative": "ask to compare or contrast two facts present in the excerpt(s).",
    "paraphrase": "use everyday wording, avoiding the exact terms in the excerpt.",
    "table_lookup": "ask for a value that appears inside the table.",
}


def llm_json(prompt: str, model: str, retries: int = 3) -> dict | None:
    for _ in range(retries):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
                response_format={"type": "json_object"},
                num_retries=4,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            continue
    return None


def strip_breadcrumb(content: str, section_path: str) -> str:
    """Chunks are stored with a breadcrumb prefix; drop it for display."""
    if section_path and content.startswith(section_path):
        return content[len(section_path):].lstrip("\n")
    return content


def passage(chunk: dict) -> str:
    body = strip_breadcrumb(chunk["content"], chunk.get("section_path", ""))
    return f"[{chunk['doc_title']} — {chunk.get('section_path','')}]\n{body}"


def counts_for(n: int, mix: dict | None = None) -> dict:
    mix = mix or CATEGORY_MIX
    out = {k: max(1, round(n * frac)) for k, frac in mix.items()}
    # adjust rounding drift to hit n exactly
    diff = n - sum(out.values())
    out["factoid"] += diff
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate the benchmark question set.")
    ap.add_argument("-n", "--num", type=int, default=150)
    ap.add_argument("--gen-model", default=settings.llm_model)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "questions.jsonl"))
    args = ap.parse_args()

    random.seed(args.seed)
    store = DocumentStore()
    chunks = store.get_active_leaf_chunks()
    if len(chunks) < 10:
        raise SystemExit(
            f"Only {len(chunks)} leaf chunks found. Ingest the corpus first "
            f"(see benchmark/README.md)."
        )

    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["document_id"], []).append(c)
    table_chunks = [c for c in chunks if "|" in c["content"]]
    titles = sorted({c["doc_title"] for c in chunks})

    mix = dict(CATEGORY_MIX)
    if not table_chunks:
        # corpus has no tables (e.g. billsum) — drop the category and renormalise
        # rather than silently emit zero of it.
        print("WARNING: no tables in corpus -> dropping 'table_lookup' category.")
        share = mix.pop("table_lookup")
        for k in mix:
            mix[k] += share / len(mix)

    targets = counts_for(args.num, mix)
    print(f"Generating {args.num} questions with {args.gen_model}: {targets}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    qid = 0

    def emit(category, question, reference, answerable, src_chunks, extra=None):
        nonlocal qid
        rec = {
            "id": f"q{qid:04d}",
            "category": category,
            "question": question,
            "reference_answer": reference,
            "answerable": answerable,
            "grounded_doc_title": src_chunks[0]["doc_title"] if src_chunks else None,
            "grounded_section": src_chunks[0].get("section_path") if src_chunks else None,
            "source_chunk_ids": [c["id"] for c in src_chunks],
        }
        if extra:
            rec.update(extra)
        rows.append(rec)
        qid += 1

    for category, count in targets.items():
        made = 0
        attempts = 0
        while made < count and attempts < count * 5:
            attempts += 1

            if category == "unanswerable":
                data = llm_json(UNANSWERABLE_PROMPT.format(titles="\n".join(f"- {t}" for t in titles)), args.gen_model)
                if data and data.get("question"):
                    emit(category, data["question"], "", False, [], {"expected_behavior": "refuse"})
                    made += 1
                continue

            if category == "false_premise":
                c = random.choice(chunks)
                data = llm_json(FALSE_PREMISE_PROMPT.format(passages=passage(c)), args.gen_model)
                if data and data.get("question"):
                    emit(category, data["question"], data.get("correction", ""), False, [c],
                         {"expected_behavior": "reject_premise", "false_premise": data.get("false_premise", "")})
                    made += 1
                continue

            # grounded categories
            if category == "table_lookup":
                if not table_chunks:
                    break
                sample = [random.choice(table_chunks)]
            elif category in ("multi_hop", "comparative"):
                doc = random.choice([v for v in by_doc.values() if len(v) >= 2] or list(by_doc.values()))
                sample = random.sample(doc, min(2, len(doc)))
            else:
                sample = [random.choice(chunks)]

            data = llm_json(
                GROUNDED_PROMPT.format(
                    category=category,
                    rule=RULES.get(category, ""),
                    passages="\n\n---\n\n".join(passage(c) for c in sample),
                ),
                args.gen_model,
            )
            if data and data.get("question") and data.get("reference_answer"):
                emit(category, data["question"], data["reference_answer"], True, sample)
                made += 1

        print(f"  {category}: {made}/{count}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} questions to {out_path}")


if __name__ == "__main__":
    main()
