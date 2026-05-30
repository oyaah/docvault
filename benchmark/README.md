# DocVault Benchmark

LLM-as-judge benchmark. No golden/exact-match dataset — retrieval, generation,
and behaviour are all scored semantically by a judge model. Questions are
synthesised from the corpus you actually ingest, so nothing is hardcoded.

## What it measures

| Axis | Metrics |
|------|---------|
| Retrieval | `context_precision`, `context_recall` |
| Generation | `faithfulness` (→ hallucination = 1−faithfulness), `answer_relevancy`, `answer_correctness` |
| Behaviour | `citation_accuracy`, `refusal_correctness` (refuses iff unanswerable) |

Question categories: `factoid`, `multi_hop`, `comparative`, `paraphrase`,
`table_lookup`, `unanswerable`, `false_premise`. Reports per-category means,
bootstrap 95% CIs, and pass/fail gates (faithfulness ≥0.85, hallucination ≤0.05,
refusal ≥0.80 by default).

## Commands (run from the repo root)

```bash
cd /Users/yashbansal/Desktop/docvault

# 0. one-time setup — deps + load secrets (.env is NOT auto-loaded by the app)
pip install -e .
set -a; source .env; set +a          # exports OPENAI_API_KEY + DOCVAULT_PINECONE_* etc.

# 1. RESET the stores — the chunker changed, so the old index/DB are stale
python -c "from docvault.retrieval import dense; dense.reset()"
rm -f data/docvault.db data/docvault.db-wal data/docvault.db-shm

# 2. DOWNLOAD a real public policy corpus (default: 40 CUAD contracts)
python benchmark/download_corpus.py --num-docs 40
#   swap corpus by naming columns, e.g.:
#   python benchmark/download_corpus.py --dataset ClimatePolicyRadar/all-document-text-data \
#       --text-col text --title-col document_name --num-docs 40

# 3. INGEST the corpus through the (fixed) pipeline, then sanity-check
docvault ingest benchmark/corpus
docvault stats

# 4. GENERATE ~150 grounded, categorized questions (uses the generator LLM)
python benchmark/generate_dataset.py -n 150

# 5. RUN the benchmark (LLM-as-judge). Smoke-test 10 first, then the full set:
python benchmark/run_benchmark.py --limit 10
python benchmark/run_benchmark.py --judge-model gpt-4o
```

Outputs:
- `benchmark/data/questions.jsonl` — the generated question set
- `benchmark/data/results.jsonl` — per-question scores (append-only; `--resume` continues)
- `benchmark/data/report.json` — aggregated metrics + gates

## Latest results (40 bills, 128 questions, judge=gpt-4o)

| Metric | Mean | 95% CI |
|--------|------|--------|
| faithfulness | 0.995 | 0.985–1.000 |
| hallucination_rate | 0.005 | — |
| answer_relevancy | 0.833 | 0.763–0.896 |
| answer_correctness | 0.779 | 0.705–0.847 |
| context_recall | 0.840 | 0.776–0.901 |
| context_precision | 0.674 | 0.606–0.743 |
| citation_accuracy | 0.995 | 0.984–1.000 |
| refusal (unanswerable) | 0.818 | — |

Known caveats for this run: `table_lookup` was empty (bills have no tables);
false-premise rejection was weak (0.60); 3/132 generated questions contained
meta-references ("the excerpt") — the generator prompt now forbids these, so a
regenerated set will be cleaner.

## Integrity notes (how we avoid fooling ourselves)

- **Failed judge calls are excluded, never defaulted.** A judge that errors or
  returns malformed JSON yields `None` and is dropped from aggregation (counted in
  `judge_failures`). It is *not* silently scored 1.0 or 0.0.
- **A missing citation on an answerable answer scores 0**, not a free pass.
- **Generator ≠ judge model**, and the judge never sees the gold reference for
  faithfulness/precision (reference-free); only `answer_correctness`/`context_recall`
  are reference-guided.
- The pipeline under test only ever receives the **question** — never the source
  chunk, reference answer, or category.

## Notes

- **Judge ≠ generator.** Generator is `gpt-4o-mini` (config); judge defaults to
  `gpt-4o` to avoid self-enhancement bias. Override with `--judge-model`.
- **Cost.** ~150 questions × (1 query + up to 6 judge calls). Use `--limit` to
  bound spend; `--resume` to recover after an interruption.
- **Tune the confidence gate** (biggest honest win — refusal on unanswerable):
  ```bash
  python benchmark/calibrate_gate.py          # sweeps threshold, cheap (no LLM)
  export DOCVAULT_CONFIDENCE_THRESHOLD=<printed value>
  ```
- **Run fully offline / without Pinecone:**
  ```bash
  export DOCVAULT_DENSE_BACKEND=local DOCVAULT_EMBEDDING_BACKEND=local
  ```
  (LLM generation still needs an API or a local LiteLLM-compatible endpoint.)
- **Ablations.** To compare pipeline variants, change a setting and re-run from
  step 1 against the same `questions.jsonl` (don't regenerate):
  - chunking: `DOCVAULT_ENABLE_PARENT_CHUNKS=false`, `DOCVAULT_BREADCRUMB_PREFIX=false`,
    `DOCVAULT_CHUNK_SIZE=...`
  - gating: `DOCVAULT_CONFIDENCE_THRESHOLD=...`
- The legacy `docvault eval` (substring match over `eval/golden_dataset.json`) is
  superseded by this harness and kept only for reference.
```
