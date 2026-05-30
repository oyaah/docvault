"""Download a real public policy corpus from HuggingFace and write it as
markdown files DocVault can ingest.

Dataset-agnostic: the dataset id and column names are CLI flags, so swapping
corpora never touches pipeline code. Default = billsum (US congressional bills) —
real legislation: long, heavily-structured policy documents with numbered
"SECTION N." clauses, which is exactly what hierarchical chunking targets.
(Must be a script-free / parquet dataset; HF `datasets` 4.x dropped loader scripts.)

Each unique document is de-duplicated, lightly cleaned, and given markdown
headings (detected from ARTICLE/SECTION/numbered-clause patterns) so the
existing markdown parser recovers a real section hierarchy.

Examples:
    # default: 40 US congressional bills
    python benchmark/download_corpus.py --num-docs 40

    # swap to any script-free HF dataset by naming its columns
    python benchmark/download_corpus.py \
        --dataset coastalcph/lex_glue --config scotus \
        --text-col text --title-col label --num-docs 40
"""

import argparse
import re
from pathlib import Path

from datasets import load_dataset


def slugify(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:80] or fallback)


HEADING_PATTERNS = [
    re.compile(r"^\s*(ARTICLE|SECTION|SEC)\.?\s+[IVXLC\d]+", re.IGNORECASE),  # SECTION 1. / SEC. 2.
    re.compile(r"^\s*\d+(\.\d+)*\.?\s+[A-Z]"),          # 1.  /  2.3  Heading
    re.compile(r"^\s*[A-Z][A-Z \-/&]{6,60}$"),           # ALL-CAPS short line
]


def to_markdown(title: str, body: str) -> str:
    """Insert markdown headings using lightweight structural heuristics."""
    lines = body.replace("\r\n", "\n").split("\n")
    out = [f"# {title}".strip(), ""]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if any(p.match(line) for p in HEADING_PATTERNS) and len(stripped) < 90:
            out.append(f"## {stripped}")
        else:
            out.append(stripped)
    # collapse 3+ blank lines
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="Download a policy corpus from HuggingFace.")
    ap.add_argument("--dataset", default="billsum")
    ap.add_argument("--config", default=None, help="dataset config name (e.g. lex_glue 'scotus')")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="text", help="column holding document body")
    ap.add_argument("--title-col", default="title", help="column holding document title")
    ap.add_argument("--num-docs", type=int, default=40)
    ap.add_argument("--min-chars", type=int, default=1500, help="skip tiny documents")
    ap.add_argument("--out", default=str(Path(__file__).parent / "corpus"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} [{args.config or 'default'}/{args.split}] (streaming)…")
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    seen: set[str] = set()
    written = 0
    for i, row in enumerate(ds):
        if written >= args.num_docs:
            break
        body = str(row.get(args.text_col) or "").strip()
        title = str(row.get(args.title_col) or f"document-{i}").strip()
        if len(body) < args.min_chars:
            continue
        key = title or body[:200]
        if key in seen:
            continue
        seen.add(key)

        slug = slugify(title, f"document-{written}")
        path = out_dir / f"{slug}.md"
        path.write_text(to_markdown(title, body), encoding="utf-8")
        written += 1
        print(f"  [{written:>3}] {path.name}  ({len(body):,} chars)")

    print(f"\nWrote {written} documents to {out_dir}")
    if written < args.num_docs:
        print(f"WARNING: only {written}/{args.num_docs} docs met --min-chars; "
              f"lower --min-chars or pick a larger dataset.")


if __name__ == "__main__":
    main()
