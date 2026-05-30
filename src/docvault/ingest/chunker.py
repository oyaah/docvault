"""Hierarchical chunker — structure-aware, table-safe, parent/child.

Design (see SPEC §1 Chunker + benchmark findings):
  - Leaf chunks: granular retrievable units (~chunk_size tokens). These are the
    only chunks that get embedded/indexed. Each is prefixed with its section
    breadcrumb so dense + BM25 retrieval see the document context ("contextual
    retrieval"), which is the single biggest cheap recall win for short policy
    clauses.
  - Parent chunks: an aggregation of all leaves under the same h(n-1) section.
    Stored but NOT indexed — fetched only at query time via parent_chunk_id to
    give the generator broader context (parent-document retrieval pattern).
  - Tables are never split mid-row: a markdown table is an atomic block.

Chunk dicts carry metadata JSON with {"role": "leaf"|"parent"}. The pipeline
keys off this to decide what to embed; storage keys off it to exclude parents
from BM25.
"""

import json

import xxhash
import tiktoken

from docvault.config import settings
from docvault.ingest.parser import ParsedDocument


_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def chunk_document(doc: ParsedDocument, document_id: str) -> list[dict]:
    """Chunk a parsed document into leaf chunks plus optional parent chunks.

    Returns a list of chunk dicts ready for DocumentStore.add_chunks().
    Parent chunks (if enabled) carry role="parent" and are referenced by their
    children via the `_parent_index` resolved during insertion.
    """
    leaves: list[dict] = []
    # parent_path -> list of leaf positions (indices into `leaves`)
    groups: dict[str, list[int]] = {}
    # parent_path -> aggregated source text for the parent chunk
    group_text: dict[str, list[str]] = {}

    for section in doc.sections:
        body = section.content.strip()
        if not body:
            continue

        parent_path = _parent_path(section.path)

        for piece in _split_blocks(body, settings.chunk_size, settings.chunk_overlap_pct):
            leaves.append(
                _make_chunk(
                    content=_with_breadcrumb(section.path, piece),
                    document_id=document_id,
                    section_path=section.path,
                    chunk_index=0,  # assigned after ordering
                    role="leaf",
                )
            )
            groups.setdefault(parent_path, []).append(len(leaves) - 1)

        group_text.setdefault(parent_path, []).append(f"## {section.path}\n{body}")

    chunks = list(leaves)

    if settings.enable_parent_chunks:
        _attach_parents(chunks, groups, group_text, document_id)

    # Assign sequential chunk_index in final list order
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    return chunks


# ── Parent chunks ───────────────────────────────────────────


def _attach_parents(
    chunks: list[dict],
    groups: dict[str, list[int]],
    group_text: dict[str, list[str]],
    document_id: str,
):
    """Create one parent chunk per group with >= 2 leaves and link children to it."""
    for parent_path, leaf_positions in groups.items():
        if len(leaf_positions) < 2:
            continue  # a lone leaf already carries its own context

        body = _truncate_tokens(
            "\n\n".join(group_text[parent_path]), settings.parent_chunk_max_tokens
        )
        parent = _make_chunk(
            content=_with_breadcrumb(parent_path, body),
            document_id=document_id,
            section_path=parent_path,
            chunk_index=0,
            role="parent",
        )
        chunks.append(parent)
        parent_index = len(chunks) - 1
        for pos in leaf_positions:
            chunks[pos]["_parent_index"] = parent_index


def _parent_path(path: str) -> str:
    """Drop the last breadcrumb segment: 'A > B > C' -> 'A > B'. Root stays itself."""
    parts = [p.strip() for p in path.split(">")]
    if len(parts) <= 1:
        return path
    return " > ".join(parts[:-1])


# ── Block-aware splitting ───────────────────────────────────


def _split_blocks(content: str, max_tokens: int, overlap_pct: float) -> list[str]:
    """Pack content into chunks <= max_tokens without breaking tables.

    Blocks are paragraphs (split on blank lines) and table blocks (maximal runs
    of '|' lines). Table blocks are atomic and may exceed max_tokens. Oversized
    paragraph blocks fall back to a token window split.
    """
    blocks = _to_blocks(content)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for block, is_table in blocks:
        bt = count_tokens(block)

        if not is_table and bt > max_tokens:
            # flush, then window-split the oversized paragraph
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_tokens = [], 0
            chunks.extend(_window_split(block, max_tokens, overlap_pct))
            continue

        if cur_tokens + bt > max_tokens and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_tokens = [], 0

        cur.append(block)
        cur_tokens += bt

    if cur:
        chunks.append("\n\n".join(cur))

    return chunks or [content]


def _to_blocks(content: str) -> list[tuple[str, bool]]:
    """Split text into (block, is_table) tuples preserving table integrity."""
    lines = content.split("\n")
    blocks: list[tuple[str, bool]] = []
    buf: list[str] = []
    buf_is_table = False

    def flush():
        nonlocal buf, buf_is_table
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append((text, buf_is_table))
        buf, buf_is_table = [], False

    for line in lines:
        is_table_line = line.lstrip().startswith("|")
        if not line.strip():
            flush()
            continue
        if buf and is_table_line != buf_is_table:
            flush()
        buf.append(line)
        buf_is_table = is_table_line

    flush()
    return blocks


def _window_split(text: str, max_tokens: int, overlap_pct: float) -> list[str]:
    enc = _get_encoder()
    tokens = enc.encode(text)
    overlap = int(max_tokens * overlap_pct)
    out: list[str] = []
    pos = 0
    while pos < len(tokens):
        end = min(pos + max_tokens, len(tokens))
        out.append(enc.decode(tokens[pos:end]))
        if end >= len(tokens):
            break
        pos = end - overlap
    return out


def _truncate_tokens(text: str, max_tokens: int) -> str:
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# ── Helpers ─────────────────────────────────────────────────


def _with_breadcrumb(section_path: str, body: str) -> str:
    if settings.breadcrumb_prefix and section_path:
        return f"{section_path}\n{body}"
    return body


def _make_chunk(
    content: str,
    document_id: str,
    section_path: str,
    chunk_index: int,
    role: str,
) -> dict:
    chunk_hash = xxhash.xxh64(content.encode()).hexdigest()
    return {
        "document_id": document_id,
        "parent_chunk_id": None,  # resolved from _parent_index at insertion
        "content": content,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "chunk_hash": chunk_hash,
        "token_count": count_tokens(content),
        "metadata": json.dumps({"role": role}),
        "role": role,  # convenience field, not persisted as a column
    }
