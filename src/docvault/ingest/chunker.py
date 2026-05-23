"""Hierarchical chunker — respects document structure."""

import xxhash
import tiktoken

from docvault.config import settings
from docvault.ingest.parser import ParsedDocument, Section


_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def chunk_document(doc: ParsedDocument, document_id: str) -> list[dict]:
    """Chunk a parsed document into hierarchical chunks.

    Returns list of chunk dicts ready for DocumentStore.add_chunks().
    """
    chunks = []
    chunk_index = 0

    for section in doc.sections:
        section_chunks = _chunk_section(section, document_id, chunk_index)
        chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    # Link parent chunks — first chunk of each section is the "parent"
    _link_parent_chunks(chunks)

    return chunks


def _chunk_section(section: Section, document_id: str, start_index: int) -> list[dict]:
    """Split a section into chunks respecting token budget."""
    content = section.content
    if not content.strip():
        return []

    token_count = count_tokens(content)
    max_tokens = settings.chunk_size
    overlap_tokens = int(max_tokens * settings.chunk_overlap_pct)

    if token_count <= max_tokens:
        # Section fits in one chunk
        return [
            _make_chunk(
                content=content,
                document_id=document_id,
                section_path=section.path,
                chunk_index=start_index,
                token_count=token_count,
            )
        ]

    # Split into overlapping windows
    enc = _get_encoder()
    tokens = enc.encode(content)
    chunks = []
    pos = 0

    while pos < len(tokens):
        end = min(pos + max_tokens, len(tokens))
        chunk_tokens = tokens[pos:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append(
            _make_chunk(
                content=chunk_text,
                document_id=document_id,
                section_path=section.path,
                chunk_index=start_index + len(chunks),
                token_count=len(chunk_tokens),
            )
        )

        if end >= len(tokens):
            break
        pos = end - overlap_tokens

    return chunks


def _make_chunk(
    content: str,
    document_id: str,
    section_path: str,
    chunk_index: int,
    token_count: int,
) -> dict:
    chunk_hash = xxhash.xxh64(content.encode()).hexdigest()
    return {
        "document_id": document_id,
        "parent_chunk_id": None,  # set later
        "content": content,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "chunk_hash": chunk_hash,
        "token_count": token_count,
        "metadata": "{}",
    }


def _link_parent_chunks(chunks: list[dict]):
    """For multi-chunk sections, first chunk becomes parent of subsequent ones."""
    section_first: dict[str, int] = {}  # section_path -> index of first chunk

    for i, chunk in enumerate(chunks):
        path = chunk["section_path"]
        if path not in section_first:
            section_first[path] = i
        else:
            # This chunk's parent is the first chunk of this section
            # We'll set parent_chunk_id after insertion (need DB IDs)
            # For now, mark with a sentinel
            chunk["_parent_index"] = section_first[path]
