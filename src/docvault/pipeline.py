"""Core pipeline — orchestrates ingest and query flows."""

import time
import logging

import numpy as np

from docvault.config import settings
from docvault.storage.documents import DocumentStore
from docvault.ingest.parser import parse_file
from docvault.ingest.chunker import chunk_document
from docvault.ingest.embedder import embed_texts, embed_query
from docvault.retrieval import dense, sparse
from docvault.retrieval.fusion import reciprocal_rank_fusion
from docvault.retrieval.reranker import rerank
from docvault.generation.generator import generate_answer, expand_query
from docvault.generation.verifier import verify_answer
from docvault.memory.session import SessionStore
from docvault.memory.cache import RetrievalCache
from docvault.ragops.tracer import QueryTrace, RetrievalTrace, GenerationTrace, VerificationTrace
from docvault.ragops.drift import compute_embedding_drift
from docvault.ragops import metrics

from pathlib import Path

logger = logging.getLogger(__name__)


class DocVaultPipeline:
    def __init__(self, store: DocumentStore | None = None):
        self.store = store or DocumentStore()
        self.sessions = SessionStore()
        self.cache = RetrievalCache()

    # ── Ingest ──────────────────────────────────────────────

    def ingest_file(
        self,
        file_path: Path,
        version: str = "1.0",
        effective_date: str | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """Ingest a single file into the pipeline."""
        t0 = time.time()

        # Parse
        parsed = parse_file(file_path)

        # Register document
        doc = self.store.add_document(
            title=parsed.title,
            source_path=str(file_path),
            version=version,
            effective_date=effective_date,
            doc_type=doc_type,
        )

        # Chunk
        chunks = chunk_document(parsed, doc.id)
        if not chunks:
            return {"document_id": doc.id, "chunks": 0, "time_ms": 0}

        # Embed
        texts = [c["content"] for c in chunks]
        hashes = [c["chunk_hash"] for c in chunks]
        embeddings = embed_texts(texts, chunk_hashes=hashes)

        # Embedding drift detection
        drift = compute_embedding_drift(embeddings)
        if drift is not None:
            logger.info(f"Embedding drift: {drift:.6f}")

        # Store chunks in SQLite (with parent linking)
        chunk_ids = self.store.add_chunks(chunks)

        # Get chunk IDs from DB for dense index
        db_chunks = self.store.get_chunks_by_document(doc.id)
        db_chunk_ids = [c["id"] for c in db_chunks]
        chunk_meta = [
            {"document_id": doc.id, "section_path": c["section_path"]}
            for c in db_chunks
        ]

        # Index in LanceDB
        dense.index_chunks(db_chunk_ids, embeddings, metadata=chunk_meta)

        # Invalidate retrieval cache
        self.cache.invalidate()

        # Update metrics
        stats = self.store.stats()
        metrics.ACTIVE_DOCUMENTS.set(stats["active_documents"])
        metrics.ACTIVE_CHUNKS.set(stats["active_chunks"])
        metrics.INGEST_TOTAL.inc()

        elapsed = (time.time() - t0) * 1000
        logger.info(f"Ingested {parsed.title}: {len(chunks)} chunks in {elapsed:.0f}ms")

        result = {
            "document_id": doc.id,
            "title": parsed.title,
            "chunks": len(chunks),
            "time_ms": round(elapsed),
        }
        if drift is not None:
            result["embedding_drift"] = drift

        return result

    def ingest_directory(
        self,
        dir_path: Path,
        version: str = "1.0",
        doc_type: str | None = None,
    ) -> list[dict]:
        """Ingest all supported files from a directory."""
        results = []
        extensions = {".md", ".txt", ".pdf", ".html", ".htm", ".docx"}

        for fp in sorted(dir_path.iterdir()):
            if fp.is_file() and fp.suffix.lower() in extensions:
                try:
                    result = self.ingest_file(fp, version=version, doc_type=doc_type)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to ingest {fp}: {e}")
                    results.append({"file": str(fp), "error": str(e)})

        return results

    # ── Query ───────────────────────────────────────────────

    def query(
        self,
        question: str,
        session_id: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        """Full query pipeline: retrieve → rerank → generate → verify → cite."""
        trace = QueryTrace(query=question, session_id=session_id)
        t_total = time.time()

        # Session memory
        history = []
        if session_id:
            self.sessions.get_or_create_session(session_id)
            history = self.sessions.get_history(session_id)

        # Query expansion for short queries
        queries = expand_query(question)
        trace.expanded_queries = queries

        # ── Retrieval ──
        t_ret = time.time()

        query_emb = embed_query(question)

        # Check cache
        cached = self.cache.get(query_emb, settings.rerank_top_k)
        if cached is not None:
            metrics.CACHE_HITS.inc()
            trace.retrieval.cache_hit = True
            reranked = cached
        else:
            metrics.CACHE_MISSES.inc()

            # Dense retrieval
            dense_results = dense.search(query_emb, top_k=settings.dense_top_k)
            trace.retrieval.dense_results = len(dense_results)

            # Sparse retrieval (BM25)
            sparse_results = sparse.search(question, self.store, top_k=settings.sparse_top_k)
            trace.retrieval.bm25_results = len(sparse_results)

            # Enrich dense results with content from DB
            for r in dense_results:
                chunk = self.store.get_chunk_by_id(r["chunk_id"])
                if chunk:
                    r["content"] = chunk["content"]
                    r["section_path"] = chunk.get("section_path", "")
                    r["document_id"] = chunk["document_id"]

            # Filter out results without content
            dense_results = [r for r in dense_results if "content" in r]

            # Fusion
            fused = reciprocal_rank_fusion(dense_results, sparse_results)
            trace.retrieval.fused_results = len(fused)

            # Rerank
            reranked = rerank(question, fused, top_k=settings.rerank_top_k)

            # Cache results
            self.cache.put(query_emb, settings.rerank_top_k, reranked)

        trace.retrieval.reranked_top_scores = [
            round(r.get("rerank_score", 0), 4) for r in reranked
        ]
        trace.retrieval.latency_ms = (time.time() - t_ret) * 1000

        # ── Confidence gate ──
        top_score = reranked[0]["rerank_score"] if reranked else 0
        metrics.CONFIDENCE_DIST.observe(top_score)

        if not reranked or top_score < settings.confidence_threshold:
            trace.confidence = "low"
            no_answer = (
                "I couldn't find this in our policies. "
                "Please check with the relevant team for the most current information."
            )
            trace.total_latency_ms = (time.time() - t_total) * 1000
            trace.save()
            metrics.QUERY_TOTAL.inc()

            if session_id:
                self.sessions.add_turn(session_id, question, no_answer, confidence="low")

            return {
                "answer": no_answer,
                "citations": [],
                "confidence": "low",
                "trace_id": trace.trace_id,
            }

        # ── Enrich chunks with document metadata ──
        context_chunks = []
        for chunk in reranked:
            doc = self.store.get_document(chunk.get("document_id", ""))
            enriched = {
                **chunk,
                "doc_title": doc.title if doc else "Unknown",
                "doc_version": doc.version if doc else "",
            }
            context_chunks.append(enriched)

            # Also fetch parent chunk for broader context
            parent = self.store.get_parent_chunk(chunk.get("chunk_id", ""))
            if parent and parent["id"] != chunk.get("chunk_id"):
                context_chunks.append({
                    "content": parent["content"],
                    "section_path": parent.get("section_path", ""),
                    "doc_title": doc.title if doc else "Unknown",
                    "doc_version": doc.version if doc else "",
                    "is_parent": True,
                })

        # Deduplicate by content prefix
        seen = set()
        unique_chunks = []
        for c in context_chunks:
            key = c["content"][:100]
            if key not in seen:
                seen.add(key)
                unique_chunks.append(c)
        context_chunks = unique_chunks

        # ── Generation ──
        t_gen = time.time()
        gen_result = generate_answer(question, context_chunks, history)
        trace.generation = GenerationTrace(
            model=gen_result["model"],
            context_tokens=gen_result["usage"]["prompt_tokens"],
            output_tokens=gen_result["usage"]["completion_tokens"],
            latency_ms=(time.time() - t_gen) * 1000,
        )

        # ── Verification ──
        verification = verify_answer(gen_result["answer"], context_chunks)
        trace.verification = VerificationTrace(
            claims_total=verification.claims_total,
            claims_verified=verification.claims_verified,
            claims_stripped=verification.claims_stripped,
            nli_scores=[c["score"] for c in verification.claims],
        )

        for score in trace.verification.nli_scores:
            metrics.NLI_SCORES.observe(score)

        if verification.claims_total > 0:
            rate = verification.claims_stripped / verification.claims_total
            metrics.HALLUCINATION_RATE.set(rate)

        # ── Final response ──
        confidence = "high" if top_score > 0.6 else "medium"
        trace.confidence = confidence
        trace.total_latency_ms = (time.time() - t_total) * 1000
        trace.save()

        metrics.QUERY_TOTAL.inc()
        metrics.QUERY_LATENCY.labels(stage="total").observe(trace.total_latency_ms / 1000)
        metrics.QUERY_LATENCY.labels(stage="retrieval").observe(trace.retrieval.latency_ms / 1000)
        metrics.QUERY_LATENCY.labels(stage="generation").observe(trace.generation.latency_ms / 1000)

        answer = verification.verified_answer
        citations = gen_result["citations"]

        if session_id:
            self.sessions.add_turn(
                session_id, question, answer,
                citations=citations, confidence=confidence, trace_id=trace.trace_id,
            )

        # Collect actual section_paths from retrieved chunks for eval
        retrieved_sections = list({
            c.get("section_path", "") for c in context_chunks
            if c.get("section_path") and not c.get("is_parent")
        })

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_sections": retrieved_sections,
            "confidence": confidence,
            "verification": {
                "claims_total": verification.claims_total,
                "claims_verified": verification.claims_verified,
                "claims_stripped": verification.claims_stripped,
            },
            "trace_id": trace.trace_id,
        }
