"""CLI for DocVault — ingest, query, stats, serve."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="DocVault — RAG pipeline for policy Q&A")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    serve_p = sub.add_parser("serve", help="Start the API server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest documents")
    ingest_p.add_argument("path", help="File or directory to ingest")
    ingest_p.add_argument("--version", default="1.0")
    ingest_p.add_argument("--doc-type", default=None)

    # query
    query_p = sub.add_parser("query", help="Query the pipeline")
    query_p.add_argument("question", help="Question to ask")
    query_p.add_argument("--session-id", default=None)

    # stats
    sub.add_parser("stats", help="Show pipeline stats")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "docvault.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "ingest":
        from docvault.pipeline import DocVaultPipeline
        from docvault.config import settings

        settings.ensure_dirs()
        pipe = DocVaultPipeline()
        path = Path(args.path)

        if path.is_dir():
            results = pipe.ingest_directory(path, version=args.version, doc_type=args.doc_type)
        else:
            results = [pipe.ingest_file(path, version=args.version, doc_type=args.doc_type)]

        for r in results:
            if "error" in r:
                print(f"  ERROR: {r['file']}: {r['error']}")
            else:
                print(f"  OK: {r['title']} — {r['chunks']} chunks in {r['time_ms']}ms")

        total = sum(r.get("chunks", 0) for r in results if "error" not in r)
        print(f"\nTotal: {len(results)} docs, {total} chunks")

    elif args.command == "query":
        from docvault.pipeline import DocVaultPipeline
        from docvault.config import settings

        settings.ensure_dirs()
        pipe = DocVaultPipeline()
        result = pipe.query(args.question, session_id=args.session_id)

        print(f"\n{result['answer']}\n")
        print(f"Confidence: {result['confidence']}")
        if result.get("citations"):
            print("\nCitations:")
            for c in result["citations"]:
                print(f"  - {c.get('document', '')} > {c.get('section', '')}")
        if result.get("verification"):
            v = result["verification"]
            print(f"\nVerification: {v['claims_verified']}/{v['claims_total']} claims verified"
                  f" ({v['claims_stripped']} stripped)")

    elif args.command == "stats":
        from docvault.storage.documents import DocumentStore
        from docvault.config import settings

        settings.ensure_dirs()
        store = DocumentStore()
        stats = store.stats()
        print(f"Active documents: {stats['active_documents']}")
        print(f"Active chunks:    {stats['active_chunks']}")

        docs = store.get_active_documents()
        if docs:
            print("\nDocuments:")
            for d in docs:
                print(f"  {d.title} v{d.version} — {d.total_chunks} chunks ({d.ingested_at})")


if __name__ == "__main__":
    main()
