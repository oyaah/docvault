from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_prefix": "DOCVAULT_"}

    # Paths
    data_dir: Path = Path("./data")
    corpus_dir: Path = Path("./corpus")

    # LLM
    llm_model: str = "gpt-4o-mini"

    # Embedding backend: "openai" (cloud API) or "local" (sentence-transformers)
    embedding_backend: str = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536  # OpenAI text-embedding-3-small
    local_embedding_model: str = "BAAI/bge-m3"

    # Chunking
    chunk_size: int = 400  # tokens
    chunk_overlap_pct: float = 0.1

    # Retrieval
    fusion_alpha: float = 0.6  # weight for dense vs sparse
    dense_top_k: int = 50
    sparse_top_k: int = 50
    fused_top_k: int = 20
    rerank_top_k: int = 5
    context_budget_tokens: int = 3000

    # Pinecone (managed vector DB)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "docvault"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Confidence
    confidence_threshold: float = 0.0  # raw ms-marco score; >0 = relevant

    # Memory
    session_ttl_seconds: int = 1800  # 30 min
    max_session_turns: int = 5
    cache_ttl_seconds: int = 3600  # 1 hour

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Observability
    trace_log_path: Path = Field(default=Path("./data/traces.jsonl"))
    metrics_enabled: bool = True
    otel_exporter_endpoint: str = ""  # OTLP endpoint, empty = stdout

    # Drift
    embedding_drift_threshold: float = 0.01  # alert if drift > this

    # ONNX models directory (for reranker + verifier)
    onnx_models_dir: Path = Path("./models")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "docvault.db"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
