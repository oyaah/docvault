# DocVault Production Deployment Guide

## Build Performance Optimization

### Current Setup (Multi-stage)
- **Build time**: ~5-8 min on ARM64 (torch dominates ~60-70% of time)
- **Image size**: ~2.5GB (torch + dependencies)
- **Cache benefit**: Wheels cached separately from source code

### Quick Local Builds
```bash
# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

# Build with progress output
docker build --progress=plain -t docvault:latest .

# For faster rebuilds, use cache from main image
docker build --cache-from docvault:latest -t docvault:latest .
```

## Production Solutions (Trade-offs)

### Option 1: ONNX Runtime (Recommended for ML Models)
**Best for**: Replacing transformer-based inference with faster execution

```python
# Replace torch + transformers inference
import onnx
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained("model_name", use_cache=False)
# ~30-50% faster inference, smaller model size
```

**Pros**: 
- 30-50% faster inference than PyTorch
- 40-60% smaller model size
- Single-file deployment (ONNX graph)

**Cons**:
- Requires model conversion pipeline
- Limited dynamic shapes support

**Implementation**:
```bash
# Add to pyproject.toml
optimum[onnxruntime]>=1.18
```

### Option 2: Sentence Transformers via API (Cloud-based Embeddings)
**Best for**: Offloading embedding computation

```python
# Instead of local sentence-transformers
import anthropic
import openai

client = openai.OpenAI()
embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="text"
)
```

**Pros**:
- No torch/transformers needed
- Serverless scaling
- Latest models available

**Cons**:
- Network latency
- API costs

### Option 3: Quantized Models (Lightweight Torch)
**Best for**: Keeping torch but reducing model size/speed

```bash
# Use quantized versions
pip install bitsandbytes  # for 8-bit/4-bit quantization

# In code:
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "model_name",
    load_in_8bit=True,  # 4x memory reduction
    device_map="auto"
)
```

**Pros**:
- 4-8x memory reduction
- Minimal accuracy loss
- Still local processing

**Cons**:
- Slightly slower inference
- Torch still required

### Option 4: Model Distillation (Smaller Models)
**Best for**: Replacing heavy models with faster variants

```bash
# Replace sentence-transformers/3 with lightweight version
pip install sentence-transformers==3.0
# Use: "all-MiniLM-L6-v2" instead of "all-mpnet-base-v2"
# ~40x smaller, 5-10% accuracy loss
```

**Build time reduction**: 60-70% (smaller model downloads)

### Option 5: Multi-service Architecture (Production-Grade)
**Best for**: Enterprise deployments

**Architecture**:
```
┌─────────────────────────────────────────┐
│ FastAPI Service (docvault-api)          │ ← No torch, ~300MB
│  - Document ingestion                   │
│  - Query orchestration                  │
│  - API handlers                         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Embedding Service (docvault-embed)      │ ← Torch only, ~2.5GB
│  - sentence-transformers                │
│  - Batch processing                     │
│  - Shared volume cache                  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Reranker Service (docvault-rerank)      │ ← Torch only, ~1.5GB
│  - transformers for cross-encoding      │
│  - Shared model cache                   │
└─────────────────────────────────────────┘
```

**Compose example**:
```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      EMBEDDING_SERVICE_URL: http://embed:9000
      RERANKER_SERVICE_URL: http://rerank:9001
  
  embed:
    build:
      context: .
      dockerfile: Dockerfile.embed
    environment:
      WORKER_CLASS: "models.embeddings"
  
  rerank:
    build:
      context: .
      dockerfile: Dockerfile.rerank
    environment:
      WORKER_CLASS: "models.reranking"
```

**Benefits**:
- Independent scaling
- Smaller image for API layer (~300MB)
- Shared model cache volumes
- Services can be deployed separately

## Kubernetes Deployment (Multi-node)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docvault-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: docvault
        image: docvault:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: docvault
spec:
  selector:
    app: docvault-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Registry Strategy (Private)

```bash
# Build multi-platform
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t my-registry.azurecr.io/docvault:latest \
  --push .

# Layer caching strategy:
# 1. Cache torch layer on build server
# 2. Reuse for all rebuilds
# 3. Only invalidate on torch version change
```

## Local Development with Fast Rebuilds

```bash
# .dockerignore (add)
.git
__pycache__
*.pyc
.pytest_cache
node_modules

# Use bind mounts for code changes
docker compose -f docker-compose.dev.yml up

# Watch for changes (if using watchfiles)
docker compose watch
```

## Summary of Options by Use Case

| Use Case | Solution | Build Time | Image Size | Complexity |
|----------|----------|-----------|-----------|-----------|
| **Local dev** | Current multi-stage | 5-8m | 2.5GB | Low |
| **Cloud API** | ONNX Runtime | 2-3m | 600MB | Medium |
| **Serverless** | Model APIs | <1m | 100MB | High |
| **Enterprise** | Multi-service arch | 2-4m each | 300MB API + 2GB shared | Medium |
| **Quantized** | 8-bit models | 3-5m | 1.2GB | Low |
| **Distilled** | Smaller models | 1-2m | 800MB | Medium |
| **Kubernetes** | Current setup + K8s | 5-8m | 2.5GB | High |

## Recommended Path Forward

1. **Short term**: Use current multi-stage build + Docker BuildKit
2. **Medium term**: Split into separate embedding/reranker services
3. **Long term**: Consider ONNX or quantized models as torch version stabilizes

Deploy with:
```bash
export DOCKER_BUILDKIT=1
docker compose -f docker-compose.yml build
docker compose up -d
```
