#!/bin/bash
# docker-build.sh — Build and push DocVault Docker image
# No torch — uses OpenAI embeddings + ONNX Runtime

set -e

VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
REGISTRY="${REGISTRY:-ghcr.io/oyaah}"
IMAGE="${REGISTRY}/docvault"

echo "Building docvault (no torch, ONNX-only)"
echo "  VCS ref: $VCS_REF"

# Ensure ONNX models exist
if [ ! -f models/reranker.onnx ] || [ ! -f models/verifier.onnx ]; then
  echo "ONNX models not found. Exporting..."
  pip install -e ".[export]" 2>/dev/null
  python scripts/export_onnx.py
fi

# Build
docker build \
  --label "org.opencontainers.image.revision=$VCS_REF" \
  -t "${IMAGE}:${VCS_REF}" \
  -t "${IMAGE}:latest" \
  .

echo "Built: ${IMAGE}:${VCS_REF}"

if [ "$1" = "--push" ]; then
  docker push "${IMAGE}:${VCS_REF}"
  docker push "${IMAGE}:latest"
  echo "Pushed to ${REGISTRY}"
fi
