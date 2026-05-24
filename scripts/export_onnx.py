#!/usr/bin/env python3
"""Export cross-encoder models to ONNX format.

Run this locally (not in production Docker image) to generate ONNX models:
    pip install docvault[export]
    python scripts/export_onnx.py

Output:
    models/reranker.onnx    (~80MB, ms-marco-MiniLM-L-6-v2)
    models/verifier.onnx    (~140MB, nli-deberta-v3-small)

These ONNX files are used at runtime with onnxruntime (no torch needed).
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODELS = {
    "reranker": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "output": "reranker.onnx",
        "max_length": 512,
    },
    "verifier": {
        "name": "cross-encoder/nli-deberta-v3-small",
        "output": "verifier.onnx",
        "max_length": 512,
    },
}

OUTPUT_DIR = Path(__file__).parent.parent / "models"


def export_model(key: str, info: dict):
    print(f"\n{'='*60}")
    print(f"Exporting {key}: {info['name']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(info["name"])
    model = AutoModelForSequenceClassification.from_pretrained(info["name"])
    model.eval()

    # Create dummy inputs
    dummy = tokenizer(
        "What is the PTO policy?",
        "Employees receive 20 days of paid time off per year.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=info["max_length"],
    )

    output_path = OUTPUT_DIR / info["output"]

    # Use model's forward() parameter order, not tokenizer dict order.
    # BERT forward: (input_ids, attention_mask, token_type_ids)
    # Tokenizer dict order is often: (input_ids, token_type_ids, attention_mask)
    # Mismatch causes silent argument swap in the ONNX graph.
    import inspect
    sig = inspect.signature(model.forward)
    forward_params = [p for p in sig.parameters if p in dummy]
    input_names = forward_params

    dynamic_axes = {name: {0: "batch", 1: "seq_len"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    torch.onnx.export(
        model,
        tuple(dummy[k] for k in input_names),
        str(output_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")

    # Verify with onnxruntime
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    verify_inputs = tokenizer(
        "test query", "test document",
        return_tensors="np", padding=True, truncation=True, max_length=64,
    )
    valid_names = {inp.name for inp in session.get_inputs()}
    feed = {k: v for k, v in verify_inputs.items() if k in valid_names}
    outputs = session.run(None, feed)
    print(f"Verification passed: output shape {outputs[0].shape}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    keys = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for key in keys:
        if key not in MODELS:
            print(f"Unknown model: {key}. Available: {list(MODELS.keys())}")
            continue
        export_model(key, MODELS[key])

    print(f"\nDone. ONNX models saved to {OUTPUT_DIR}/")
    print("These files should be committed to the repo or uploaded to S3.")


if __name__ == "__main__":
    main()
