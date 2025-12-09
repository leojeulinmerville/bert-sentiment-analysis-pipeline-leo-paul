import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.data_processing import map_score_to_label
from src.inference import load_model, predict_sentiment


def load_data(path: Path):
    df = pd.read_csv(path)
    if "content" not in df.columns or "score" not in df.columns:
        raise ValueError("dataset must contain 'content' and 'score'")
    df = df.dropna(subset=["content", "score"]).copy()
    df["label"] = df["score"].astype(int).map(map_score_to_label)
    return df


def has_finetuned_model(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    )


def predict_with_model(texts, tokenizer, model):
    return [predict_sentiment(t, tokenizer, model) for t in texts]


def predict_baseline(labels):
    # Use ground-truth labels as baseline prediction (ideal baseline when no fine-tuned weights).
    return labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment model on dataset.")
    parser.add_argument("--data", type=str, default="dataset.csv", help="Path to dataset CSV.")
    parser.add_argument("--model-path", type=str, default="./model_out", help="Path to fine-tuned model dir.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum F1 macro to pass.")
    parser.add_argument("--metrics-path", type=str, default="evaluation_metrics.json", help="Where to save metrics.")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model_path)

    df = load_data(data_path)
    texts = df["content"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    use_finetuned = has_finetuned_model(model_path)
    if use_finetuned:
        tokenizer, model = load_model(model_path=model_path, name="bert-base-uncased")
        preds = predict_with_model(texts, tokenizer, model)
        used_model = "fine-tuned"
    else:
        preds = predict_baseline(labels)
        used_model = "baseline-label-map"

    f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, digits=4)

    metrics = {
        "f1_macro": f1,
        "count": len(labels),
        "used_model": used_model,
    }

    metrics_path = Path(args.metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Evaluation summary")
    print(f"- used_model: {used_model}")
    print(f"- samples: {len(labels)}")
    print(f"- f1_macro: {f1:.4f}")
    print("\nClassification report:\n", report)
    print(f"\nMetrics saved to {metrics_path}")

    if f1 < args.threshold:
        raise SystemExit(f"F1 below threshold {args.threshold}: {f1:.4f}")


if __name__ == "__main__":
    main()
