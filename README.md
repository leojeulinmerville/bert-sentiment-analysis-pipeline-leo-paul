# BERT Sentiment Analysis Pipeline

**Authors:**  
- **Léo Merville** (Student 1) — Data Extraction, Data Cleaning, Testing, Documentation  
- **Paul Mouyebissi** (Student 2) — Model Training, Inference, Tokenization Review  

**Course:** Collaborative AI Project (MLOps)  
**Date:** October–November 2025  

---

## Project Overview

This project implements a **complete sentiment analysis pipeline** using **BERT (bert-base-uncased)**.  
It processes textual reviews stored in a CSV file and predicts **sentiment labels**: *negative, neutral, or positive*.  

The work was carried out collaboratively, following real-world **software engineering practices**:  
- Version control & peer reviews  
- Unit testing & integration testing  
- Continuous Integration (CI/CD) via GitHub Actions  
- Agile project management via Trello and Discord

---

## Pipeline Architecture

| Stage | File | Description | Lead |
|--------|------|-------------|------|
| **Data Extraction** | `src/data_extraction.py` | Load raw CSV data, validate columns, handle errors | Léo |
| **Data Processing** | `src/data_processing.py` | Clean text, map sentiment labels, tokenize inputs | Léo & Paul |
| **Model Training** | `src/model.py` | Fine-tune pretrained BERT for sentiment classification | Paul |
| **Inference** | `src/inference.py` | Predict sentiment from new text inputs | Paul |
| **Testing** | `tests/unit/` & `tests/integration/` | Validate components with unit and E2E tests | Both |

---

## Dataset

Reviews are stored in `dataset.csv` with:
- `content`: text of the review  
- `score`: rating (1–5)

Labels mapping:
| Rating | Sentiment | Label |
|---------|-----------|-------|
| 1–2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4–5 | Positive | 2 |

---

## Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.11 |
| **Modeling** | Hugging Face Transformers (`bert-base-uncased`), PyTorch |
| **Data Handling** | pandas, scikit-learn |
| **Testing** | pytest, pytest-cov |
| **CI/CD** | GitHub Actions |

---

## Git Workflow

Feature-branch model with peer review:
| Branch | Purpose | Owner |
|---------|----------|--------|
| `main` | Stable version (merged after review) | Both |
| `feature/data-extraction` | Data loading functions | Léo |
| `feature/data-processing-cleaning` | Text cleaning & dataset split | Léo |
| `feature/data-processing-tokenizer` | Tokenization logic | Léo & Paul |
| `feature/model-training` | Model fine-tuning | Paul |
| `feature/inference` | Inference script | Paul |
| `feature/testing-and-report` | Integration tests & report | Léo |

---

## Testing

Run the full suite:
```bash
pytest -v
```
Coverage target: ~90%.

---

## How to Run (local)

```bash
git clone https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul.git
cd bert-sentiment-analysis-pipeline-leo-paul
pip install -r requirements.txt
pytest -v

python - <<'PY'
from src.inference import load_model, predict_sentiment
tok, mdl = load_model(model_path="./model_out", name="bert-base-uncased")
print(predict_sentiment("I love this app!", tok, mdl))
PY
```

---

## Docker (Part 2)

### Build image
```bash
docker build -t bert-sentiment-cli .
```

### Run CLI inside container
Defaults to base `bert-base-uncased`; mount fine-tuned weights to use custom model.
```bash
docker run --rm bert-sentiment-cli --text "I love this app!"
# with custom weights
docker run --rm -v /path/to/model_out:/app/model_out bert-sentiment-cli --text "I love this app!"
```

### Volumes (C02)
- Persist fine-tuned weights: `-v bert_model_out:/app/model_out` (or bind a host path)
- Persist logs: `-v bert_logs:/app/logs`
- Optional HF cache: `-v bert_hf_cache:/app/.cache/huggingface`

Examples:
```bash
# Named volumes
docker run --rm \
  -v bert_model_out:/app/model_out \
  -v bert_logs:/app/logs \
  bert-sentiment-cli --text "Great product!"

# Bind mount host directories
docker run --rm \
  -v /abs/path/model_out:/app/model_out \
  -v /abs/path/logs:/app/logs \
  bert-sentiment-cli --text "Great product!"
```

### Notes
- Entry point: `python -m src.cli`
- Model cache: `/app/.cache/huggingface` (inside container)
- Fine-tuned model path expected at `/app/model_out` (use a volume for persistence)

## Docker Compose (C03)

Bring up the app with predefined volumes:
```bash
docker compose up bert_app
```

Override text at runtime:
```bash
docker compose run --rm bert_app --text "This is awesome!"
```

Services:
- `bert_app`: builds a local image, uses volumes for model/logs/HF cache, falls back to base model if `model_out` is empty

Volumes:
- `model_out` -> `/app/model_out`
- `logs` -> `/app/logs`
- `hf_cache` -> `/app/.cache/huggingface`

## CI/CD (C04)

GitHub Actions workflows:
- `test.yml`: runs pytest on pushes/PRs.
- `evaluate.yml`: runs `python scripts/evaluate.py` (F1 macro >= 0.85) and uploads `evaluation_metrics.json`.
- `build.yml`: on push to `main`, runs tests + eval then builds and pushes Docker image to Docker Hub.

Docker Hub:
- Repository: `leo0679/bert-sentiment`
- Secrets required: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

Evaluation details:
- Uses `scripts/evaluate.py` on `dataset.csv`
- Threshold: F1 macro >= 0.85 (fails workflow otherwise)
- If no fine-tuned weights in `model_out`, falls back to baseline label mapping
