# 🧠 Collaborative Sentiment Analysis Pipeline — Final Report

**Authors**  
- Léo Merville (Student 1)  
- Paul Mouyebissi (Student 2)  

**Course:** MLOps – Aivancity  
**Date:** November 2025  

---

## 1. Introduction

The goal of this project was to implement a **complete sentiment analysis pipeline** using **BERT (bert-base-uncased)**.  
The pipeline includes:
- Data extraction and validation  
- Text cleaning and tokenization  
- BERT fine-tuning (Hugging Face Trainer)  
- Inference for sentiment prediction  
- Unit and integration testing with Pytest  
- Continuous Integration via GitHub Actions  

All work followed professional **DevOps & collaborative standards**: Git branching, code reviews, Trello coordination, and peer validation.

---

## 2. Work Distribution

| Role | Student | Main Responsibilities |
|------|----------|------------------------|
| Student 1 | **Léo Merville** | Data extraction, data cleaning, testing (unit + E2E), documentation, CI setup |
| Student 2 | **Paul Mouyebissi** | Model fine-tuning, inference module, tokenization, PR reviews, communication |

---

## 3. Methodology & Architecture

### Repository Structure

<img width="870" height="1353" alt="image" src="https://github.com/user-attachments/assets/41501a0b-b180-4dc7-9d25-be00d0ca4cc9" />


### Pipeline Flow
`Dataset → Cleaning/Labeling → Tokenization → Model Fine-Tuning → Inference`

Each module was developed on a **dedicated branch**, reviewed via PR, and merged only after passing all unit tests.

---

## 4. Implementation Overview

| File | Description |
|------|--------------|
| `data_extraction.py` | Loads CSV, validates columns, handles missing files |
| `data_processing.py` | Cleans text, maps numeric scores to 3 sentiment labels, tokenizes with BERT |
| `model.py` | Builds and fine-tunes BERT with Hugging Face Trainer |
| `inference.py` | Loads trained model, predicts sentiment (0–2) |
| `tests/` | Unit + integration tests ensuring pipeline integrity |

---

## 5. Testing & Results

✅ **10 tests passed (unit + integration)**

<img width="1505" height="909" alt="Capture d&#39;écran 2025-11-04 141605" src="https://github.com/user-attachments/assets/d38b1637-fca4-44a9-b3ef-96e0a7a0cee8" />


- **Unit tests** verified all core modules individually.  
- **Integration test** checked full pipeline: data → tokenization → inference.

### Command
```bash
pytest -v
```
End-to-end success

All 10/10 tests passed in 34.61s confirming full pipeline functionality.

6. Collaboration & Project Management
Trello Board

All project phases were tracked on a shared Trello board with roles, labels, and PR links.
Each card matched a Git branch and contained checklists and comments.

<img width="3198" height="1729" alt="Capture d&#39;écran 2025-11-04 142029" src="https://github.com/user-attachments/assets/77bcc3ff-0df5-489c-a733-8fb5de2fea7a" />

Example Cards

Roles & Labels + Repo

<img width="2127" height="1355" alt="Capture d&#39;écran 2025-11-04 142114" src="https://github.com/user-attachments/assets/4d652ec3-a7fe-4eef-8193-a2cb74202ca0" />

<img width="2137" height="1003" alt="Capture d&#39;écran 2025-11-04 142126" src="https://github.com/user-attachments/assets/45f467e3-9ce5-41a7-ac23-67e2c31a856c" />

Data Extraction (Student 1)

<img width="2140" height="1486" alt="Capture d&#39;écran 2025-11-04 142207" src="https://github.com/user-attachments/assets/536c0498-7f84-4861-aa9d-bcc52acadc73" />

Model Training (Student 2)

<img width="2165" height="1643" alt="Capture d&#39;écran 2025-11-04 142235" src="https://github.com/user-attachments/assets/1d582cb4-4fa8-4694-99cd-8a96d58536e1" />

Tokenization

<img width="2140" height="1311" alt="Capture d&#39;écran 2025-11-04 142249" src="https://github.com/user-attachments/assets/ab1f6398-3fa7-40a7-96c8-ba49e682b4d7" />

Chronogram
<img width="2137" height="1439" alt="Capture d&#39;écran 2025-11-04 142143" src="https://github.com/user-attachments/assets/f914b58e-1faa-4820-8661-1bebab8e1118" />

### 6.1 Pull Requests & Code Reviews

Throughout the project, every new feature or module was developed on a separate branch  
and merged into `main` only after **peer review** and **successful CI tests**.

#### Summary of Main PRs

| PR # | Title | Author | Reviewer | Description | Status |
|------|--------|---------|-----------|--------------|---------|
| #3 | `ci: add GitHub Actions pytest workflow` | Léo | Paul | Setup of continuous integration to auto-run Pytest | ✅ Merged |
| #4 | `feat(processing): cleaning, label mapping, stratified split + unit tests` | Léo | Paul | Added preprocessing module and tests | ✅ Merged |
| #5 | `feature/data processing tokenizer` | Léo | Paul | Implemented tokenizer & PR review from Student 2 | ✅ Merged |
| #6 | `feat(model): model training + inference scripts` | Paul | Léo | Added model fine-tuning & inference modules | ✅ Merged |
| #7 | `test: add end-to-end pipeline test` | Léo | Paul | Full pipeline integration test (E2E) | ✅ Merged |
   #### Direct PR Links
- [#3 – CI Workflow](https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul/pull/3)
- [#4 – Data Processing Cleaning](https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul/pull/4)
- [#5 – Tokenizer](https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul/pull/5)
- [#6 – Model + Inference](https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul/pull/6)
- [#7 – Integration Test](https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul/pull/7)

7. Communication


Team communication took place on Discord, including:

Daily coordination messages

Debugging (inference & tests)

Validation of each stage before merge

<img width="3075" height="1848" alt="Capture d&#39;écran 2025-11-04 142339" src="https://github.com/user-attachments/assets/7fd1a91d-7ef0-458b-b234-72282eaef545" />

8. Continuous Integration

A GitHub Actions workflow automatically ran Pytest on each push:

Triggered on pull_request and main

Ensured all tests were green before merge

9. Challenges & Fixes
Issue	Cause	Solution
ModuleNotFoundError: src	Pytest import issue	Added __init__.py and pytest.ini path
Long model download	Hugging Face cache initialization	Cached model locally
GitHub Actions path errors	Relative import mismatch	Updated working dir
Dataset column mismatch	CSV headers	Validation added in load_reviews_csv()

11. Conclusion & Future Work

The project achieved a fully functional sentiment analysis pipeline, integrated and tested under real MLOps constraints.

Achievements

Modular pipeline from extraction to inference

100% of tests passed

Collaborative workflow (Trello + GitHub)

CI/CD setup

Future Improvements

Fine-tune model for higher accuracy

Add confusion matrix visualization

Serve inference via FastAPI endpoint

11. Deliverables Summary
Deliverable	Status
GitHub repository (public): https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul

README.md	:Complete
Unit & Integration Tests	:10 passed
Trello Board	: [Linked](https://trello.com/invite/b/6900a589ef735a1e9525226e/ATTI5bbd19b68216146280901ba7de845048B2407F38/sentiment-analysis-project-leo-merville-paul-mouyebissi)
Discord Communication Proof	:Included
Final Report	:This document

<img width="870" height="1353" alt="Capture d&#39;écran 2025-11-04 143114" src="https://github.com/user-attachments/assets/730656dd-689b-4de0-9d96-54285b7f575d" />

---

# Part 2 – MLOps (Docker, Compose, CI/CD)

## Repository & Image
- Repo: https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul
- Docker Hub: `leo0679/bert-sentiment`
- Students: student_1 (dev) `leojeulinmerville`; student_2 (review) `derfoj`

## C01 – Dockerfile
- Base: `python:3.11-slim`, `WORKDIR /app`, `PYTHONUNBUFFERED=1`, HF cache at `/app/.cache/huggingface`.
- Installs `requirements.txt`, copies `src/` and `dataset.csv`.
- Entrypoint: `python -m src.cli` (e.g., `--text "Hello"`).
- Local test: `docker build -t bert-sentiment-cli .` then `docker run --rm bert-sentiment-cli --text "I love this app!"`.

## C02 – Volumes
- Persistent dirs created: `/app/model_out`, `/app/logs`, optional `/app/.cache/huggingface`.
- Named volume example:
  - `docker run --rm -v bert_model_out:/app/model_out -v bert_logs:/app/logs bert-sentiment-cli --text "Great product!"`
- Bind mounts:
  - `docker run --rm -v /abs/path/model_out:/app/model_out -v /abs/path/logs:/app/logs bert-sentiment-cli --text "Great product!"`

## C03 – Docker Compose
- `docker-compose.yml` service `bert_app` builds local image, mounts `model_out`, `logs`, `hf_cache`, sets `HF_HOME`.
- Commands:
  - `docker compose up bert_app`
  - `docker compose run --rm bert_app --text "Compose check"`
- Inference falls back to base model if mounted `model_out` is empty/invalid.

## C04 – CI/CD (GitHub Actions)
- `test.yml`: runs `pytest -v` on push/PR.
- `evaluate.yml`: runs `python scripts/evaluate.py --threshold 0.85` (F1 macro >= 0.85) and uploads `evaluation_metrics.json`; sets `PYTHONPATH` to repo root.
- `build.yml`: on push to `main`, runs tests + eval, then builds/pushes Docker image to `leo0679/bert-sentiment` (tags: `latest`, `${{ github.sha }}`) using `docker/login-action@v3` and GHA cache.
- Secrets required: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (write scope).

## Evaluation Script
- `scripts/evaluate.py`: loads `dataset.csv`, maps scores to labels, uses fine-tuned model if present (`model_out` with config + weights), otherwise baseline label mapping. Computes F1 macro; fails if < 0.85; saves metrics to `evaluation_metrics.json`.

## Evidence to Capture (screenshots)
<img width="3199" height="1088" alt="image" src="https://github.com/user-attachments/assets/67696299-5230-4f99-be19-5953708fee01" />
<img width="3194" height="1447" alt="image" src="https://github.com/user-attachments/assets/0d30724e-f775-4250-964c-606cbcddc3a3" />
<img width="3199" height="1582" alt="image" src="https://github.com/user-attachments/assets/73d2ac23-77ed-40da-ba72-289315d3b049" />

<img width="2633" height="1402" alt="image" src="https://github.com/user-attachments/assets/f079a1e0-e4fc-4ffb-af4a-7c1c5ceef89e" />

- PRs: merged #9 (C01), #10 (C02), #11 (C03), #12 (C04) with approvals.
  ![WhatsApp Image 2025-12-09 à 17 12 46_73b293fb](https://github.com/user-attachments/assets/4afa220a-def6-4853-b85e-7ea6a9ed10ae)

![WhatsApp Image 2025-12-09 à 17 12 47_dd362c75](https://github.com/user-attachments/assets/b64582e5-ec45-461c-a8c8-ae8b4064bb57)

![WhatsApp Image 2025-12-09 à 17 12 46_36a8fc76](https://github.com/user-attachments/assets/52cfc7a9-77fe-4247-97af-4177205578d3)

![WhatsApp Image 2025-12-09 à 17 12 47_785e6c13](https://github.com/user-attachments/assets/aca909fe-680e-45ee-840a-c72d6c81bfb0)

![WhatsApp Image 2025-12-09 à 17 12 47_ed59e04c](https://github.com/user-attachments/assets/a922eaa9-c298-40fc-a4ad-7eeb709ab5bf)

![WhatsApp Image 2025-12-09 à 17 12 17_13b94405](https://github.com/user-attachments/assets/ba3b8955-f713-4bb4-8142-f969f94f71fa)


## How to Run (summary)
```bash
# Build and run CLI
docker build -t bert-sentiment-cli .
docker run --rm bert-sentiment-cli --text "I love this app!"

# With volumes
docker run --rm \
  -v bert_model_out:/app/model_out \
  -v bert_logs:/app/logs \
  bert-sentiment-cli --text "Great product!"

# Compose
docker compose run --rm bert_app --text "Compose check"
```
