# 🧠 BERT Sentiment Analysis Pipeline

**Authors:**  
- **Léo MERVILLE** (Student 1) – Data Extraction, Data Cleaning, Testing, Documentation  
- **Paul MOUYEBISSI** (Student 2) – Model Training, Inference, Tokenization Review  

**Course:** Collaborative AI Project (MLOps)  
**Date:** October–November 2025  

---

## 📋 Project Overview

This project implements a **complete sentiment analysis pipeline** using **BERT (bert-base-uncased)**.  
It processes textual reviews stored in a CSV file and predicts **sentiment labels**: *negative, neutral, or positive*.  

The work was carried out collaboratively, following real-world **software engineering practices**:  
- Version control & peer reviews  
- Unit testing & integration testing  
- Continuous Integration (CI/CD) via GitHub Actions  
- Agile project management via **Trello** and **Discord**

---

## 🧩 Pipeline Architecture

| Stage | File | Description | Lead |
|--------|------|-------------|------|
| **Data Extraction** | `src/data_extraction.py` | Load raw CSV data, validate columns, handle errors | Léo |
| **Data Processing** | `src/data_processing.py` | Clean text, map sentiment labels, tokenize inputs | Léo & Paul |
| **Model Training** | `src/model.py` | Fine-tune pretrained BERT for sentiment classification | Paul |
| **Inference** | `src/inference.py` | Predict sentiment from new text inputs | Paul |
| **Testing** | `tests/unit/` & `tests/integration/` | Validate all components through unit and E2E tests | Both |

---

## 🧠 Dataset

The dataset is composed of user-generated textual reviews (similar to Google Play or App Store).  
Each review includes:
- `content`: the text of the review  
- `score`: a numerical rating (1–5)

These ratings are mapped to sentiment classes:
| Rating | Sentiment | Label |
|---------|------------|--------|
| 1–2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4–5 | Positive | 2 |

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.11 |
| **Modeling** | Hugging Face Transformers (`bert-base-uncased`), PyTorch |
| **Data Handling** | pandas, scikit-learn |
| **Testing** | pytest, pytest-cov |
| **CI/CD** | GitHub Actions |
| **Project Management** | Trello (kanban workflow), Discord (communication) |

---

## 🔁 Git Workflow

Development followed a **feature-branch model**:

| Branch | Purpose | Owner |
|---------|----------|--------|
| `main` | Stable version (merged after review) | Both |
| `feature/data-extraction` | Data loading functions | Léo |
| `feature/data-processing-cleaning` | Text cleaning & dataset split | Léo |
| `feature/data-processing-tokenizer` | Tokenization logic | Léo & Paul |
| `feature/model-training` | Model fine-tuning | Paul |
| `feature/inference` | Inference script | Paul |
| `feature/testing-and-report` | Final integration tests & report | Léo |

Each new feature → **Pull Request → Review by peer → Merge into `main`**.  
All PRs included CI checks and comments for validation.

---

## ✅ Trello & Communication

**Trello Board:** *Sentiment Analysis Project – Léo Merville & Paul Mouyebissi*  
(Columns: *To Do* → *In Progress* → *In Review* → *Done*)

**Discord Server:** Used for day-to-day collaboration and debugging discussions.

### Roles & Labels
| Label | Area |
|--------|------|
| 🟩 Data | Extraction / Cleaning |
| 🟨 Model | Training / Inference |
| 🟧 Testing | Unit & Integration Tests |
| 🟪 Documentation | Reporting |
| 🟦 Backend | Project setup / CI |

<img width="3198" height="1729" alt="Capture d&#39;écran 2025-11-04 142029" src="https://github.com/user-attachments/assets/29678bd5-af21-475c-bb7f-378f25b47b8d" />

---

## 🧪 Testing

All modules include **unit and integration tests**.  
To run the full suite with coverage:

Coverage target: ≥ 90%

✅ Test Summary
Type	Location	Description
Unit Tests	tests/unit/	Validate each module independently
Integration Test	tests/integration/test_pipeline_end_to_end.py	Full pipeline verification
Continuous Integration	.github/workflows/tests.yml	Auto-run tests on each PR

All 10 tests passed successfully in the final version.

<img width="1505" height="909" alt="Capture d&#39;écran 2025-11-04 141605" src="https://github.com/user-attachments/assets/ca905947-2a6b-44bb-b44a-90984e784bae" />

⚙️ Continuous Integration (CI)

A GitHub Actions workflow runs automatically on every push or pull request.

Steps:

Set up Python environment

Install dependencies from requirements.txt

Run all tests with coverage

Block PR merge if tests fail

Example log:

pytest -v
tests/unit/test_data_extraction.py::test_load_valid_csv PASSED
tests/unit/test_data_processing.py::test_clean_text_basic PASSED
tests/integration/test_pipeline_end_to_end.py::test_pipeline_end_to_end PASSED

📦 How to Run the Project
Installation
git clone https://github.com/leojeulinmerville/bert-sentiment-analysis-pipeline-leo-paul.git
cd bert-sentiment-analysis-pipeline-leo-paul
pip install -r requirements.txt

Run Tests
pytest -v

Quick Inference Example
from src.inference import load_model, predict_sentiment
tok, mdl = load_model(model_path="./model_out", name="bert-base-uncased")
print(predict_sentiment("I love this app!", tok, mdl))  # → 0 / 1 / 2

🧾 Deliverables
Deliverable	Description	Status
GitHub Repository (Public)	Full code, commits, branches, CI logs
Trello Board	Project tracking and reviews
Discord Proof	Collaboration evidence (screenshots)
Project Report	Markdown & PDF format under /docs/
Unit & Integration Tests	All passed (10/10)
CI/CD Workflow	Tests executed automatically

### Conclusion

The project successfully delivered a working sentiment analysis pipeline with:

Functional data-to-inference flow

Reliable test coverage

Verified collaboration & review process

Automated CI pipeline

Future improvements:

Add evaluation metrics (Accuracy, F1-Score)

Integrate confusion matrix visualization

Expose inference through an API (FastAPI)
```bash
pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing
