🧠 BERT Sentiment Analysis Pipeline

Authors: Léo MERVILLE (Student 1) & Paul MOUYEBISSI (Student 2)
Course: Collaborative AI Project
Date: October–November 2025

📋 Project Overview

This project aims to build a complete Sentiment Analysis pipeline using a BERT model.
The pipeline processes textual reviews (from a CSV dataset) and predicts sentiment labels (positive, neutral, negative).
The work is collaborative, following professional software engineering practices — version control, unit testing, CI/CD, and project management via Trello.

🧩 Pipeline Components
Stage	File	Description	Lead
Data Extraction	data_extraction.py	Load raw CSV data, handle missing values & validation errors	Léo
Data Processing	data_processing.py	Clean text, map sentiment labels, tokenize for BERT	Léo & Paul
Model Training	model.py	Fine-tune pretrained BERT for sentiment classification	Paul
Inference	inference.py	Predict sentiment for new text inputs	Paul
Testing	tests/unit/	Validate all functions & components	Both
🧠 Dataset

The dataset used is a collection of textual reviews (e.g., Google Play/App Store style).
Each row includes a review text and a numerical rating (score).
These ratings are mapped to sentiment labels as follows:

1–2 → Negative (0)

3 → Neutral (1)

4–5 → Positive (2)

🧰 Tech Stack

Python 3.11

Transformers (Hugging Face) – bert-base-uncased

PyTorch

scikit-learn

pytest / pytest-cov – unit tests & coverage

GitHub Actions – Continuous Integration (CI)

Trello + Discord – Collaboration & communication

🔁 Git Workflow

We follow a feature-branch model:

Branch	Purpose	Owner
main	Stable version (merged PRs only)	Both
feature/data-extraction	Data loading functions	Léo
feature/data-processing-cleaning	Text cleaning & split	Léo
feature/data-processing-tokenizer	Tokenization logic	Paul & Léo
feature/model-training	Model fine-tuning	Paul
feature/inference	Inference script	Paul

Each new feature is implemented in a separate branch → Pull Request → Peer Review → Merge.

✅ Trello & Communication

Trello Board: Sentiment Analysis Project – Léo Merville & Paul Mouyebissi

(Lists: To Do / In Progress / In Review / Done)

Discord Server: Internal communication and code reviews

Roles & Labels:

🟩 Data

🟨 Model

🟧 Testing

🟪 Documentation

🟦 Backend

🧪 Testing

Unit tests are written for every component under tests/unit/.
To run all tests with coverage:

pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing


Coverage target: ≥ 90%

⚙️ Continuous Integration (CI)

A GitHub Actions workflow runs automatically on every push or pull request:

Install dependencies

Run tests

Report coverage results

Prevent merge if tests fail

📁 Project Structure (planned)
bert-sentiment-analysis-pipeline-leo-paul/
│
├── data_extraction.py
├── data_processing.py
├── model.py
├── inference.py
│
├── tests/
│   └── unit/
│       ├── test_data_extraction.py
│       ├── test_data_processing.py
│       ├── test_model.py
│       └── test_inference.py
│
├── requirements.txt
└── README.md

🧾 Deliverables

GitHub Repository (Public) – Final project with branches, commits, PRs, and CI logs

Trello Screenshots – Showing collaboration workflow

Project Report (PDF) – Approach, challenges, division of work, and improvements