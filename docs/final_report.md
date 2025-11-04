# Collaborative Sentiment Analysis Pipeline — Final Report

**Authors**: Léo Merville (Student 1), Paul Mouyebissi (Student 2)  
**Date**: Nov 2025

## 1. Introduction
Objectif, contexte, choix BERT (bert-base-uncased).

## 2. Work Distribution
- Student 1 (Léo): Extraction, Processing (clean/split/tokenizer tests), CI, E2E.
- Student 2 (Paul): Model training (Trainer), Inference, model/inference tests, review.

## 3. Methodology & Architecture
Décrire chaque fichier: data_extraction.py, data_processing.py, model.py, inference.py, tests (unit + integration), CI.

## 4. Implementation Highlights
Label mapping (1–2=neg/0, 3=neu/1, 4–5=pos/2), max_length=128, Trainer.

## 5. Testing & CI
- Unit tests: extraction + processing + (model/inference si présent)
- Integration test: **PASSED** (insérer capture)
- GitHub Actions: workflow tests.yml (insérer capture)

## 6. Collaboration & PM
Trello: captures To Do/In Progress/In Review/Done + cartes avec PR links.  
Git workflow: branches feature/*, PRs, reviews croisées (insérer captures).

## 7. Challenges & Fixes
Imports src, premier run HF (téléchargements), gestion branches/PR.

## 8. Conclusion & Future Work
Améliorer nettoyage, métriques (F1 macro), sauvegarde tokenizer, API FastAPI/UI.
