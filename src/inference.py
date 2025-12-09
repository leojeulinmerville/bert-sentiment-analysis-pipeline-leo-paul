import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path="./model_out", name="bert-base-uncased"):
    """Load fine-tuned BERT model and tokenizer; fallback to base model if directory is missing."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    path = Path(model_path)
    if path.exists():
        model = AutoModelForSequenceClassification.from_pretrained(path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name)
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    """Predict sentiment label (0, 1, 2) for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    return pred
