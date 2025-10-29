import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path="./model_out", name="bert-base-uncased"):
    """Load fine-tuned BERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    """Predict sentiment label (0, 1, 2) for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    return pred
