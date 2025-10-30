from src.inference import predict_sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_inference_dummy():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    pred = predict_sentiment("I love this app!", tok, model)
    assert pred in [0, 1, 2]
