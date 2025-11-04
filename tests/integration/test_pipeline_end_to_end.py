# tests/integration/test_pipeline_end_to_end.py
import pytest
from src.data_extraction import load_reviews_csv
from src.data_processing import build_tokenizer, tokenize_texts
from src.model import build_model
from src.inference import predict_sentiment

@pytest.mark.integration
def test_pipeline_end_to_end():

    df = load_reviews_csv("dataset.csv")
    sample = df.sample(3, random_state=42)
    texts = sample["content"].tolist()

    tokenizer = build_tokenizer()
    model = build_model(num_labels=3)

    tokens = tokenize_texts(tokenizer, texts)
    assert "input_ids" in tokens
    assert tokens["input_ids"].shape[0] == len(texts)

    results = []
    for t in texts:
        label = predict_sentiment(t, tokenizer, model)
        assert isinstance(label, int)
        assert label in (0, 1, 2)
        results.append(label)

    assert len(results) == len(texts)
    print("\nSample predictions:", results)
