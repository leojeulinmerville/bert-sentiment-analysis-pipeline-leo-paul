from src.model import build_model

def test_model_output_shape():
    model = build_model()
    out = model(input_ids=None, attention_mask=None, labels=None)
    assert hasattr(model, "classifier")
