import torch
from src.model import build_model

def test_model_output_shape():
    model = build_model()
    model.eval()  # pour éviter les erreurs de dropout en test

    # Crée un batch factice de 2 phrases avec 8 tokens chacune
    input_ids = torch.randint(0, 30522, (2, 8))  # 30522 = vocab size BERT base
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Vérifie que la sortie contient des logits
    assert hasattr(outputs, "logits")

    # Vérifie la forme : (batch_size, num_labels)
    logits = outputs.logits
    assert logits.shape[0] == 2
    assert logits.ndim == 2

    # Vérifie aussi que le modèle a un classifieur
    assert hasattr(model, "classifier")
