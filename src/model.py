from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def build_model(name="bert-base-uncased", num_labels=3):
    """Return a pretrained BERT model ready for fine-tuning."""
    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)

def train_model(model, tokenizer, X_train, y_train, X_val, y_val, output_dir="./model_out"):
    """Fine-tune the BERT model on sentiment data."""
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_dir="./logs",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds
    )

    trainer.train()
    return model
