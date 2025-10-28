import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def map_score_to_label(score: int) -> int:
    """
    Map numerical scores to sentiment labels:
    1-2 → 0 (negative), 3 → 1 (neutral), 4-5 → 2 (positive)
    """
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2

_url_re = re.compile(r"http\S+|www\.\S+")
_word_re = re.compile(r"[^a-z0-9'\s]")

def clean_text(text: str) -> str:
    """
    Clean review text:
    - Lowercase
    - Remove URLs and special characters
    - Compact multiple spaces
    """
    t = str(text).lower()
    t = _url_re.sub(" ", t)
    t = _word_re.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def prepare_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare stratified train/validation splits with labels and cleaned text.
    Returns lists: X_train, X_val, y_train, y_val
    """
    if "content" not in df.columns or "score" not in df.columns:
        raise ValueError("DataFrame must contain 'content' and 'score'")

    w = df[["content", "score"]].dropna().copy()
    w["label"] = w["score"].astype(int).map(map_score_to_label)
    w["text"] = w["content"].astype(str).map(clean_text)

    X_tr, X_val, y_tr, y_val = train_test_split(
        w["text"], w["label"],
        test_size=test_size, random_state=random_state, stratify=w["label"]
    )

    return X_tr.tolist(), X_val.tolist(), y_tr.tolist(), y_val.tolist()

def build_tokenizer(name: str = "bert-base-uncased"):
    """Return a Hugging Face tokenizer for BERT."""
    return AutoTokenizer.from_pretrained(name)

def tokenize_texts(tokenizer, texts, max_length: int = 128):
    """Tokenize a list of strings for BERT (padded/truncated)."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
