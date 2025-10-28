import pytest
import pandas as pd
from src.data_extraction import load_reviews_csv

def test_load_valid_csv(tmp_path):
    # CSV temporaire correct
    csv_path = tmp_path / "reviews.csv"
    df = pd.DataFrame({"content": ["Good app", "Bad update"], "score": [5, 1]})
    df.to_csv(csv_path, index=False)

    result = load_reviews_csv(csv_path)
    assert list(result.columns) == ["content", "score"]
    assert result.shape[0] == 2

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_reviews_csv("C:/invalid/path/non_existing.csv")

def test_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_reviews_csv(bad_csv)
