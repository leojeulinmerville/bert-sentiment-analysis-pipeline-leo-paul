import pandas as pd
from src.data_processing import map_score_to_label, clean_text, prepare_splits

def test_label_mapping():
    assert map_score_to_label(1) == 0
    assert map_score_to_label(2) == 0
    assert map_score_to_label(3) == 1
    assert map_score_to_label(4) == 2
    assert map_score_to_label(5) == 2

def test_clean_text_basic():
    assert clean_text("Hello, WORLD!! https://x.y") == "hello world"

def test_prepare_splits_stratified():
    df = pd.DataFrame({
        "content": ["bad","meh","good","great","awful","okay","nice","super","mid"],
        "score":   [1,      3,     5,      5,      1,      3,      4,      4,     3]
    })
    Xtr, Xval, ytr, yval = prepare_splits(df, test_size=0.33, random_state=0)
    assert len(Xtr) + len(Xval) == len(df)
    assert set(ytr + yval).issubset({0,1,2})
