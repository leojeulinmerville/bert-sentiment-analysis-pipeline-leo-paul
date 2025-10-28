from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["content", "score"]

def load_reviews_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop missing reviews or invalid scores
    df = df.dropna(subset=["content", "score"])
    df = df.reset_index(drop=True)

    return df
