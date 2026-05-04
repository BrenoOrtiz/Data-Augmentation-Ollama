import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from config import LABEL_NAMES


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Some generated CSVs only carry `sentiment`; derive `label` if missing."""
    if "label" not in df.columns:
        df = df.copy()
        df["label"] = df["sentiment"].map(LABEL_NAMES)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


class SentimentDataset(Dataset):
    """Tokenizes on-the-fly from a pandas DataFrame with `text` and `label`."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        df = _ensure_label_column(df)
        self.texts: list[str] = df["text"].astype(str).tolist()
        self.labels: list[int] = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
