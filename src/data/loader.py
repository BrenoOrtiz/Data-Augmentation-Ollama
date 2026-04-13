import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASET_FILE, LABEL_NAMES, RAW_DIR, SEED, TEST_SIZE

logger = logging.getLogger(__name__)


def load_financial_phrasebank() -> pd.DataFrame:
    """
    Load Financial PhraseBank from a local text file.

    Expected format (one entry per line):
        sentence text@label
    where label is one of: positive | neutral | negative
    """
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}\n"
            "Download it from HuggingFace and place it at that path."
        )

    records = []
    with DATASET_FILE.open(encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            *sentence_parts, label = line.rsplit("@", maxsplit=1)
            text = "@".join(sentence_parts).strip()
            label = label.strip().lower()
            if label not in LABEL_NAMES:
                logger.warning("Unknown label %r — skipping line.", label)
                continue
            records.append({"text": text, "sentiment": label, "label": LABEL_NAMES[label]})

    df = pd.DataFrame(records).drop_duplicates(subset="text").reset_index(drop=True)

    logger.info("Loaded %d unique samples from %s", len(df), DATASET_FILE.name)
    logger.info("Label distribution:\n%s", df["sentiment"].value_counts().to_string())

    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train / test split. Test set is fixed and never augmented."""
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info("Split → train=%d  test=%d", len(train_df), len(test_df))
    return train_df, test_df


def save_raw_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(RAW_DIR / "train_full.csv", index=False)
    test_df.to_csv(RAW_DIR / "test.csv", index=False)
    logger.info("Raw splits saved to %s", RAW_DIR)
