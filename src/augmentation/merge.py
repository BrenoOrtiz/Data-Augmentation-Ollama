import logging

import pandas as pd

from config import (
    AUGMENTATION_RATIOS,
    GENERATED_DIR,
    LABEL_NAMES,
    OLLAMA_MODELS,
    SEED,
)

logger = logging.getLogger(__name__)


def _slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def _normalize_restricted(df: pd.DataFrame) -> pd.DataFrame:
    """Restricted CSVs may lack `label`/`source`/`model` — fill them in."""
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = df["sentiment"].map(LABEL_NAMES)
    df["source"] = "original"
    df["model"] = "—"
    return df[["text", "label", "sentiment", "source", "model"]]


def _normalize_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = df["sentiment"].map(LABEL_NAMES)
    return df[["text", "label", "sentiment", "source", "model"]]


def merge_augmented_files() -> None:
    """
    Rebuild every `train_augmented_<ratio>pct.csv` as the concatenation of
    its sibling `train_restricted_<ratio>pct.csv` + `synthetic_only_<ratio>pct.csv`.

    Useful when the augmented files were saved incorrectly — synthetic
    generation is expensive, so we just remerge from disk.
    """
    for llm in OLLAMA_MODELS:
        slug = _slug(llm)
        model_dir = GENERATED_DIR / slug
        if not model_dir.exists():
            logger.warning("Skipping %s — %s does not exist.", llm, model_dir)
            continue

        for ratio in AUGMENTATION_RATIOS:
            tag = f"{int(ratio * 100)}pct"
            restricted_path = model_dir / f"train_restricted_{tag}.csv"
            synthetic_path = model_dir / f"synthetic_only_{tag}.csv"
            augmented_path = model_dir / f"train_augmented_{tag}.csv"

            if not restricted_path.exists() or not synthetic_path.exists():
                logger.warning(
                    "Skipping %s/%s — missing restricted or synthetic CSV.",
                    slug, tag,
                )
                continue

            restricted_df = _normalize_restricted(pd.read_csv(restricted_path))
            synthetic_df = _normalize_synthetic(pd.read_csv(synthetic_path))

            augmented_df = (
                pd.concat([restricted_df, synthetic_df], ignore_index=True)
                .dropna(subset=["text", "label"])
                .assign(label=lambda d: d["label"].astype(int))
                .sample(frac=1, random_state=SEED)
                .reset_index(drop=True)
            )

            augmented_df.to_csv(augmented_path, index=False)

            logger.info(
                "Rebuilt %s  (real=%d + synthetic=%d → total=%d)",
                augmented_path.relative_to(GENERATED_DIR.parent),
                len(restricted_df),
                len(synthetic_df),
                len(augmented_df),
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    merge_augmented_files()
