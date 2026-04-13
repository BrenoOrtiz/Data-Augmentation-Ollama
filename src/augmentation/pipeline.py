import logging
import math

import pandas as pd
from tqdm import tqdm

from config import AUGMENTATION_RATIOS, GENERATED_DIR, LABEL_NAMES, OLLAMA_MODELS, RAW_DIR, SEED
from src.augmentation.generator import generate_sentence
from src.data.restrictor import restrict_training_data

logger = logging.getLogger(__name__)

_LABEL_TO_ID: dict[str, int] = LABEL_NAMES  # {"negative": 0, "neutral": 1, "positive": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_quota(train_df: pd.DataFrame, ratio: float) -> dict[str, int]:
    """
    Compute per-class synthetic sample counts.

    Total synthetic = ratio * len(train_df), distributed proportionally
    to the original class distribution so balance is preserved.
    """
    total_synthetic = math.ceil(len(train_df) * ratio)
    proportions = train_df["sentiment"].value_counts(normalize=True)
    return {
        sentiment: max(1, math.ceil(total_synthetic * prop))
        for sentiment, prop in proportions.items()
    }


def _slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


# ---------------------------------------------------------------------------
# Core generation for one (model, ratio) scenario
# ---------------------------------------------------------------------------

def _generate_scenario(
    train_df: pd.DataFrame,
    model: str,
    ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For one scenario:
      1. Restrict real data to (1 - ratio) * N samples (stratified).
      2. Generate ratio * N synthetic samples (stratified by class).
      3. Return (augmented_df, synthetic_df).

    The augmented dataset has the same total size as the original train_df.
    """
    restricted_df = restrict_training_data(train_df, ratio)
    quota = _synthetic_quota(train_df, ratio)  # based on FULL train to match original N
    total_synthetic = sum(quota.values())

    logger.info(
        "  Generating %d synthetic samples %s",
        total_synthetic,
        {k: v for k, v in quota.items()},
    )

    records: list[dict] = []

    for sentiment, count in quota.items():
        failures = 0
        generated = 0
        max_failures = count * 3

        with tqdm(
            total=count,
            desc=f"    {sentiment:8s}",
            unit="sent",
            leave=False,
        ) as pbar:
            while generated < count:
                text = generate_sentence(model, sentiment)
                if text:
                    records.append(
                        {
                            "text": text,
                            "label": _LABEL_TO_ID[sentiment],
                            "sentiment": sentiment,
                            "source": "synthetic",
                            "model": model,
                        }
                    )
                    generated += 1
                    failures = 0
                    pbar.update(1)
                else:
                    failures += 1
                    if failures >= max_failures:
                        logger.error(
                            "Too many failures for model=%s sentiment=%s — stopping.",
                            model,
                            sentiment,
                        )
                        break

    synthetic_df = pd.DataFrame(records)

    original_part = restricted_df.copy()
    original_part["source"] = "original"
    original_part["model"] = "—"

    augmented_df = (
        pd.concat([original_part, synthetic_df], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
        .assign(label=lambda df: df["label"].astype(int))
    )

    return augmented_df, synthetic_df


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def run_augmentation_pipeline(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all (model × ratio) combinations.

    For each scenario the real training set is reduced by `ratio` and
    replaced with an equal amount of synthetic data, keeping total N constant.

    Saves under data/generated/<model_slug>/:
      - train_augmented_<ratio>pct.csv   — full augmented training set
      - synthetic_only_<ratio>pct.csv    — only the generated samples
      - train_restricted_<ratio>pct.csv  — real-only restricted set (baseline)

    Returns a summary DataFrame.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for model in OLLAMA_MODELS:
        slug = _slug(model)
        model_dir = GENERATED_DIR / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n%s", "=" * 70)
        logger.info("Model: %s", model)
        logger.info("%s", "=" * 70)

        for ratio in AUGMENTATION_RATIOS:
            ratio_tag = f"{int(ratio * 100)}pct"
            logger.info(
                "\n  Scenario: -%d%% real  +%d%% synthetic  (total = %d)",
                int(ratio * 100),
                int(ratio * 100),
                len(train_df),
            )

            augmented_df, synthetic_df = _generate_scenario(train_df, model, ratio)

            # Restricted (real-only) set — useful as a no-augmentation baseline
            restricted_df = restrict_training_data(train_df, ratio)
            restr_path = model_dir / f"train_restricted_{ratio_tag}.csv"
            restricted_df.to_csv(restr_path, index=False)

            aug_path = model_dir / f"train_augmented_{ratio_tag}.csv"
            augmented_df.to_csv(aug_path, index=False)

            syn_path = model_dir / f"synthetic_only_{ratio_tag}.csv"
            synthetic_df.to_csv(syn_path, index=False)

            logger.info(
                "  Saved → real=%d  synthetic=%d  total=%d",
                len(restricted_df),
                len(synthetic_df),
                len(augmented_df),
            )

            summary_rows.append(
                {
                    "model": model,
                    "scenario": ratio_tag,
                    "real_samples": len(restricted_df),
                    "synthetic_samples": len(synthetic_df),
                    "total_samples": len(augmented_df),
                    "original_train_size": len(train_df),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(GENERATED_DIR / "augmentation_summary.csv", index=False)
    logger.info("\nSummary saved to %s", GENERATED_DIR / "augmentation_summary.csv")

    return summary_df
