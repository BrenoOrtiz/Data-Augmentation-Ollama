import logging

from config import AUGMENTATION_RATIOS, GENERATED_DIR, RAW_DIR
from src.augmentation.pipeline import run_augmentation_pipeline
from src.data.loader import load_financial_phrasebank, save_raw_splits, split_dataset


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Data Augmentation with Ollama — Step 1: Dataset Generation")
    logger.info("=" * 70)
    logger.info(
        "Scenarios: %s",
        [f"-{int(r*100)}%% real / +{int(r*100)}%% synthetic" for r in AUGMENTATION_RATIOS],
    )

    df = load_financial_phrasebank()

    train_df, test_df = split_dataset(df)
    save_raw_splits(train_df, test_df)

    summary_df = run_augmentation_pipeline(train_df)


    logger.info("\n%s", "=" * 70)
    logger.info("AUGMENTATION SUMMARY")
    logger.info("%s", "=" * 70)
    print("\n" + summary_df.to_string(index=False) + "\n")

    logger.info("Raw data  → %s", RAW_DIR)
    logger.info("Generated → %s", GENERATED_DIR)
    logger.info("Step 1 complete.")


if __name__ == "__main__":
    main()
