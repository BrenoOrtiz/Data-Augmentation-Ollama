import logging

from config import RESULTS_DIR
from src.analysis.stats import main as run_stats
from src.training.pipeline import run_finetuning_pipeline


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
    logger.info("Step 2: TinyBERT fine-tuning across all (LLM × ratio × variant) runs")
    logger.info("=" * 70)

    summary_df = run_finetuning_pipeline()

    logger.info("\n%s", "=" * 70)
    logger.info("FINETUNING SUMMARY")
    logger.info("%s", "=" * 70)
    if not summary_df.empty:
        print("\n" + summary_df.to_string(index=False) + "\n")
    logger.info("Results → %s", RESULTS_DIR)

    logger.info("\n%s", "=" * 70)
    logger.info("Step 3: Computing summary statistics")
    logger.info("%s", "=" * 70)
    run_stats()


if __name__ == "__main__":
    main()
