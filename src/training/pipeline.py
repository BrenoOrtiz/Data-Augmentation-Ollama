import logging

import pandas as pd

from config import (
    AUGMENTATION_RATIOS,
    GENERATED_DIR,
    MODELS_DIR,
    OLLAMA_MODELS,
    RAW_DIR,
    RESULTS_DIR,
    TRAIN_VARIANTS,
)
from src.training.finetune import finetune_tinybert

logger = logging.getLogger(__name__)


def _slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def _load_test_set() -> pd.DataFrame:
    test_path = RAW_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found at {test_path}. Run main.py (Step 1) first."
        )
    return pd.read_csv(test_path)


def run_finetuning_pipeline() -> pd.DataFrame:
    """
    Fine-tune TinyBERT for every (LLM × ratio × variant) combination.

    Variants:
      - restricted: real-only, (1 - ratio) * N samples (no augmentation baseline)
      - augmented:  real (1 - ratio)*N + synthetic ratio*N  → total N

    Saves:
      - results/finetuning_results.csv  — one row per run with metrics
      - models/<slug>/<variant>_<ratio>pct/model/  — trained TinyBERT
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    test_df = _load_test_set()
    logger.info("Loaded test set: %d samples", len(test_df))

    rows: list[dict] = []

    for llm in OLLAMA_MODELS:
        slug = _slug(llm)
        gen_dir = GENERATED_DIR / slug
        if not gen_dir.exists():
            logger.warning("Skipping %s — no generated data at %s", llm, gen_dir)
            continue

        for ratio in AUGMENTATION_RATIOS:
            ratio_tag = f"{int(ratio * 100)}pct"

            for variant in TRAIN_VARIANTS:
                csv_path = gen_dir / f"train_{variant}_{ratio_tag}.csv"
                if not csv_path.exists():
                    logger.warning("Missing %s — skipping.", csv_path)
                    continue

                run_name = f"{slug}__{variant}_{ratio_tag}"
                output_dir = MODELS_DIR / slug / f"{variant}_{ratio_tag}"

                logger.info("\n%s", "=" * 70)
                logger.info("Run: %s", run_name)
                logger.info("Train file: %s", csv_path)
                logger.info("%s", "=" * 70)

                train_df = pd.read_csv(csv_path)
                metrics = finetune_tinybert(
                    train_df=train_df,
                    test_df=test_df,
                    output_dir=output_dir,
                    run_name=run_name,
                )

                rows.append(
                    {
                        "llm": llm,
                        "variant": variant,
                        "ratio": ratio_tag,
                        "train_size": len(train_df),
                        **{k: round(float(v), 4) for k, v in metrics.items()},
                    }
                )

                # Persist incrementally so a crash doesn't lose results.
                pd.DataFrame(rows).to_csv(
                    RESULTS_DIR / "finetuning_results.csv", index=False
                )

    summary_df = pd.DataFrame(rows)
    logger.info("\nFinetuning results saved to %s", RESULTS_DIR / "finetuning_results.csv")
    return summary_df
