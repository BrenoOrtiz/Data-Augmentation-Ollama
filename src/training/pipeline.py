import copy
import logging

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    AUGMENTATION_RATIOS,
    GENERATED_DIR,
    LABEL_NAMES,
    MODELS_DIR,
    OLLAMA_MODELS,
    RAW_DIR,
    RESULTS_DIR,
    TINYBERT_MODEL,
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
      - baseline_full: full real training set (no removal, no augmentation)
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

    # Load tokenizer + base model ONCE. Repeated `from_pretrained` calls in the
    # same process (a) reuse a httpx client that gets closed and (b) on Windows
    # without Developer Mode trigger symlink permission errors. We snapshot the
    # initial weights so each run resets to the same starting point.
    logger.info("Loading %s …", TINYBERT_MODEL)
    id2label = {v: k for k, v in LABEL_NAMES.items()}
    tokenizer = AutoTokenizer.from_pretrained(TINYBERT_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        TINYBERT_MODEL,
        num_labels=len(LABEL_NAMES),
        id2label=id2label,
        label2id=dict(LABEL_NAMES),
    )
    initial_state = copy.deepcopy(base_model.state_dict())

    results_csv = RESULTS_DIR / "finetuning_results.csv"
    rows: list[dict] = []
    done: set[tuple[str, str, str]] = set()
    if results_csv.exists():
        prev = pd.read_csv(results_csv)
        rows = prev.to_dict(orient="records")
        done = {(str(r["llm"]), str(r["variant"]), str(r["ratio"])) for r in rows}
        logger.info("Resuming — %d run(s) already in %s", len(done), results_csv)

    # Baseline: train on the full real dataset (no augmentation, no removal).
    full_train_path = RAW_DIR / "train_full.csv"
    if full_train_path.exists():
        run_name = "baseline__full"
        output_dir = MODELS_DIR / "baseline" / "full"
        baseline_key = ("-", "baseline_full", "0pct")

        if baseline_key in done:
            logger.info("Skipping %s — already in results CSV.", run_name)
        else:
            logger.info("\n%s", "=" * 70)
            logger.info("Run: %s (full real dataset baseline)", run_name)
            logger.info("Train file: %s", full_train_path)
            logger.info("%s", "=" * 70)

            train_df = pd.read_csv(full_train_path)
            metrics = finetune_tinybert(
                train_df=train_df,
                test_df=test_df,
                tokenizer=tokenizer,
                base_model=base_model,
                initial_state=initial_state,
                output_dir=output_dir,
                run_name=run_name,
            )

            rows.append(
                {
                    "llm": "-",
                    "variant": "baseline_full",
                    "ratio": "0pct",
                    "train_size": len(train_df),
                    **{k: round(float(v), 4) for k, v in metrics.items()},
                }
            )
            done.add(baseline_key)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
    else:
        logger.warning("Baseline skipped — %s not found.", full_train_path)

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

                key = (llm, variant, ratio_tag)
                if key in done:
                    logger.info("Skipping %s/%s_%s — already in results CSV.", llm, variant, ratio_tag)
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
                    tokenizer=tokenizer,
                    base_model=base_model,
                    initial_state=initial_state,
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
                done.add(key)

                # Persist incrementally so a crash doesn't lose results.
                pd.DataFrame(rows).to_csv(results_csv, index=False)

    summary_df = pd.DataFrame(rows)
    logger.info("\nFinetuning results saved to %s", results_csv)
    return summary_df
