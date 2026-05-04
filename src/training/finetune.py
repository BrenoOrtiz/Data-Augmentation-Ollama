import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from config import (
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    SEED,
    TRAIN_BATCH_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.training.dataset import SentimentDataset

logger = logging.getLogger(__name__)


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def finetune_tinybert(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    base_model: PreTrainedModel,
    initial_state: dict,
    output_dir: Path,
    run_name: str,
) -> dict[str, float]:
    """
    Fine-tune TinyBERT on `train_df`, evaluate on `test_df`.

    The caller passes a single `tokenizer` + `base_model` instance that is
    reused across runs. We reset the model to `initial_state` (a snapshot of
    its weights right after `from_pretrained`) before every call so each run
    starts from identical initial parameters — fair comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model = base_model
    model.load_state_dict(copy.deepcopy(initial_state))

    train_ds = SentimentDataset(train_df, tokenizer, MAX_SEQ_LENGTH)
    eval_ds = SentimentDataset(test_df, tokenizer, MAX_SEQ_LENGTH)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        run_name=run_name,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=False,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )

    logger.info("[%s] Training on %d samples …", run_name, len(train_ds))
    trainer.train()

    logger.info("[%s] Evaluating on %d samples …", run_name, len(eval_ds))
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    return {k.replace("eval_", ""): v for k, v in metrics.items()}
