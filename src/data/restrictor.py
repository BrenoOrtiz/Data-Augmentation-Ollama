import logging

import pandas as pd

from config import SEED

logger = logging.getLogger(__name__)


def restrict_training_data(train_df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    Simulate data scarcity for a given scenario ratio.

    Removes `ratio` fraction of the training set (stratified by class),
    leaving (1 - ratio) * N real samples. The pipeline will then generate
    exactly `ratio * N` synthetic samples to restore the original dataset size.

    Parameters
    ----------
    train_df : full training DataFrame
    ratio    : fraction to remove (e.g. 0.10, 0.25, 0.50)

    Returns
    -------
    restricted_df : (1 - ratio) fraction of train_df, class-balanced
    """
    keep_frac = 1.0 - ratio

    restricted = (
        train_df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(frac=keep_frac, random_state=SEED))
        .reset_index(drop=True)
    )

    logger.info(
        "  Restriction -%d%%: %d → %d real samples kept",
        int(ratio * 100),
        len(train_df),
        len(restricted),
    )

    return restricted
