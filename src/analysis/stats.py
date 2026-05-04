"""Compute summary statistics from results/finetuning_results.csv.

Focus:
  1. Per-LLM, per-ratio: augmented − restricted delta on each metric.
  2. Per-LLM, per-variant: how each metric varies as ratio grows.
  3. Overall ranking of (llm, variant) configurations.
"""

import logging

import pandas as pd

from config import RESULTS_DIR

logger = logging.getLogger(__name__)

_METRICS = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "loss"]


def _ratio_to_int(tag: str) -> int:
    return int(tag.replace("pct", ""))


def _format(df: pd.DataFrame) -> str:
    return df.round(4).to_string()


def augmented_vs_restricted(df: pd.DataFrame) -> pd.DataFrame:
    """For each (llm, ratio), compute augmented − restricted on every metric."""
    pivot = df.pivot_table(
        index=["llm", "ratio"],
        columns="variant",
        values=_METRICS,
    )

    if "augmented" not in pivot.columns.get_level_values(1) or \
       "restricted" not in pivot.columns.get_level_values(1):
        logger.warning("Need both variants in the data to compute deltas.")
        return pd.DataFrame()

    rows = []
    for (llm, ratio), row in pivot.iterrows():
        entry = {"llm": llm, "ratio": ratio, "ratio_int": _ratio_to_int(ratio)}
        for metric in _METRICS:
            aug = row.get((metric, "augmented"))
            res = row.get((metric, "restricted"))
            if pd.isna(aug) or pd.isna(res):
                entry[f"{metric}_delta"] = None
                entry[f"{metric}_pct_change"] = None
                continue
            entry[f"{metric}_delta"] = aug - res
            entry[f"{metric}_pct_change"] = (aug - res) / res if res else None
        rows.append(entry)

    return (
        pd.DataFrame(rows)
        .sort_values(["llm", "ratio_int"])
        .drop(columns="ratio_int")
        .reset_index(drop=True)
    )


def variation_across_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """For each (llm, variant), how each metric changes as ratio grows.

    Reports min, max, range, and slope (= (last − first) / (last_pct − first_pct))
    where first/last are the lowest and highest available ratios.
    """
    df = df.copy()
    df["ratio_int"] = df["ratio"].map(_ratio_to_int)

    rows = []
    for (llm, variant), grp in df.groupby(["llm", "variant"]):
        grp = grp.sort_values("ratio_int")
        if len(grp) < 1:
            continue
        first, last = grp.iloc[0], grp.iloc[-1]
        span = last["ratio_int"] - first["ratio_int"]
        for metric in _METRICS:
            slope = (
                (last[metric] - first[metric]) / span
                if span and pd.notna(first[metric]) and pd.notna(last[metric])
                else None
            )
            rows.append(
                {
                    "llm": llm,
                    "variant": variant,
                    "metric": metric,
                    "min": grp[metric].min(),
                    "max": grp[metric].max(),
                    "range": grp[metric].max() - grp[metric].min(),
                    "first_ratio_value": first[metric],
                    "last_ratio_value": last[metric],
                    "slope_per_pct": slope,
                    "n_ratios": len(grp),
                }
            )
    return pd.DataFrame(rows).sort_values(["llm", "variant", "metric"]).reset_index(drop=True)


def overall_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank every (llm, variant, ratio) by f1_macro and accuracy."""
    cols = ["llm", "variant", "ratio", "train_size"] + _METRICS
    return df[cols].sort_values("f1_macro", ascending=False).reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    csv_path = RESULTS_DIR / "finetuning_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run train.py first.")

    df = pd.read_csv(csv_path)
    df = df.dropna(how="all").reset_index(drop=True)

    print("=" * 78)
    print(f"Loaded {len(df)} runs from {csv_path.relative_to(RESULTS_DIR.parent)}")
    print("=" * 78)
    print(_format(df[["llm", "variant", "ratio", "train_size", "accuracy", "f1_macro", "loss"]]))

    print("\n" + "=" * 78)
    print("[1] Augmented - Restricted (per llm x ratio)")
    print("    delta > 0  -> augmented better")
    print("=" * 78)
    deltas = augmented_vs_restricted(df)
    if deltas.empty:
        print("Not enough data: need both variants for at least one (llm, ratio) pair.")
    else:
        cols = ["llm", "ratio"] + [
            c for c in deltas.columns if c.endswith("_delta") and c.split("_")[0] in {"accuracy", "f1"} or c.startswith(("f1_macro_delta", "accuracy_delta", "loss_delta"))
        ]
        show = deltas[
            ["llm", "ratio", "accuracy_delta", "f1_macro_delta", "f1_weighted_delta", "loss_delta"]
        ]
        print(_format(show))

    print("\n" + "=" * 78)
    print("[2] Variation across ratios (per llm x variant)")
    print("    slope_per_pct: change in metric per +1pp of ratio")
    print("=" * 78)
    var = variation_across_ratios(df)
    if var.empty:
        print("No data to analyze.")
    else:
        focus = var[var["metric"].isin(["accuracy", "f1_macro", "loss"])]
        print(_format(focus[["llm", "variant", "metric", "min", "max", "range", "slope_per_pct", "n_ratios"]]))

    print("\n" + "=" * 78)
    print("[3] Overall ranking by f1_macro")
    print("=" * 78)
    print(_format(overall_ranking(df)))

    out_dir = RESULTS_DIR / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not deltas.empty:
        deltas.to_csv(out_dir / "augmented_vs_restricted.csv", index=False)
    if not var.empty:
        var.to_csv(out_dir / "variation_across_ratios.csv", index=False)
    overall_ranking(df).to_csv(out_dir / "ranking.csv", index=False)
    print("\nDetailed CSVs saved to", out_dir)


if __name__ == "__main__":
    main()
