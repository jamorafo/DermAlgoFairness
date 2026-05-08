#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of these columns found: {candidates}. Available columns: {df.columns.tolist()}")


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "auc_pr": average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
    }
    return out


def bootstrap_gap(df, y_col, score_col, subgroup_col, metric, n_boot=2000, threshold=0.5, seed=1):
    rng = np.random.default_rng(seed)

    dark = df[df[subgroup_col] == "dark"].copy()
    light = df[df[subgroup_col] == "light"].copy()

    if len(dark) == 0 or len(light) == 0:
        raise ValueError("Both dark and light subgroups are required.")

    obs_dark = compute_metrics(dark[y_col], dark[score_col], threshold)[metric]
    obs_light = compute_metrics(light[y_col], light[score_col], threshold)[metric]
    obs_gap = obs_light - obs_dark

    gaps = []
    for _ in range(n_boot):
        dark_b = dark.sample(n=len(dark), replace=True, random_state=int(rng.integers(0, 2**32 - 1)))
        light_b = light.sample(n=len(light), replace=True, random_state=int(rng.integers(0, 2**32 - 1)))

        try:
            m_dark = compute_metrics(dark_b[y_col], dark_b[score_col], threshold)[metric]
            m_light = compute_metrics(light_b[y_col], light_b[score_col], threshold)[metric]
            gaps.append(m_light - m_dark)
        except Exception:
            continue

    gaps = np.asarray(gaps, dtype=float)
    gaps = gaps[~np.isnan(gaps)]

    ci_low, ci_high = np.percentile(gaps, [2.5, 97.5])

    p_left = np.mean(gaps <= 0)
    p_right = np.mean(gaps >= 0)
    p_value = 2 * min(p_left, p_right)
    p_value = min(float(p_value), 1.0)

    z = obs_gap / np.std(gaps, ddof=1) if len(gaps) > 1 and np.std(gaps, ddof=1) > 0 else np.nan

    return {
        "metric": metric,
        "dark": obs_dark,
        "light": obs_light,
        "gap_light_minus_dark": obs_gap,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "z_bootstrap": z,
        "p_value_bootstrap": p_value,
        "n_boot_valid": len(gaps),
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap light-vs-dark subgroup comparisons on BOSQUE predictions.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    models = ["resnet50", "densenet121", "mobilenetv2", "efficientnetv2b0", "vgg16"]

    rows = []

    for model in models:
        pred_files = sorted(Path("outputs/predictions").glob(f"bosque_public_predictions_{model}_seed*.csv"))

        if not pred_files:
            print(f"WARNING: no prediction files found for {model}")
            continue

        model_frames = []
        for path in pred_files:
            df = pd.read_csv(path)
            df["prediction_file"] = path.name
            model_frames.append(df)

        df = pd.concat(model_frames, ignore_index=True)

        subgroup_col = find_column(df, ["skin_group"])
        y_col = find_column(df, ["y_true", "label_binary", "label", "target"])
        score_col = find_column(df, ["y_score", "score", "pred_score", "prediction_score", "probability", "malignant_probability"])

        df = df[df[subgroup_col].isin(["dark", "light"])].copy()

        for metric in METRICS:
            out = bootstrap_gap(
                df=df,
                y_col=y_col,
                score_col=score_col,
                subgroup_col=subgroup_col,
                metric=metric,
                n_boot=args.n_boot,
                threshold=args.threshold,
                seed=args.seed,
            )
            out = {"model": model, **out}
            out["n_dark_rows"] = int((df[subgroup_col] == "dark").sum())
            out["n_light_rows"] = int((df[subgroup_col] == "light").sum())
            out["n_prediction_files"] = len(pred_files)
            rows.append(out)

    out_df = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction across all model-metric comparisons.
    out_df = out_df.sort_values("p_value_bootstrap").reset_index(drop=True)
    m = len(out_df)
    ranks = np.arange(1, m + 1)
    raw = out_df["p_value_bootstrap"].astype(float).to_numpy()

    bh = raw * m / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    out_df["p_value_fdr_bh"] = bh
    out_df["significant_fdr_0_05"] = out_df["p_value_fdr_bh"] < 0.05

    # Restore readable ordering.
    model_order = {name: i for i, name in enumerate(models)}
    metric_order = {name: i for i, name in enumerate(METRICS)}
    out_df["model_order"] = out_df["model"].map(model_order)
    out_df["metric_order"] = out_df["metric"].map(metric_order)
    out_df = out_df.sort_values(["model_order", "metric_order"]).drop(columns=["model_order", "metric_order"])

    out_path = Path("outputs/tables/bosque_light_dark_bootstrap_comparison.csv")
    out_df.to_csv(out_path, index=False)

    readable = out_df.copy()
    numeric_cols = [
        "dark", "light", "gap_light_minus_dark",
        "bootstrap_ci_low", "bootstrap_ci_high",
        "z_bootstrap", "p_value_bootstrap", "p_value_fdr_bh",
    ]
    for col in numeric_cols:
        if col in readable.columns:
            readable[col] = readable[col].astype(float).round(4)

    def stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    readable["sig_fdr"] = readable["p_value_fdr_bh"].apply(stars)

    readable_path = Path("outputs/tables/bosque_light_dark_bootstrap_comparison_readable.csv")
    readable.to_csv(readable_path, index=False)

    print("Wrote:", out_path)
    print("Wrote:", readable_path)
    print()
    print("FDR-significant gaps only:")
    cols = [
        "model", "metric", "dark", "light", "gap_light_minus_dark",
        "bootstrap_ci_low", "bootstrap_ci_high",
        "p_value_bootstrap", "p_value_fdr_bh", "sig_fdr",
    ]
    print(readable[readable["significant_fdr_0_05"]][cols].to_string(index=False))


if __name__ == "__main__":
    main()
