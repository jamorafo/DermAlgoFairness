#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Architecture-level BH sensitivity analysis for BOSQUE light--dark gaps.

For each bootstrap replicate:

1. BOSQUE light and dark images are resampled separately.
2. The same resampled image indices are applied to all five seeds and all five
   architectures.
3. Seed-specific light-minus-dark metric gaps are calculated.
4. The five gaps are averaged within architecture.
5. A two-sided null-centred bootstrap p-value is calculated for each
   architecture--metric contrast.
6. Benjamini--Hochberg adjustment is applied to the prespecified family of
   5 architectures x 4 primary metrics = 20 tests.

Benjamini--Yekutieli adjusted p-values are also reported as a conservative
dependence sensitivity analysis.

Inference is conditional on the five observed training seeds and on
image-level independence within the observed BOSQUE ORP.

Inputs:
  outputs/logs/bosque_gap_prediction_manifest.csv

Outputs:
  outputs/tables/bosque_architecture_level_bh_sensitivity.csv
  outputs/publication_tables/table_09_architecture_level_bh_sensitivity.csv
  outputs/publication_tables/table_09_architecture_level_bh_sensitivity.tex
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from dermalgo.seeds import get_analysis_seed, get_training_seeds

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]

LOGS = ROOT / "outputs" / "logs"
TABLES = ROOT / "outputs" / "tables"
PUB = ROOT / "outputs" / "publication_tables"

TABLES.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)

MANIFEST_INPUT = (
    LOGS / "bosque_gap_prediction_manifest.csv"
)

MODELS = [
    "resnet50",
    "densenet121",
    "mobilenetv2",
    "efficientnetv2b0",
    "vgg16",
]

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenetv2": "MobileNetV2",
    "efficientnetv2b0": "EfficientNetV2B0",
    "vgg16": "VGG16",
}

SEEDS = get_training_seeds()

PRIMARY_METRICS = [
    "recall",
    "auc_pr",
    "f1",
    "precision",
]

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
}

ID_CANDIDATES = [
    "image_id",
    "isic_id",
    "filename",
    "file_name",
    "image_name",
    "filepath",
    "file_path",
    "image_path",
    "path",
    "image",
]

Y_CANDIDATES = [
    "y_true",
    "label_binary",
    "label",
    "target",
]

SCORE_CANDIDATES = [
    "y_score",
    "score",
    "pred_score",
    "prediction_score",
    "probability",
    "malignant_probability",
]

GROUP_CANDIDATES = [
    "skin_group",
]


def find_column(frame, candidates, purpose):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        f"No {purpose} column found. "
        f"Candidates: {candidates}. "
        f"Available columns: {frame.columns.tolist()}"
    )


def choose_common_id_column(raw_frames):
    for candidate in ID_CANDIDATES:
        valid = True

        for frame in raw_frames.values():
            if candidate not in frame.columns:
                valid = False
                break

            values = frame[candidate]

            if values.isna().any():
                valid = False
                break

            if values.astype(str).duplicated().any():
                valid = False
                break

        if valid:
            return candidate

    return None


def normalize_frame(frame, id_column):
    y_column = find_column(
        frame,
        Y_CANDIDATES,
        "outcome",
    )

    score_column = find_column(
        frame,
        SCORE_CANDIDATES,
        "score",
    )

    group_column = find_column(
        frame,
        GROUP_CANDIDATES,
        "phototype-group",
    )

    normalized = pd.DataFrame(
        {
            "y_true": pd.to_numeric(
                frame[y_column],
                errors="raise",
            ).astype(int),
            "y_score": pd.to_numeric(
                frame[score_column],
                errors="raise",
            ).astype(float),
            "skin_group": (
                frame[group_column]
                .astype(str)
                .str.strip()
                .str.lower()
            ),
        }
    )

    if id_column is not None:
        normalized["alignment_id"] = (
            frame[id_column].astype(str)
        )

    if normalized.isna().any().any():
        raise ValueError(
            "A prediction file contains missing normalized values."
        )

    if not set(
        normalized["y_true"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "A prediction file contains non-binary outcomes."
        )

    return normalized


def load_aligned_predictions(manifest):
    required_manifest = {
        "model",
        "seed",
        "prediction_file",
    }

    missing = (
        required_manifest
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    if len(manifest) != 25:
        raise ValueError(
            f"Expected 25 manifest rows, found {len(manifest)}."
        )

    if manifest.duplicated(
        ["model", "seed"]
    ).any():
        raise ValueError(
            "Manifest contains duplicate model--seed rows."
        )

    raw_frames = {}

    for row in manifest.itertuples(index=False):
        model = str(row.model)
        seed = int(row.seed)

        path = ROOT / str(row.prediction_file)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {path}"
            )

        frame = pd.read_csv(path)

        if len(frame) != 151:
            raise ValueError(
                f"{path} contains {len(frame)} rows; "
                "expected 151."
            )

        raw_frames[(model, seed)] = frame

    expected_keys = {
        (model, seed)
        for model in MODELS
        for seed in SEEDS
    }

    if set(raw_frames) != expected_keys:
        missing_keys = (
            expected_keys - set(raw_frames)
        )
        extra_keys = (
            set(raw_frames) - expected_keys
        )

        raise ValueError(
            "Unexpected canonical file set. "
            f"Missing: {sorted(missing_keys)}; "
            f"extra: {sorted(extra_keys)}."
        )

    id_column = choose_common_id_column(
        raw_frames
    )

    normalized = {
        key: normalize_frame(
            frame,
            id_column,
        )
        for key, frame in raw_frames.items()
    }

    reference_key = (MODELS[0], SEEDS[0])
    reference = normalized[reference_key].copy()

    if id_column is not None:
        reference_ids = (
            reference["alignment_id"]
            .astype(str)
            .tolist()
        )

        reference_id_set = set(reference_ids)

        aligned = {}

        for key, frame in normalized.items():
            frame_ids = (
                frame["alignment_id"]
                .astype(str)
            )

            if set(frame_ids) != reference_id_set:
                raise ValueError(
                    f"{key}: image identifiers differ "
                    "from the reference system."
                )

            indexed = (
                frame.assign(
                    alignment_id=frame_ids
                )
                .set_index("alignment_id")
                .loc[reference_ids]
                .reset_index()
            )

            aligned[key] = indexed

        alignment_mode = (
            f"unique identifier: {id_column}"
        )

    else:
        aligned = {
            key: frame.reset_index(drop=True)
            for key, frame in normalized.items()
        }

        alignment_mode = (
            "validated common row order; "
            "no unique identifier column available"
        )

    reference = aligned[reference_key]

    reference_y = (
        reference["y_true"]
        .to_numpy(dtype=int)
    )

    reference_group = (
        reference["skin_group"]
        .to_numpy(dtype=str)
    )

    for key, frame in aligned.items():
        observed_y = (
            frame["y_true"]
            .to_numpy(dtype=int)
        )

        observed_group = (
            frame["skin_group"]
            .to_numpy(dtype=str)
        )

        if not np.array_equal(
            observed_y,
            reference_y,
        ):
            raise ValueError(
                f"{key}: outcomes are not aligned "
                "with the reference system."
            )

        if not np.array_equal(
            observed_group,
            reference_group,
        ):
            raise ValueError(
                f"{key}: subgroup labels are not aligned "
                "with the reference system."
            )

    group_counts = (
        pd.Series(reference_group)
        .value_counts()
        .to_dict()
    )

    if group_counts.get("light", 0) != 105:
        raise ValueError(
            "Expected 105 light images, found "
            f"{group_counts.get('light', 0)}."
        )

    if group_counts.get("dark", 0) != 46:
        raise ValueError(
            "Expected 46 dark images, found "
            f"{group_counts.get('dark', 0)}."
        )

    scores = {
        key: frame["y_score"].to_numpy(
            dtype=float
        )
        for key, frame in aligned.items()
    }

    return (
        reference_y,
        reference_group,
        scores,
        alignment_mode,
        id_column,
    )


def compute_primary_metrics(
    y_true,
    y_score,
    threshold,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    y_pred = (
        y_score >= threshold
    ).astype(int)

    metrics = {
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "auc_pr": np.nan,
    }

    if len(np.unique(y_true)) == 2:
        metrics["auc_pr"] = (
            average_precision_score(
                y_true,
                y_score,
            )
        )

    return metrics


def adjust_pvalues(
    raw_pvalues,
    dependence_factor=1.0,
):
    raw = np.asarray(
        raw_pvalues,
        dtype=float,
    )

    if np.isnan(raw).any():
        raise ValueError(
            "Cannot adjust missing p-values."
        )

    number_tests = len(raw)
    order = np.argsort(raw)

    sorted_raw = raw[order]

    ranks = np.arange(
        1,
        number_tests + 1,
        dtype=float,
    )

    adjusted_sorted = (
        sorted_raw
        * number_tests
        * dependence_factor
        / ranks
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted = np.empty_like(
        adjusted_sorted
    )

    adjusted[order] = adjusted_sorted

    return adjusted


def fmt(value, digits=3):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_p(value):
    if pd.isna(value):
        return ""

    value = float(value)

    if value < 0.0001:
        return r"$<0.0001$"

    return f"{value:.4f}"


def latex_escape(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def build_latex_table(publication):
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Architecture-level multiplicity-adjusted "
            r"BOSQUE light--dark subgroup sensitivity analysis.}"
        ),
        r"\label{tab:architecture-bh-sensitivity}",
        r"\setlength{\tabcolsep}{3.3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        (
            r"\textbf{Metric} & "
            r"\textbf{Architecture} & "
            r"\textbf{Mean gap} & "
            r"\textbf{SD seeds} & "
            r"\textbf{95\% CI} & "
            r"\textbf{$p$} & "
            r"\textbf{$p_{\mathrm{BH}}$} & "
            r"\textbf{$p_{\mathrm{BY}}$} & "
            r"\textbf{BH $<0.05$} \\"
        ),
        r"\midrule",
    ]

    previous_metric = None

    for _, row in publication.iterrows():
        metric = latex_escape(row["Metric"])

        metric_cell = (
            metric
            if metric != previous_metric
            else ""
        )

        interval = (
            f"[{fmt(row['bootstrap_ci_low'])}, "
            f"{fmt(row['bootstrap_ci_high'])}]"
        )

        lines.append(
            f"{metric_cell} & "
            f"{latex_escape(row['Architecture'])} & "
            f"{fmt(row['observed_mean_gap'])} & "
            f"{fmt(row['sd_seed_gaps'])} & "
            f"{interval} & "
            f"{fmt_p(row['p_value_raw'])} & "
            f"{fmt_p(row['p_value_bh'])} & "
            f"{fmt_p(row['p_value_by'])} & "
            f"{latex_escape(row['bh_significant'])} \\\\"
        )

        previous_metric = metric

        if row["Architecture"] == "VGG16":
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: The architecture-level estimand is the mean "
                r"light-minus-dark performance gap across the five locked "
                r"seed-specific systems. Positive values indicate higher "
                r"performance in the light-phototype group. BOSQUE light "
                r"and dark images were resampled separately; within each "
                r"bootstrap replicate, the same sampled image indices were "
                r"applied to all 25 systems. The 95\% intervals are "
                r"percentile-bootstrap intervals. Two-sided $p$-values were "
                r"obtained from the null-centred bootstrap distribution. "
                r"Benjamini--Hochberg adjustment was applied to the "
                r"prespecified family of 20 architecture--primary-metric "
                r"tests. Benjamini--Yekutieli adjusted values are reported "
                r"as a conservative dependence sensitivity analysis. "
                r"Inference is conditional on the five observed training "
                r"seeds and on image-level independence within the BOSQUE "
                r"ORP. This secondary analysis does not replace the "
                r"locked-system TAC, ETC, benchmark-preservation, or "
                r"interval results."
            ),
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Architecture-level BH sensitivity analysis."
        )
    )

    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=get_analysis_seed("architecture_level_bh"),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    if args.n_boot < 1000:
        print(
            "WARNING: fewer than 1000 bootstrap "
            "replicates were requested."
        )

    if not MANIFEST_INPUT.exists():
        raise FileNotFoundError(
            f"Missing manifest: {MANIFEST_INPUT}"
        )

    manifest = pd.read_csv(
        MANIFEST_INPUT
    )

    (
        y_true,
        skin_group,
        scores,
        alignment_mode,
        id_column,
    ) = load_aligned_predictions(
        manifest
    )

    light_positions = np.flatnonzero(
        skin_group == "light"
    )

    dark_positions = np.flatnonzero(
        skin_group == "dark"
    )

    y_light = y_true[light_positions]
    y_dark = y_true[dark_positions]

    score_light = {
        key: values[light_positions]
        for key, values in scores.items()
    }

    score_dark = {
        key: values[dark_positions]
        for key, values in scores.items()
    }

    test_keys = [
        (metric, model)
        for metric in PRIMARY_METRICS
        for model in MODELS
    ]

    observed_rows = []
    observed_statistics = {}

    for metric, model in test_keys:
        seed_gaps = []

        for seed in SEEDS:
            light_metrics = (
                compute_primary_metrics(
                    y_light,
                    score_light[(model, seed)],
                    args.threshold,
                )
            )

            dark_metrics = (
                compute_primary_metrics(
                    y_dark,
                    score_dark[(model, seed)],
                    args.threshold,
                )
            )

            gap = (
                light_metrics[metric]
                - dark_metrics[metric]
            )

            if not np.isfinite(gap):
                raise RuntimeError(
                    f"Observed {metric} gap is undefined "
                    f"for {model}, seed {seed}."
                )

            seed_gaps.append(gap)

        seed_gaps = np.asarray(
            seed_gaps,
            dtype=float,
        )

        observed_mean = float(
            seed_gaps.mean()
        )

        observed_statistics[
            (metric, model)
        ] = observed_mean

        observed_rows.append(
            {
                "metric": metric,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_seeds": len(seed_gaps),
                "observed_mean_gap": observed_mean,
                "sd_seed_gaps": float(
                    seed_gaps.std(ddof=1)
                ),
                "min_seed_gap": float(
                    seed_gaps.min()
                ),
                "max_seed_gap": float(
                    seed_gaps.max()
                ),
                "n_positive_seed_gaps": int(
                    (seed_gaps > 0).sum()
                ),
            }
        )

    bootstrap = np.full(
        (
            args.n_boot,
            len(test_keys),
        ),
        np.nan,
        dtype=float,
    )

    rng = np.random.default_rng(
        args.seed
    )

    for replicate in range(args.n_boot):
        sampled_light = rng.integers(
            0,
            len(y_light),
            size=len(y_light),
        )

        sampled_dark = rng.integers(
            0,
            len(y_dark),
            size=len(y_dark),
        )

        bootstrap_statistics = {}

        for model in MODELS:
            gaps_by_metric = {
                metric: []
                for metric in PRIMARY_METRICS
            }

            for seed in SEEDS:
                light_metrics = (
                    compute_primary_metrics(
                        y_light[sampled_light],
                        score_light[
                            (model, seed)
                        ][sampled_light],
                        args.threshold,
                    )
                )

                dark_metrics = (
                    compute_primary_metrics(
                        y_dark[sampled_dark],
                        score_dark[
                            (model, seed)
                        ][sampled_dark],
                        args.threshold,
                    )
                )

                for metric in PRIMARY_METRICS:
                    gap = (
                        light_metrics[metric]
                        - dark_metrics[metric]
                    )

                    if np.isfinite(gap):
                        gaps_by_metric[
                            metric
                        ].append(gap)

            for metric in PRIMARY_METRICS:
                values = np.asarray(
                    gaps_by_metric[metric],
                    dtype=float,
                )

                if len(values) == len(SEEDS):
                    bootstrap_statistics[
                        (metric, model)
                    ] = float(
                        values.mean()
                    )

        for column, key in enumerate(test_keys):
            if key in bootstrap_statistics:
                bootstrap[
                    replicate,
                    column,
                ] = bootstrap_statistics[key]

        if (
            (replicate + 1) % 500 == 0
            or replicate + 1 == args.n_boot
        ):
            print(
                "Bootstrap replicate "
                f"{replicate + 1}/{args.n_boot}"
            )

    results = pd.DataFrame(
        observed_rows
    )

    bootstrap_rows = []

    for column, (metric, model) in enumerate(
        test_keys
    ):
        draws = bootstrap[:, column]
        draws = draws[np.isfinite(draws)]

        minimum_valid = int(
            np.ceil(0.90 * args.n_boot)
        )

        if len(draws) < minimum_valid:
            raise RuntimeError(
                f"{model}, {metric}: only "
                f"{len(draws)} valid bootstrap draws; "
                f"minimum required is {minimum_valid}."
            )

        observed = observed_statistics[
            (metric, model)
        ]

        ci_low, ci_high = np.percentile(
            draws,
            [2.5, 97.5],
        )

        null_centred_deviations = (
            draws - observed
        )

        extreme = int(
            (
                np.abs(
                    null_centred_deviations
                )
                >= abs(observed)
            ).sum()
        )

        raw_p = (
            1 + extreme
        ) / (
            len(draws) + 1
        )

        bootstrap_rows.append(
            {
                "metric": metric,
                "model": model,
                "bootstrap_mean_gap": float(
                    draws.mean()
                ),
                "bootstrap_ci_low": float(
                    ci_low
                ),
                "bootstrap_ci_high": float(
                    ci_high
                ),
                "n_boot_requested": args.n_boot,
                "n_boot_valid": len(draws),
                "p_value_raw": float(raw_p),
            }
        )

    bootstrap_summary = pd.DataFrame(
        bootstrap_rows
    )

    results = results.merge(
        bootstrap_summary,
        on=["metric", "model"],
        how="left",
        validate="one_to_one",
    )

    if len(results) != 20:
        raise RuntimeError(
            f"Expected 20 tests, found {len(results)}."
        )

    raw_pvalues = results[
        "p_value_raw"
    ].to_numpy(dtype=float)

    results["p_value_bh"] = adjust_pvalues(
        raw_pvalues,
        dependence_factor=1.0,
    )

    harmonic_number = float(
        np.sum(
            1 / np.arange(
                1,
                len(results) + 1,
                dtype=float,
            )
        )
    )

    results["p_value_by"] = adjust_pvalues(
        raw_pvalues,
        dependence_factor=harmonic_number,
    )

    results["bh_significant_0_05"] = (
        results["p_value_bh"]
        <= args.alpha
    )

    results["by_significant_0_05"] = (
        results["p_value_by"]
        <= args.alpha
    )

    results["ci_excludes_zero"] = (
        (results["bootstrap_ci_low"] > 0)
        | (results["bootstrap_ci_high"] < 0)
    )

    results["n_light"] = len(
        light_positions
    )

    results["n_dark"] = len(
        dark_positions
    )

    results["threshold"] = (
        args.threshold
    )

    results["bootstrap_seed"] = (
        args.seed
    )

    results["alpha"] = (
        args.alpha
    )

    results["multiplicity_family"] = (
        "5 architectures x 4 primary metrics"
    )

    results["hypothesis"] = (
        "two-sided architecture mean light-dark gap = 0"
    )

    results["alignment_mode"] = (
        alignment_mode
    )

    results["image_id_column"] = (
        id_column
        if id_column is not None
        else ""
    )

    model_order = {
        model: index
        for index, model in enumerate(MODELS)
    }

    metric_order = {
        metric: index
        for index, metric in enumerate(
            PRIMARY_METRICS
        )
    }

    results["_metric_order"] = (
        results["metric"].map(metric_order)
    )

    results["_model_order"] = (
        results["model"].map(model_order)
    )

    results = (
        results.sort_values(
            [
                "_metric_order",
                "_model_order",
            ]
        )
        .drop(
            columns=[
                "_metric_order",
                "_model_order",
            ]
        )
        .reset_index(drop=True)
    )

    raw_path = (
        TABLES
        / "bosque_architecture_level_bh_sensitivity.csv"
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    publication = pd.DataFrame(
        {
            "Metric": results[
                "metric"
            ].map(METRIC_LABELS),
            "Architecture": results[
                "model"
            ].map(MODEL_LABELS),
            "n_seeds": results["n_seeds"],
            "observed_mean_gap": results[
                "observed_mean_gap"
            ],
            "sd_seed_gaps": results[
                "sd_seed_gaps"
            ],
            "bootstrap_ci_low": results[
                "bootstrap_ci_low"
            ],
            "bootstrap_ci_high": results[
                "bootstrap_ci_high"
            ],
            "n_boot_valid": results[
                "n_boot_valid"
            ],
            "p_value_raw": results[
                "p_value_raw"
            ],
            "p_value_bh": results[
                "p_value_bh"
            ],
            "p_value_by": results[
                "p_value_by"
            ],
            "bh_significant": np.where(
                results[
                    "bh_significant_0_05"
                ],
                "Yes",
                "No",
            ),
            "by_significant": np.where(
                results[
                    "by_significant_0_05"
                ],
                "Yes",
                "No",
            ),
        }
    )

    publication_csv = (
        PUB
        / "table_09_architecture_level_bh_sensitivity.csv"
    )

    publication_tex = (
        PUB
        / "table_09_architecture_level_bh_sensitivity.tex"
    )

    publication.to_csv(
        publication_csv,
        index=False,
    )

    publication_tex.write_text(
        build_latex_table(publication),
        encoding="utf-8",
    )

    print()
    print("Alignment:", alignment_mode)
    print("Saved:", raw_path)
    print("Saved:", publication_csv)
    print("Saved:", publication_tex)
    print()

    print(
        publication.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
