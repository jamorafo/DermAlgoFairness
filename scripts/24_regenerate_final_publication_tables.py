#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regenerate final publication Tables 01, 02, 07, and 08 from the
verified lesion-cluster TAC/ETC analysis.

Input
-----
outputs/tables/interval_tac_etc_by_seed.csv

Outputs
-------
outputs/publication_tables/table_01_internal_external_performance.csv
outputs/publication_tables/table_01_internal_external_performance.tex
outputs/publication_tables/table_02_bosque_subgroup_performance.csv
outputs/publication_tables/table_02_bosque_subgroup_performance.tex
outputs/publication_tables/table_07_source_pr_diagnostics.csv
outputs/publication_tables/table_07_source_pr_diagnostics.tex
outputs/publication_tables/table_08_target_pr_diagnostics.csv
outputs/publication_tables/table_08_target_pr_diagnostics.tex
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "tables" / "interval_tac_etc_by_seed.csv"
PUB = ROOT / "outputs" / "publication_tables"
PUB.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "ResNet50",
    "DenseNet121",
    "MobileNetV2",
    "EfficientNetV2B0",
    "VGG16",
]

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "ResNet50": "ResNet50",
    "densenet121": "DenseNet121",
    "DenseNet121": "DenseNet121",
    "mobilenetv2": "MobileNetV2",
    "MobileNetV2": "MobileNetV2",
    "efficientnetv2b0": "EfficientNetV2B0",
    "EfficientNetV2B0": "EfficientNetV2B0",
    "vgg16": "VGG16",
    "VGG16": "VGG16",
}

PRIMARY_METRICS = [
    "recall",
    "auc_pr",
    "f1",
    "precision",
]

SECONDARY_METRICS = [
    "accuracy",
    "specificity",
    "auc_roc",
]

METRIC_ORDER = PRIMARY_METRICS + SECONDARY_METRICS

PERFORMANCE_TABLE_ORDER = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc_roc",
    "auc_pr",
]

METRIC_LABELS = {
    "recall": "Recall / sensitivity",
    "auc_pr": "AUC-PR",
    "f1": "F1-score",
    "precision": "Precision",
    "accuracy": "Accuracy",
    "specificity": "Specificity",
    "auc_roc": "AUC-ROC",
}

TARGET_ORDER = [
    "BOSQUE overall",
    "BOSQUE light",
    "BOSQUE dark",
]

TARGET_LABELS = {
    "BOSQUE overall": "Overall",
    "BOSQUE light": "Light phototype",
    "BOSQUE dark": "Dark phototype",
}



def latex_escape(value) -> str:
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
        .replace("±", r"$\pm$")
    )


def fmt(value, digits=3) -> str:
    return f"{float(value):.{digits}f}"


def mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = PUB / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_tex(content: str, name: str) -> Path:
    path = PUB / f"{name}.tex"
    path.write_text(content, encoding="utf-8")
    print(f"Saved: {path}")
    return path


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "seed",
        "metric",
        "metric_group",
        "target_condition",
        "n_source",
        "n_source_clusters",
        "n_target",
        "source_bootstrap_unit",
        "target_bootstrap_unit",
        "n_boot_requested",
        "n_boot_valid",
        "source_performance",
        "source_ci_low",
        "source_ci_high",
        "target_performance",
        "target_ci_low",
        "target_ci_high",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Final TAC/ETC table is missing columns: {sorted(missing)}"
        )

    out = df.copy()
    out["model"] = out["model"].map(MODEL_LABELS)

    if out["model"].isna().any():
        raise RuntimeError("Unknown model labels are present.")

    if set(out["model"]) != set(MODEL_ORDER):
        raise RuntimeError(
            f"Unexpected architectures: {sorted(out['model'].unique())}"
        )

    if set(out["metric"]) != set(METRIC_ORDER):
        raise RuntimeError(
            f"Unexpected metrics: {sorted(out['metric'].unique())}"
        )

    if set(out["target_condition"]) != set(TARGET_ORDER):
        raise RuntimeError(
            "Unexpected target conditions: "
            f"{sorted(out['target_condition'].unique())}"
        )

    if len(out) != 25 * 7 * 3:
        raise RuntimeError(
            f"Expected 525 TAC/ETC rows, found {len(out)}."
        )

    if set(out["n_source"]) != {990}:
        raise RuntimeError(
            f"Unexpected source image counts: {sorted(out['n_source'].unique())}"
        )

    if set(out["n_source_clusters"]) != {747}:
        raise RuntimeError(
            "Unexpected source lesion counts: "
            f"{sorted(out['n_source_clusters'].unique())}"
        )

    expected_targets = {
        "BOSQUE overall": 151,
        "BOSQUE light": 105,
        "BOSQUE dark": 46,
    }

    for condition, expected_n in expected_targets.items():
        observed = set(
            out.loc[
                out["target_condition"] == condition,
                "n_target",
            ]
        )

        if observed != {expected_n}:
            raise RuntimeError(
                f"{condition}: expected n={expected_n}, "
                f"found {sorted(observed)}."
            )

    if set(out["n_boot_requested"]) != {10000}:
        raise RuntimeError("The table does not use 10,000 replicates.")

    if out["n_boot_valid"].min() != 10000:
        raise RuntimeError(
            "At least one analysis has fewer than 10,000 valid replicates."
        )

    if set(out["source_bootstrap_unit"]) != {"lesion"}:
        raise RuntimeError(
            "Source bootstrap unit is not consistently lesion-level."
        )

    return out


def make_table_01(df: pd.DataFrame) -> pd.DataFrame:
    source = (
        df.loc[
            df["target_condition"] == "BOSQUE overall",
            ["model", "seed", "metric", "source_performance"],
        ]
        .drop_duplicates(["model", "seed", "metric"])
        .copy()
    )

    target = (
        df.loc[
            df["target_condition"] == "BOSQUE overall",
            ["model", "seed", "metric", "target_performance"],
        ]
        .drop_duplicates(["model", "seed", "metric"])
        .copy()
    )

    if len(source) != 25 * 7:
        raise RuntimeError(
            f"Table 01 source: expected 175 rows, found {len(source)}."
        )

    if len(target) != 25 * 7:
        raise RuntimeError(
            f"Table 01 target: expected 175 rows, found {len(target)}."
        )

    rows = []

    for model in MODEL_ORDER:
        for dataset, data, value_column in [
            (
                "HAM10000 internal",
                source,
                "source_performance",
            ),
            (
                "BOSQUE external",
                target,
                "target_performance",
            ),
        ]:
            row = {
                "model": model,
                "dataset": dataset,
            }

            model_data = data[data["model"] == model]

            for metric in PERFORMANCE_TABLE_ORDER:
                values = model_data.loc[
                    model_data["metric"] == metric,
                    value_column,
                ]

                if len(values) != 5:
                    raise RuntimeError(
                        f"{model}, {dataset}, {metric}: "
                        f"expected five seeds, found {len(values)}."
                    )

                row[metric] = mean_sd(values)

            rows.append(row)

    return pd.DataFrame(rows)


def latex_table_01(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Internal source and external target performance "
            r"across five independently randomized training runs.}"
        ),
        r"\label{tab:internal-external-performance}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Architecture} & \textbf{Evaluation ORP} & "
            r"\textbf{Acc.} & \textbf{Prec.} & \textbf{Recall} & "
            r"\textbf{Spec.} & \textbf{F1} & "
            r"\textbf{AUC-ROC} & \textbf{AUC-PR} \\"
        ),
        r"\midrule",
    ]

    previous_model = None

    for _, row in table.iterrows():
        model = latex_escape(row["model"])
        model_cell = model if model != previous_model else ""

        lines.append(
            f"{model_cell} & "
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['accuracy'])} & "
            f"{latex_escape(row['precision'])} & "
            f"{latex_escape(row['recall'])} & "
            f"{latex_escape(row['specificity'])} & "
            f"{latex_escape(row['f1'])} & "
            f"{latex_escape(row['auc_roc'])} & "
            f"{latex_escape(row['auc_pr'])} \\\\"
        )

        previous_model = model

        if row["dataset"] == "BOSQUE external":
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: Values are mean $\pm$ standard deviation across "
                r"five independently randomized training runs. HAM10000 "
                r"internal denotes the fixed lesion-grouped held-out source "
                r"ORP ($990$ images from $747$ lesions). BOSQUE external "
                r"denotes the external target ORP ($n=151$). Point estimates "
                r"are computed for the malignant class under the locked "
                r"binary diagnostic task."
            ),
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def make_table_02(df: pd.DataFrame) -> pd.DataFrame:
    subgroup_conditions = [
        ("BOSQUE light", "light", 105),
        ("BOSQUE dark", "dark", 46),
    ]

    rows = []

    for model in MODEL_ORDER:
        for condition, subgroup, expected_n in subgroup_conditions:
            data = df[
                (df["model"] == model)
                & (df["target_condition"] == condition)
            ]

            row = {
                "model": model,
                "subgroup": subgroup,
                "n": expected_n,
            }

            for metric in PERFORMANCE_TABLE_ORDER:
                values = data.loc[
                    data["metric"] == metric,
                    "target_performance",
                ]

                if len(values) != 5:
                    raise RuntimeError(
                        f"{model}, {condition}, {metric}: "
                        f"expected five seeds, found {len(values)}."
                    )

                row[metric] = mean_sd(values)

            rows.append(row)

    return pd.DataFrame(rows)


def latex_table_02(table: pd.DataFrame) -> str:
    subgroup_labels = {
        "light": "Light phototype",
        "dark": "Dark phototype",
    }

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{BOSQUE target-side performance by phototype "
            r"group across five independently randomized training runs.}"
        ),
        r"\label{tab:bosque-subgroup-performance}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Architecture} & \textbf{Subgroup} & "
            r"\textbf{$n$} & \textbf{Acc.} & \textbf{Prec.} & "
            r"\textbf{Recall} & \textbf{Spec.} & \textbf{F1} & "
            r"\textbf{AUC-ROC} & \textbf{AUC-PR} \\"
        ),
        r"\midrule",
    ]

    previous_model = None

    for _, row in table.iterrows():
        model = latex_escape(row["model"])
        model_cell = model if model != previous_model else ""

        lines.append(
            f"{model_cell} & "
            f"{latex_escape(subgroup_labels[row['subgroup']])} & "
            f"{int(row['n'])} & "
            f"{latex_escape(row['accuracy'])} & "
            f"{latex_escape(row['precision'])} & "
            f"{latex_escape(row['recall'])} & "
            f"{latex_escape(row['specificity'])} & "
            f"{latex_escape(row['f1'])} & "
            f"{latex_escape(row['auc_roc'])} & "
            f"{latex_escape(row['auc_pr'])} \\\\"
        )

        previous_model = model

        if row["subgroup"] == "dark":
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: Values are mean $\pm$ standard deviation across "
                r"five independently randomized training runs. The target "
                r"ORPs contain $105$ light-phototype and $46$ "
                r"dark-phototype images. BOSQUE contains one image per "
                r"lesion. These summaries describe performance under the "
                r"observed BOSQUE subgroup conditions and do not by "
                r"themselves establish population-level generalizability."
            ),
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def make_table_07(df: pd.DataFrame) -> pd.DataFrame:
    source = (
        df.loc[
            df["target_condition"] == "BOSQUE overall",
            [
                "model",
                "seed",
                "metric",
                "n_source",
                "n_source_clusters",
                "source_performance",
                "source_ci_low",
                "source_ci_high",
            ],
        ]
        .drop_duplicates(["model", "seed", "metric"])
        .copy()
    )

    if len(source) != 25 * 7:
        raise RuntimeError(
            f"Table 07: expected 175 source rows, found {len(source)}."
        )

    source["half_width"] = (
        source["source_ci_high"] - source["source_ci_low"]
    ) / 2

    rows = []

    for metric in METRIC_ORDER:
        data = source[source["metric"] == metric]

        rows.append(
            {
                "Metric": METRIC_LABELS[metric],
                "n_model_seed": len(data),
                "n_source_images": int(data["n_source"].iloc[0]),
                "n_source_lesions": int(
                    data["n_source_clusters"].iloc[0]
                ),
                "mean_source": data["source_performance"].mean(),
                "sd_across_systems": data["source_performance"].std(
                    ddof=1
                ),
                "mean_half_width": data["half_width"].mean(),
                "max_half_width": data["half_width"].max(),
            }
        )

    return pd.DataFrame(rows)


def latex_table_07(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Finite-ORP precision of HAM10000 held-out "
            r"source performance estimates.}"
        ),
        r"\label{tab:source-pr-diagnostics}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Metric} & \textbf{$n$ systems} & "
            r"\textbf{Images} & \textbf{Lesions} & "
            r"\textbf{Mean} & \textbf{SD systems} & "
            r"\textbf{Mean HW} & \textbf{Max HW} \\"
        ),
        r"\midrule",
    ]

    for _, row in table.iterrows():
        lines.append(
            f"{latex_escape(row['Metric'])} & "
            f"{int(row['n_model_seed'])} & "
            f"{int(row['n_source_images'])} & "
            f"{int(row['n_source_lesions'])} & "
            f"{fmt(row['mean_source'])} & "
            f"{fmt(row['sd_across_systems'])} & "
            f"{fmt(row['mean_half_width'])} & "
            f"{fmt(row['max_half_width'])} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: Each row summarizes 25 locked "
                r"architecture--run systems evaluated on the fixed "
                r"HAM10000 source ORP of $990$ images from $747$ lesions. "
                r"HW denotes the half-width of the 95\% percentile-bootstrap "
                r"interval. Source intervals use 10{,}000 lesion-cluster "
                r"bootstrap replicates, preserving all images belonging to "
                r"a sampled lesion. SD systems describes variation across "
                r"architectures and training runs and is not a sampling "
                r"standard error. Interval width is reported "
                r"descriptively without imposing an additional post hoc "
                r"precision threshold. These summaries do not establish "
                r"clinical adequacy, bias control, or population "
                r"representativity."
            ),
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def make_table_08(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in METRIC_ORDER:
        for condition in TARGET_ORDER:
            data = df[
                (df["metric"] == metric)
                & (df["target_condition"] == condition)
            ]

            if len(data) != 25:
                raise RuntimeError(
                    f"{metric}, {condition}: expected 25 rows, "
                    f"found {len(data)}."
                )

            half_width = (
                data["target_ci_high"] - data["target_ci_low"]
            ) / 2

            rows.append(
                {
                    "Metric": METRIC_LABELS[metric],
                    "Target condition": TARGET_LABELS[condition],
                    "n_model_seed": len(data),
                    "n_target": int(data["n_target"].iloc[0]),
                    "mean_target": data[
                        "target_performance"
                    ].mean(),
                    "sd_across_systems": data[
                        "target_performance"
                    ].std(ddof=1),
                    "mean_half_width": half_width.mean(),
                    "max_half_width": half_width.max(),
                }
            )

    return pd.DataFrame(rows)


def latex_table_08(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Finite-ORP precision of BOSQUE overall and "
            r"phototype-specific target performance estimates.}"
        ),
        r"\label{tab:target-pr-diagnostics}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Metric} & \textbf{Target} & "
            r"\textbf{$n$ systems} & \textbf{$n_T$} & "
            r"\textbf{Mean} & \textbf{SD systems} & "
            r"\textbf{Mean HW} & \textbf{Max HW} \\"
        ),
        r"\midrule",
    ]

    previous_metric = None

    for _, row in table.iterrows():
        metric = latex_escape(row["Metric"])
        metric_cell = metric if metric != previous_metric else ""

        lines.append(
            f"{metric_cell} & "
            f"{latex_escape(row['Target condition'])} & "
            f"{int(row['n_model_seed'])} & "
            f"{int(row['n_target'])} & "
            f"{fmt(row['mean_target'])} & "
            f"{fmt(row['sd_across_systems'])} & "
            f"{fmt(row['mean_half_width'])} & "
            f"{fmt(row['max_half_width'])} \\\\"
        )

        previous_metric = metric

        if row["Target condition"] == "Dark phototype":
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize",
            (
                r"Notes: Each row summarizes 25 locked "
                r"architecture--run systems. The BOSQUE ORPs contain "
                r"$151$ images overall, $105$ light-phototype images, "
                r"and $46$ dark-phototype images. HW denotes the "
                r"half-width of the 95\% percentile-bootstrap interval. "
                r"Target intervals use 10{,}000 image-level bootstrap "
                r"replicates; because BOSQUE contains one image per "
                r"lesion, this is also a lesion-level resampling scheme. "
                r"SD systems describes architecture and training-run "
                r"variation and is not a sampling standard error. "
                r"Interval width is reported descriptively without imposing "
                r"an additional post hoc precision threshold. These summaries "
                r"do not establish clinical adequacy or generalizability "
                r"beyond BOSQUE."
            ),
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing final analysis table: {INPUT}")

    df = validate_input(pd.read_csv(INPUT))

    table_01 = make_table_01(df)
    save_csv(
        table_01,
        "table_01_internal_external_performance",
    )
    save_tex(
        latex_table_01(table_01),
        "table_01_internal_external_performance",
    )

    table_02 = make_table_02(df)
    save_csv(
        table_02,
        "table_02_bosque_subgroup_performance",
    )
    save_tex(
        latex_table_02(table_02),
        "table_02_bosque_subgroup_performance",
    )

    table_07 = make_table_07(df)
    save_csv(
        table_07,
        "table_07_source_pr_diagnostics",
    )
    save_tex(
        latex_table_07(table_07),
        "table_07_source_pr_diagnostics",
    )

    table_08 = make_table_08(df)
    save_csv(
        table_08,
        "table_08_target_pr_diagnostics",
    )
    save_tex(
        latex_table_08(table_08),
        "table_08_target_pr_diagnostics",
    )

    print()
    print("FINAL TABLES 01, 02, 07, AND 08 REGENERATED")


if __name__ == "__main__":
    main()
