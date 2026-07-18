#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Predictive Representativity documentation tables.

Outputs:
  outputs/publication_tables/table_05_orp_pr_assessment.csv
  outputs/publication_tables/table_05_orp_pr_assessment.tex
  outputs/publication_tables/table_06_uncertainty_sources.csv
  outputs/publication_tables/table_06_uncertainty_sources.tex

These tables document the ORP/PR layer that precedes TAC/ETC.
They are intentionally conceptual-documentation tables, not additional
performance-estimation tables.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "outputs" / "publication_tables"
PUB.mkdir(parents=True, exist_ok=True)


def latex_escape(value):
    """Minimal LaTeX escaping for table text."""
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


def write_table(path, content):
    path.write_text(content, encoding="utf-8")
    print(f"Saved: {path}")


def make_table_05_dataframe():
    rows = [
        {
            "Claim": "Source overall performance",
            "ORP": "HAM10000 internal test split",
            "Estimand": r"$\theta_S^M(f_{a,s})$",
            "Sampling or evaluation basis": (
                "Held-out source evaluation split from the HAM10000 empirical frame; "
                "non-augmented test images only."
            ),
            "PR status": "Conditionally PR-adequate",
            "Scope of inference": (
                "Overall performance under the HAM10000 internal evaluation condition."
            ),
            "Limitation": (
                "Does not establish performance in a broader clinical deployment population."
            ),
        },
        {
            "Claim": "Source subgroup performance",
            "ORP": "HAM10000 internal test split",
            "Estimand": r"$\theta_{S,g}^M(f_{a,s})$",
            "Sampling or evaluation basis": (
                "Source subgroup labels by skin phototype are unavailable or not reliable "
                "for the intended subgroup estimand."
            ),
            "PR status": "PR-inadequate",
            "Scope of inference": "Not supported.",
            "Limitation": (
                "Direct source subgroup TAC/ETC and source-subgroup transportability "
                "are not estimable."
            ),
        },
        {
            "Claim": "Target overall performance",
            "ORP": "BOSQUE external test set",
            "Estimand": r"$\theta_T^M(f_{a,s})$",
            "Sampling or evaluation basis": (
                "External target-side evaluation ORP with non-augmented target images."
            ),
            "PR status": "PR-adequate with limited scope",
            "Scope of inference": "Overall performance under the BOSQUE evaluation condition.",
            "Limitation": (
                "Does not by itself support national or regional deployment inference."
            ),
        },
        {
            "Claim": "Target subgroup performance",
            "ORP": "BOSQUE phototype groups",
            "Estimand": r"$\theta_{T,g}^M(f_{a,s})$",
            "Sampling or evaluation basis": (
                "Target-side phototype-labelled ORP; subgroup estimates computed within "
                "BOSQUE light and dark groups."
            ),
            "PR status": "PR-adequate as diagnostic evidence",
            "Scope of inference": (
                "Phototype-specific performance under the BOSQUE subgroup evaluation condition."
            ),
            "Limitation": (
                "Subgroup sample size and finite-ORP uncertainty limit formal adjudication."
            ),
        },
        {
            "Claim": "Benchmark-preservation subgroup comparison",
            "ORP": "HAM10000 overall source ORP and BOSQUE subgroup ORP",
            "Estimand": r"$\Delta_{S\to T,g}^{M,\mathrm{bench}}(f_{a,s})$",
            "Sampling or evaluation basis": (
                "Overall source performance is compared with target subgroup performance."
            ),
            "PR status": "PR-adequate only as benchmark preservation",
            "Scope of inference": (
                "Whether target subgroup performance preserves the overall source benchmark."
            ),
            "Limitation": (
                "Not equivalent to full source-subgroup-to-target-subgroup transportability."
            ),
        },
        {
            "Claim": "Population-level deployment performance",
            "ORP": "BOSQUE or HAM10000 alone",
            "Estimand": r"$\theta_{\mathrm{deploy}}^M(f_{a,s})$",
            "Sampling or evaluation basis": (
                "No documented probability sampling frame for a broader deployment population."
            ),
            "PR status": "Evidentially unresolved / not supported",
            "Scope of inference": "Not supported by the available ORPs alone.",
            "Limitation": (
                "Requires a prospectively defined deployment ORP or defensible population "
                "sampling design."
            ),
        },
    ]

    return pd.DataFrame(rows)


def make_table_06_dataframe():
    rows = [
        {
            "Uncertainty source": "Source ORP uncertainty",
            "Applies to": "HAM10000 internal test ORP",
            "Meaning": (
                "Finite evaluation uncertainty for a locked model under the source "
                "evaluation condition."
            ),
            "Treatment": (
                "Metric-specific confidence intervals or bootstrap intervals computed on "
                "non-augmented source test images."
            ),
            "Not treated as": "Training-seed variability or deployment-population uncertainty.",
        },
        {
            "Uncertainty source": "Target ORP uncertainty",
            "Applies to": "BOSQUE overall and subgroup ORPs",
            "Meaning": (
                "Finite external evaluation uncertainty for target overall and subgroup "
                "performance."
            ),
            "Treatment": (
                "Bootstrap or metric-specific intervals computed on non-augmented BOSQUE "
                "evaluation images."
            ),
            "Not treated as": "Evidence for national or regional population representativity.",
        },
        {
            "Uncertainty source": "Training stochasticity",
            "Applies to": "CNN model-development procedure",
            "Meaning": (
                "Variation in the learned function due to initialization, minibatch ordering, "
                "augmentation, and optimization."
            ),
            "Treatment": (
                "Five independently trained locked replicas per architecture; conclusions "
                "summarized across model--seed decisions."
            ),
            "Not treated as": "Additional sampling information from the source or target ORP.",
        },
        {
            "Uncertainty source": "Training augmentation",
            "Applies to": "Training images only",
            "Meaning": (
                "Transformations used to expose the optimizer to modified versions of "
                "training images."
            ),
            "Treatment": (
                "Part of model construction; affects the learned function "
                r"$f_{a,s}$."
            ),
            "Not treated as": "Additional independent evaluation observations.",
        },
        {
            "Uncertainty source": "Training oversampling / class balancing",
            "Applies to": "Training distribution only",
            "Meaning": (
                "Modification of class frequencies presented to the learning algorithm."
            ),
            "Treatment": (
                "Part of the development procedure; documented separately from the "
                "evaluation ORP."
            ),
            "Not treated as": "A change in the source or target performance estimand.",
        },
        {
            "Uncertainty source": "Threshold dependence",
            "Applies to": "Threshold-based metrics",
            "Meaning": (
                "Sensitivity, specificity, precision, and F1 depend on a fixed operating "
                "threshold."
            ),
            "Treatment": (
                "Threshold fixed before external target evaluation; AUC-based metrics "
                "reported as threshold-free complements."
            ),
            "Not treated as": "A parameter tuned on the external target ORP.",
        },
        {
            "Uncertainty source": "Clustering / dependence",
            "Applies to": "Images from same lesion, patient, center, or device",
            "Meaning": (
                "Evaluation units may not be fully independent if multiple images share "
                "higher-level origins."
            ),
            "Treatment": (
                "Document available clustering information; use cluster bootstrap when "
                "the relevant identifiers are available."
            ),
            "Not treated as": "Ignorable if repeated units are known and identifiable.",
        },
        {
            "Uncertainty source": "Target sampling scope",
            "Applies to": "BOSQUE external ORP",
            "Meaning": (
                "The target ORP supports claims under the BOSQUE evaluation condition."
            ),
            "Treatment": (
                "Inference is restricted to the BOSQUE condition unless a broader sampling "
                "design is justified."
            ),
            "Not treated as": "Automatic generalization to a national or regional population.",
        },
    ]

    return pd.DataFrame(rows)


def build_latex_table_05(df):
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Objective Reference Point and Predictive Representativity assessment by validation claim.}"
    )
    lines.append(r"\label{tab:orp-pr-assessment}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{p{2.5cm}p{2.5cm}p{2.1cm}p{3.4cm}p{2.5cm}p{3.3cm}p{3.2cm}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Claim} & \textbf{ORP} & \textbf{Estimand} & "
        r"\textbf{Sampling/evaluation basis} & \textbf{PR status} & "
        r"\textbf{Scope of inference} & \textbf{Limitation} \\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['Claim'])} & "
            f"{latex_escape(row['ORP'])} & "
            f"{row['Estimand']} & "
            f"{latex_escape(row['Sampling or evaluation basis'])} & "
            f"{latex_escape(row['PR status'])} & "
            f"{latex_escape(row['Scope of inference'])} & "
            f"{latex_escape(row['Limitation'])} \\\\"
        )
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: PR denotes Predictive Representativity. The table separates "
        r"the evidential status of each claim before TAC/ETC decision rules are "
        r"applied. A PR-adequate claim is not necessarily adequate or transported; "
        r"it only means that the ORP, estimand, estimator, and uncertainty "
        r"procedure are aligned with the stated scope of inference."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_latex_table_06(df):
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Uncertainty sources distinguished in the ORP-based validation analysis.}"
    )
    lines.append(r"\label{tab:uncertainty-sources}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{p{3.0cm}p{3.0cm}p{4.2cm}p{4.2cm}p{3.7cm}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Uncertainty source} & \textbf{Applies to} & "
        r"\textbf{Meaning} & \textbf{Treatment} & \textbf{Not treated as} \\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['Uncertainty source'])} & "
            f"{latex_escape(row['Applies to'])} & "
            f"{latex_escape(row['Meaning'])} & "
            f"{latex_escape(row['Treatment'])} & "
            f"{latex_escape(row['Not treated as'])} \\\\"
        )
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(
        r"Notes: Evaluation uncertainty is conditional on a locked predictive "
        r"system and is estimated from the corresponding non-augmented ORP. "
        r"Training stochasticity concerns variation in the learned function across "
        r"random seeds and is summarized separately from ORP sampling uncertainty. "
        r"Training augmentation and oversampling are model-development procedures, "
        r"not mechanisms for increasing the effective evaluation sample size."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    table_05 = make_table_05_dataframe()
    table_06 = make_table_06_dataframe()

    table_05_csv = PUB / "table_05_orp_pr_assessment.csv"
    table_05_tex = PUB / "table_05_orp_pr_assessment.tex"

    table_06_csv = PUB / "table_06_uncertainty_sources.csv"
    table_06_tex = PUB / "table_06_uncertainty_sources.tex"

    table_05.to_csv(table_05_csv, index=False)
    table_06.to_csv(table_06_csv, index=False)

    write_table(table_05_tex, build_latex_table_05(table_05))
    write_table(table_06_tex, build_latex_table_06(table_06))

    print(f"Saved: {table_05_csv}")
    print(f"Saved: {table_06_csv}")
    print()
    print("Table 05 preview:")
    print(table_05.to_string(index=False))
    print()
    print("Table 06 preview:")
    print(table_06.to_string(index=False))


if __name__ == "__main__":
    main()
