#!/usr/bin/env python3
"""Generate ORP sizing/power tables and figures for the BOSQUE case study.

The calculations are planning diagnostics, not replacements for the bootstrap
inference used in the dissertation. They assume independent Bernoulli-eligible
units and use large-sample normal approximations for two-sample proportion
contrasts. Rare-event calculations use the exact binomial probability of
observing at least one event under a zero-acceptance (c=0) plan.
"""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.stats import norm
import statsmodels
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "orp_sizing"
FIG_DIR = OUTPUT_ROOT / "figures"
TAB_DIR = OUTPUT_ROOT / "tables"
OUT_DIR = OUTPUT_ROOT / "csv"
for d in (OUTPUT_ROOT, FIG_DIR, TAB_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
BASELINE = 0.85
CURRENT = [
    ("All observations (accuracy-like)", 105, 46),
    ("Malignant lesions (recall)", 77, 21),
    ("Benign lesions (specificity)", 28, 25),
]


def power_two_proportions(n_light: int, n_dark: int, gap: float,
                          p_light: float = BASELINE,
                          alpha: float = ALPHA) -> float:
    """Approximate two-sided power for p_light vs p_light-gap."""
    p_dark = p_light - gap
    if not (0.0 < p_dark < 1.0):
        return float("nan")
    effect = abs(proportion_effectsize(p_light, p_dark))
    return float(
        NormalIndPower().power(
            effect_size=effect,
            nobs1=n_light,
            alpha=alpha,
            ratio=n_dark / n_light,
            alternative="two-sided",
        )
    )


def minimum_detectable_gap(n_light: int, n_dark: int, target_power: float,
                           p_light: float = BASELINE,
                           alpha: float = ALPHA) -> float:
    """Solve for the absolute gap giving target two-sided power."""
    def objective(gap: float) -> float:
        return power_two_proportions(n_light, n_dark, gap, p_light, alpha) - target_power

    upper = p_light - 1e-6
    if objective(upper) < 0:
        return float("nan")
    return float(brentq(objective, 1e-8, upper))


def required_equal_n(gap: float, target_power: float,
                     p_light: float = BASELINE,
                     alpha: float = ALPHA) -> int:
    """Equal sample size per group for a two-sided two-proportion contrast."""
    effect = abs(proportion_effectsize(p_light, p_light - gap))
    n = NormalIndPower().solve_power(
        effect_size=effect,
        power=target_power,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )
    return int(math.ceil(float(n)))


def tac_required_n(tau: float, shortfall: float, target_power: float,
                   alpha: float = ALPHA) -> int:
    """Normal-approximation n for one-sided detection of p=tau-shortfall."""
    p1 = tau - shortfall
    za = norm.ppf(1.0 - alpha)
    zb = norm.ppf(target_power)
    numerator = za * math.sqrt(tau * (1.0 - tau)) + zb * math.sqrt(p1 * (1.0 - p1))
    return int(math.ceil((numerator / shortfall) ** 2))


def detection_probability(n: int, event_rate: float) -> float:
    return 1.0 - (1.0 - event_rate) ** n


def required_n_for_detection(event_rate: float, target_detection: float) -> int:
    return int(math.ceil(math.log(1.0 - target_detection) / math.log(1.0 - event_rate)))


# ---------------------------------------------------------------------------
# Current power diagnostics
# ---------------------------------------------------------------------------
rows = []
for label, n_light, n_dark in CURRENT:
    rows.append({
        "eligible_contrast": label,
        "n_light": n_light,
        "n_dark": n_dark,
        "power_gap_0_10": power_two_proportions(n_light, n_dark, 0.10),
        "power_gap_0_15": power_two_proportions(n_light, n_dark, 0.15),
        "mdd_80": minimum_detectable_gap(n_light, n_dark, 0.80),
        "mdd_90": minimum_detectable_gap(n_light, n_dark, 0.90),
    })
current_df = pd.DataFrame(rows)
current_df.to_csv(OUT_DIR / "orp_current_subgroup_power.csv", index=False)

# LaTeX table 1
tex1 = r"""\begin{table}[h!]
\centering
\scriptsize
\caption{Approximate power of the observed BOSQUE phototype-group sizes for
binary performance contrasts.}
\label{tab:app-orp-current-power}
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Eligible units} & $n_{\mathrm{light}}$ & $n_{\mathrm{dark}}$ &
\textbf{Power, $\delta=0.10$} & \textbf{Power, $\delta=0.15$} &
\textbf{MDD, 80\%} & \textbf{MDD, 90\%} \\
\midrule
"""
label_map = {
    "All observations (accuracy-like)": "All observations (accuracy-like)",
    "Malignant lesions (recall)": "Malignant lesions (recall)",
    "Benign lesions (specificity)": "Benign lesions (specificity)",
}
for _, row in current_df.iterrows():
    tex1 += (
        f"{label_map[row['eligible_contrast']]} & {int(row['n_light'])} & {int(row['n_dark'])} & "
        f"{100*row['power_gap_0_10']:.1f}\\% & {100*row['power_gap_0_15']:.1f}\\% & "
        f"{row['mdd_80']:.3f} & {row['mdd_90']:.3f} \\\\\n"
    )
tex1 += r"""\bottomrule
\end{tabular}

\vspace{2pt}
\parbox{0.98\textwidth}{\footnotesize
\textit{Notes:} MDD denotes the minimum detectable absolute light--dark gap.
Calculations assume a light-condition performance of $0.85$, a dark-condition
performance of $0.85-\delta$, a two-sided $\alpha=0.05$ test, independent
eligible lesions, and the large-sample two-proportion approximation. The table
is a prospective planning diagnostic; it is not a reanalysis of the bootstrap
contrasts and is not applicable directly to F1-score or AUC--PR.}
\end{table}
"""
(TAB_DIR / "table_orp_current_power.tex").write_text(tex1, encoding="utf-8")


# Concise main-chapter summary table.
summary = r"""\begin{table}[h!]
\centering
\scriptsize
\caption{Evidential capacity of the observed BOSQUE metric-specific denominators.}
\label{tab:orp-evidential-capacity-summary}
\setlength{\tabcolsep}{3.3pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Performance question} & $n_{\mathrm{light}}$ & $n_{\mathrm{dark}}$ &
\textbf{Power, $\delta=0.10$} & \textbf{MDD, 80\%} & \textbf{Eligible units} \\
\midrule
"""
summary_labels = [
    ("Accuracy-like metric", "All lesions"),
    ("Recall / sensitivity", "Malignant lesions"),
    ("Specificity", "Benign lesions"),
]
for (_, row), (metric_label, units_label) in zip(current_df.iterrows(), summary_labels):
    summary += (
        f"{metric_label} & {int(row['n_light'])} & {int(row['n_dark'])} & "
        f"{100*row['power_gap_0_10']:.1f}\\% & {row['mdd_80']:.3f} & {units_label} \\\\\n"
    )
summary += r"""\bottomrule
\end{tabular}

\vspace{2pt}
\parbox{0.98\textwidth}{\footnotesize
\textit{Notes:} MDD is the minimum detectable absolute light--dark difference.
Calculations assume $p_{\mathrm{light}}=0.85$,
$p_{\mathrm{dark}}=0.85-\delta$, independent eligible lesions, and a two-sided
$\alpha=0.05$ large-sample two-proportion approximation. They are planning
diagnostics and do not replace the case-study bootstrap analysis.}
\end{table}
"""
(TAB_DIR / "table_orp_evidential_capacity_summary.tex").write_text(summary, encoding="utf-8")

# ---------------------------------------------------------------------------
# Prospective planning requirements
# ---------------------------------------------------------------------------
gaps = [0.05, 0.10, 0.15, 0.20]
planning_rows = []
for gap in gaps:
    planning_rows.append({
        "gap": gap,
        "n_per_group_80": required_equal_n(gap, 0.80),
        "n_per_group_90": required_equal_n(gap, 0.90),
    })
planning_df = pd.DataFrame(planning_rows)
planning_df.to_csv(OUT_DIR / "orp_prospective_subgroup_sample_sizes.csv", index=False)

failure_rates = [0.001, 0.01, 0.05]
rare_rows = []
for q in failure_rates:
    rare_rows.append({
        "failure_rate": q,
        "n_detect_80": required_n_for_detection(q, 0.80),
        "n_detect_90": required_n_for_detection(q, 0.90),
        "n_detect_95": required_n_for_detection(q, 0.95),
    })
rare_df = pd.DataFrame(rare_rows)
rare_df.to_csv(OUT_DIR / "orp_rare_failure_sample_sizes.csv", index=False)

tex2 = r"""\begin{table}[h!]
\centering
\small
\caption{Prospective ORP sizing examples under independent-unit assumptions.}
\label{tab:app-orp-prospective-sizing}
\begin{minipage}[t]{0.47\textwidth}
\centering
\textbf{Panel A: equal subgroup sizes}\par\vspace{2pt}
\scriptsize
\begin{tabular}{@{}rrr@{}}
\toprule
\textbf{Absolute gap $\delta$} & \textbf{$n$/group, 80\%} & \textbf{$n$/group, 90\%} \\
\midrule
"""
for _, row in planning_df.iterrows():
    tex2 += f"{row['gap']:.2f} & {int(row['n_per_group_80'])} & {int(row['n_per_group_90'])} \\\\\n"
tex2 += r"""\bottomrule
\end{tabular}
\end{minipage}
\hfill
\begin{minipage}[t]{0.47\textwidth}
\centering
\textbf{Panel B: detection of at least one failure}\par\vspace{2pt}
\scriptsize
\begin{tabular}{@{}rrrr@{}}
\toprule
\textbf{Failure rate $q$} & \textbf{$n$, 80\%} & \textbf{$n$, 90\%} & \textbf{$n$, 95\%} \\
\midrule
"""
for _, row in rare_df.iterrows():
    tex2 += (
        f"{100*row['failure_rate']:.1f}\\% & {int(row['n_detect_80'])} & "
        f"{int(row['n_detect_90'])} & {int(row['n_detect_95'])} \\\\\n"
    )
tex2 += r"""\bottomrule
\end{tabular}
\end{minipage}

\vspace{4pt}
\parbox{0.98\textwidth}{\footnotesize
\textit{Notes:} Panel A assumes $p_{\mathrm{light}}=0.85$,
$p_{\mathrm{dark}}=0.85-\delta$, a two-sided $\alpha=0.05$ test, equal group
sizes, and the large-sample two-proportion approximation. Panel B is exact for
a zero-acceptance plan: $n$ is the number of independent eligible units needed
for probability $1-\beta$ of observing at least one failure when the failure
rate is $q$. Clustering, unequal weights, or non-exchangeable curation require
larger nominal samples or simulation under the intended design.}
\end{table}
"""
(TAB_DIR / "table_orp_prospective_sizing.tex").write_text(tex2, encoding="utf-8")

# Current rare-event detection probabilities for audit/reproducibility.
current_ns = [21, 25, 46, 77, 105, 151]
detection_rows = []
for n in current_ns:
    for q in failure_rates:
        detection_rows.append({"n": n, "failure_rate": q, "detection_probability": detection_probability(n, q)})
pd.DataFrame(detection_rows).to_csv(OUT_DIR / "orp_current_rare_event_detection.csv", index=False)

# TAC examples used in the prose.
tac_rows = []
for tau in [0.70, 0.80, 0.85]:
    for shortfall in [0.05, 0.10, 0.15]:
        tac_rows.append({
            "tau": tau,
            "shortfall": shortfall,
            "n_80": tac_required_n(tau, shortfall, 0.80),
            "n_90": tac_required_n(tau, shortfall, 0.90),
        })
pd.DataFrame(tac_rows).to_csv(OUT_DIR / "orp_tac_sample_sizes.csv", index=False)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Figure 1: power versus subgroup gap for current eligible denominators.
grid = np.linspace(0.01, 0.45, 180)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
styles = ["-", "--", ":"]
for (label, n_light, n_dark), ls in zip(CURRENT, styles):
    powers = [power_two_proportions(n_light, n_dark, g) for g in grid]
    ax.plot(grid, powers, linestyle=ls, linewidth=2, label=f"{label}: {n_light}/{n_dark}")
ax.axhline(0.80, linestyle="--", linewidth=1)
ax.axhline(0.90, linestyle=":", linewidth=1)
ax.set_xlabel("Absolute light-dark performance gap")
ax.set_ylabel("Approximate two-sided power")
ax.set_xlim(0.0, 0.45)
ax.set_ylim(0.0, 1.01)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_orp_current_subgroup_power.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "figure_orp_current_subgroup_power.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Figure 2: probability of observing at least one failure.
q_grid = np.linspace(0.0, 0.10, 201)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
curve_ns = [21, 46, 105, 151]
styles2 = ["-", "--", "-.", ":"]
for n, ls in zip(curve_ns, styles2):
    probs = [detection_probability(n, q) for q in q_grid]
    ax.plot(100*q_grid, probs, linestyle=ls, linewidth=2, label=f"n={n}")
for level, ls in [(0.80, "--"), (0.90, ":"), (0.95, "-.")]:
    ax.axhline(level, linestyle=ls, linewidth=0.8)
ax.set_xlabel("Failure rate among eligible units (%)")
ax.set_ylabel("Probability of observing at least one failure")
ax.set_xlim(0.0, 10.0)
ax.set_ylim(0.0, 1.01)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(True, linewidth=0.4, alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_orp_rare_event_detection.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "figure_orp_rare_event_detection.png", dpi=220, bbox_inches="tight")
plt.close(fig)

metadata = {
    "alpha": ALPHA,
    "baseline_performance": BASELINE,
    "current_counts": CURRENT,
    "software": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
    },
    "methods": {
        "subgroup_power": "NormalIndPower with Cohen arcsine effect size, two-sided alpha=0.05",
        "rare_event_detection": "Exact binomial complement 1-(1-q)^n",
        "tac_planning": "One-sided large-sample normal approximation",
    },
}
(OUTPUT_ROOT / "orp_sizing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("Generated ORP-sizing outputs in", OUTPUT_ROOT)
