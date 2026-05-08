#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

TABLES_DIR = Path("outputs/tables")
OUT_DIR = Path("outputs/publication_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["resnet50", "densenet121", "mobilenetv2", "efficientnetv2b0", "vgg16"]

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenetv2": "MobileNetV2",
    "efficientnetv2b0": "EfficientNetV2B0",
    "vgg16": "VGG16",
}

METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]


def mean_std(series: pd.Series) -> str:
    return f"{series.mean():.3f} ± {series.std(ddof=1):.3f}"


def make_internal_external_table() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        path = TABLES_DIR / f"{model}_all_seed_metrics.csv"
        df = pd.read_csv(path)

        for dataset, g in df.groupby("dataset"):
            row = {"model": MODEL_LABELS[model], "dataset": dataset}
            for metric in METRICS:
                row[metric] = mean_std(g[metric])
            rows.append(row)

    return pd.DataFrame(rows)


def make_subgroup_table() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        path = TABLES_DIR / f"{model}_all_seed_subgroup_metrics.csv"
        df = pd.read_csv(path)

        for subgroup, g in df.groupby("subgroup"):
            row = {
                "model": MODEL_LABELS[model],
                "subgroup": subgroup,
                "n": int(g["n"].iloc[0]) if "n" in g.columns else "",
            }
            for metric in METRICS:
                row[metric] = mean_std(g[metric])
            rows.append(row)

    out = pd.DataFrame(rows)
    out["subgroup"] = pd.Categorical(out["subgroup"], ["dark", "light"], ordered=True)
    out["model"] = pd.Categorical(out["model"], [MODEL_LABELS[m] for m in MODELS], ordered=True)
    return out.sort_values(["model", "subgroup"]).astype({"model": str, "subgroup": str})


def make_gap_table() -> pd.DataFrame:
    path = TABLES_DIR / "bosque_light_dark_bootstrap_comparison_readable.csv"
    df = pd.read_csv(path)

    sig = df[df["significant_fdr_0_05"]].copy()
    sig["model"] = sig["model"].map(MODEL_LABELS).fillna(sig["model"])

    cols = [
        "model",
        "metric",
        "dark",
        "light",
        "gap_light_minus_dark",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "p_value_bootstrap",
        "p_value_fdr_bh",
        "sig_fdr",
    ]
    sig = sig[cols]

    for c in ["dark", "light", "gap_light_minus_dark", "bootstrap_ci_low", "bootstrap_ci_high", "p_value_bootstrap", "p_value_fdr_bh"]:
        sig[c] = sig[c].astype(float).round(4)

    return sig.sort_values(["model", "metric"])


def latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_simple_latex(df: pd.DataFrame, path: Path) -> None:
    col_spec = "l" * len(df.columns)
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in df.columns) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(df: pd.DataFrame, name: str) -> None:
    csv_path = OUT_DIR / f"{name}.csv"
    tex_path = OUT_DIR / f"{name}.tex"

    df.to_csv(csv_path, index=False)
    write_simple_latex(df, tex_path)

    print("Wrote:", csv_path)
    print("Wrote:", tex_path)
    print(df.to_string(index=False))
    print()


def main() -> None:
    write_outputs(make_internal_external_table(), "table_01_internal_external_performance")
    write_outputs(make_subgroup_table(), "table_02_bosque_subgroup_performance")
    write_outputs(make_gap_table(), "table_03_fdr_significant_light_dark_gaps")


if __name__ == "__main__":
    main()
