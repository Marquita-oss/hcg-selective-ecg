"""Regenera las figuras multisemilla desde los CSV agregados incluidos."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = PACKAGE_ROOT / "03_RESULTADOS" / "aggregate_multiseed"
DATA_DIR = AGG_DIR / "figure_data"
FIG_DIR = AGG_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "P1_MTL_BALANCED_PROPOSED": "#0072B2",
    "P0_MTL_UNBALANCED": "#009E73",
    "B3_THRESHOLD_UNCERTAINTY": "#E69F00",
    "RANDOM_REFERRAL": "#777777",
}
LABELS = {
    "P1_MTL_BALANCED_PROPOSED": "P1 multitarea balanceado",
    "P0_MTL_UNBALANCED": "P0 multitarea",
    "B3_THRESHOLD_UNCERTAINTY": "B3 incertidumbre por umbral",
    "RANDOM_REFERRAL": "Seleccion aleatoria",
}


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#D9E1E5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage() -> None:
    data = pd.read_csv(DATA_DIR / "multiseed_fig04_risk_coverage.csv")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for model, group in data.groupby("model_or_score"):
        group = group.sort_values("coverage")
        color = COLORS.get(model, "#444444")
        ax.plot(group["coverage"] * 100, group["mean"] * 100, marker="o", linewidth=2, color=color, label=LABELS.get(model, model))
        ax.fill_between(group["coverage"] * 100, group["ci95_low"] * 100, group["ci95_high"] * 100, color=color, alpha=0.14)
    ax.set_title("Riesgo diagnostico segun cobertura", fontweight="bold")
    ax.set_xlabel("ECG conservados sin derivar (%)")
    ax.set_ylabel("Riesgo entre ECG conservados (%)")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig04_risk_coverage_multiseed")


def plot_capture() -> None:
    data = pd.read_csv(DATA_DIR / "multiseed_fig05_capture.csv")
    metrics = [
        ("LowCertCapture@q", "Captura de baja certeza humana"),
        ("ErrorCapture@q", "Captura de errores diagnosticos"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        subset = data[data["metric"] == metric]
        for model, group in subset.groupby("model_or_score"):
            group = group.sort_values("q")
            color = COLORS.get(model, "#444444")
            ax.plot(group["q"] * 100, group["mean"] * 100, marker="o", linewidth=2, color=color, label=LABELS.get(model, model))
            ax.fill_between(group["q"] * 100, group["ci95_low"] * 100, group["ci95_high"] * 100, color=color, alpha=0.14)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Presupuesto de revision q (%)")
        ax.set_ylabel("Casos capturados (%)")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "fig05_capture_multiseed")


if __name__ == "__main__":
    plot_risk_coverage()
    plot_capture()
    print(f"Figuras agregadas regeneradas en: {FIG_DIR}")
