from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA = PACKAGE_ROOT / "03_RESULTADOS" / "paper_source_tables"
OUT = PACKAGE_ROOT / "05_FIGURAS" / "manuscrito"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#0B3A5B",
    "blue": "#2F6B8A",
    "teal": "#2A8C82",
    "gold": "#D59B2D",
    "red": "#B64A4A",
    "gray": "#68747D",
    "light": "#E9EFF2",
    "ink": "#17242D",
}

MODEL_COLORS = {
    "B3": COLORS["gray"],
    "B2prob": COLORS["gold"],
    "P0": COLORS["blue"],
    "P1": COLORS["teal"],
}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#8A969E",
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax, xy, width, height, title, body, color, face="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.7,
        edgecolor=color,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", weight="bold", color=color, fontsize=11)
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center", fontsize=9.5, linespacing=1.25)


def make_design_figure():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Diseño del estudio: dos usos de la certeza consignada en PTB-XL", ha="center", va="top", fontsize=15, weight="bold", color=COLORS["navy"])

    rounded_box(ax, (0.03, 0.58), 0.19, 0.24, "PTB-XL", "ECG 12 derivaciones\n+ likelihoods por código SCP", COLORS["navy"], COLORS["light"])
    rounded_box(ax, (0.28, 0.58), 0.18, 0.24, "Etiquetas", "Hard: presencia\nSoft: likelihood / 100", COLORS["blue"])
    rounded_box(ax, (0.53, 0.58), 0.18, 0.24, "Etapa A", "B1 hard vs. B2 soft\nDiscriminación y calibración", COLORS["gold"])
    rounded_box(ax, (0.78, 0.58), 0.19, 0.24, "Resultado A", "Efecto global no concluyente\nEfecto dependiente de la clase", COLORS["gold"], "#FFF8E8")

    rounded_box(ax, (0.28, 0.16), 0.18, 0.24, "Ambigüedad", "a = máximo (1 - soft)\nBaja certeza: a ≥ 0,20", COLORS["teal"])
    rounded_box(ax, (0.53, 0.16), 0.18, 0.24, "Etapa B", "P0/P1 multitarea\nCabeza diagnóstica + revisión", COLORS["teal"])
    rounded_box(ax, (0.78, 0.16), 0.19, 0.24, "Resultado B", "Mayor enriquecimiento de\nanotaciones de baja certeza", COLORS["teal"], "#EAF6F4")

    arrows = [
        ((0.22, 0.70), (0.28, 0.70)),
        ((0.46, 0.70), (0.53, 0.70)),
        ((0.71, 0.70), (0.78, 0.70)),
        ((0.37, 0.58), (0.37, 0.40)),
        ((0.46, 0.28), (0.53, 0.28)),
        ((0.71, 0.28), (0.78, 0.28)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.8, color=COLORS["navy"], shrinkA=2, shrinkB=2))

    ax.text(0.03, 0.07, "Misma señal de origen; preguntas y criterios de evaluación distintos.", fontsize=10, color=COLORS["gray"], style="italic")
    save(fig, "fig01_diseno_estudio")


def make_calibration_figure():
    df = pd.read_csv(DATA / "calibration_class_diffs.csv")
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=True)

    for ax, metric, title in zip(axes, ["Brier", "ECE"], ["Cambio en Brier", "Cambio en ECE (10 bins)"]):
        sub = df[df["metric"] == metric].set_index("class").loc[classes].reset_index()
        y = np.arange(len(sub))
        vals = sub["diff_mean"].to_numpy()
        low = sub["ci95_low"].to_numpy()
        high = sub["ci95_high"].to_numpy()
        colors = [COLORS["teal"] if v < 0 else COLORS["red"] for v in vals]
        ax.axvline(0, color=COLORS["ink"], lw=1, ls="--")
        for yi, val, lo, hi, color in zip(y, vals, low, high, colors):
            ax.errorbar(
                val,
                yi,
                xerr=np.array([[val - lo], [hi - val]]),
                fmt="none",
                ecolor=color,
                elinewidth=2,
                capsize=4,
                zorder=2,
            )
        ax.scatter(vals, y, c=colors, s=55, zorder=3, edgecolor="white", linewidth=0.8)
        for i, row in sub.iterrows():
            if bool(row["ci_excludes_zero"]):
                ax.text(high[i] + 0.004, i, "*", va="center", fontsize=14, weight="bold", color=colors[i])
        ax.set_title(title, weight="bold", color=COLORS["navy"])
        ax.set_xlabel("Diferencia B2 soft - B1 hard")
        ax.grid(axis="x", color="#DCE3E7", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()
    axes[0].set_yticks(np.arange(len(classes)), classes)
    axes[0].text(0.02, -0.22, "← favorece B2 soft", transform=axes[0].transAxes, color=COLORS["teal"], weight="bold")
    axes[1].text(0.98, -0.22, "favorece B1 hard →", transform=axes[1].transAxes, ha="right", color=COLORS["red"], weight="bold")
    fig.suptitle("El efecto de la supervisión suave sobre calibración depende de la clase", fontsize=14, weight="bold", color=COLORS["navy"], y=1.02)
    fig.text(0.5, -0.02, "Media de cinco semillas e IC95% t pareado. * El intervalo no incluye cero.", ha="center", color=COLORS["gray"], fontsize=9)
    fig.tight_layout()
    save(fig, "fig02_calibracion_por_clase")


def make_review_figure():
    df = pd.read_csv(DATA / "review_metrics.csv")
    order = ["B3_THRESHOLD_UNCERTAINTY", "B2_PROB_UNCERTAINTY", "P0_MTL_UNBALANCED", "P1_MTL_BALANCED_PROPOSED"]
    labels = ["B3\nUmbral", "B2\nProbabilidad", "P0\nMultitarea", "P1\nMultitarea balanceado"]
    keys = ["B3", "B2prob", "P0", "P1"]
    sub = df.set_index("model_or_score").loc[order]
    vals = sub["review_AUPRC_mean"].to_numpy()
    low = sub["review_AUPRC_ci95_low"].to_numpy()
    high = sub["review_AUPRC_ci95_high"].to_numpy()
    x = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(x, vals, color=[MODEL_COLORS[k] for k in keys], width=0.66, edgecolor="white", linewidth=1)
    ax.errorbar(x, vals, yerr=np.vstack([vals - low, high - vals]), fmt="none", ecolor=COLORS["ink"], capsize=4, lw=1.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.011, f"{val:.3f}", ha="center", va="bottom", weight="bold")
    ax.plot([0, 0, 3, 3], [0.405, 0.420, 0.420, 0.405], color=COLORS["teal"], lw=1.5)
    ax.text(1.5, 0.424, "P1 - B3 = +0,107; IC95% [0,086; 0,128]", ha="center", fontsize=9.5, color=COLORS["teal"], weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("AUPRC para baja certeza")
    ax.set_ylim(0, 0.46)
    ax.set_title("La tarea auxiliar aprende a ordenar anotaciones de baja certeza", fontsize=14, weight="bold", color=COLORS["navy"])
    ax.grid(axis="y", color="#DCE3E7", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, -0.23, "Mayor AUPRC indica mejor priorización de los casos positivos poco frecuentes.", transform=ax.transAxes, color=COLORS["gray"], fontsize=9)
    fig.tight_layout()
    save(fig, "fig03_auprc_revision")


def make_enrichment_figure():
    df = pd.read_csv(DATA / "enrichment_summary.csv")
    order = ["B3", "B2prob", "P0", "P1"]
    names = {"B3": "B3 umbral", "B2prob": "B2 probabilidad", "P0": "P0 multitarea", "P1": "P1 multitarea balanceado"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for score in order:
        sub = df[df["score"] == score].sort_values("q")
        axes[0].plot(sub["q"] * 100, sub["LowCertCapture@q_mean"] * 100, marker="o", lw=2.2, ms=4.5, color=MODEL_COLORS[score], label=names[score])
        axes[1].plot(sub["q"] * 100, sub["LowCertPrecision@q_mean"] * 100, marker="o", lw=2.2, ms=4.5, color=MODEL_COLORS[score], label=names[score])

    axes[0].plot([5, 30], [5, 30], ls="--", color="#9AA5AC", lw=1.2, label="Selección aleatoria")
    axes[0].set_title("Captura acumulada", weight="bold", color=COLORS["navy"])
    axes[0].set_ylabel("Casos de baja certeza capturados (%)")
    axes[1].set_title("Precisión de la cola", weight="bold", color=COLORS["navy"])
    axes[1].set_ylabel("Registros derivados con baja certeza (%)")
    for ax in axes:
        ax.set_xlabel("Presupuesto de revisión q (%)")
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.grid(color="#DCE3E7", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(20, color="#BFC8CD", lw=1, ls=":")
    axes[0].annotate("P1: 36,9%", xy=(20, 36.94), xytext=(22.2, 43), arrowprops=dict(arrowstyle="->", color=COLORS["teal"]), color=COLORS["teal"], weight="bold")
    axes[1].annotate("P1: 39,0%", xy=(20, 38.95), xytext=(22.5, 41.4), arrowprops=dict(arrowstyle="->", color=COLORS["teal"]), color=COLORS["teal"], weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("La supervisión multitarea enriquece la cola de revisión", fontsize=14, weight="bold", color=COLORS["navy"], y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "fig04_enriquecimiento_baja_certeza")


def make_capture_comparison():
    df = pd.read_csv(DATA / "enrichment_summary.csv")
    order = ["B3", "B2prob", "P0", "P1"]
    names = ["B3\nUmbral", "B2\nProbabilidad", "P0\nMultitarea", "P1\nBalanceado"]
    sub = df[(df["q"] == 0.20)].set_index("score").loc[order]
    low = sub["LowCertCapture@q_mean"].to_numpy() * 100
    err = sub["ErrorCapture@q_mean"].to_numpy() * 100
    x = np.arange(len(order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    b1 = ax.bar(x - width / 2, low, width, color=COLORS["teal"], label="Captura de baja certeza")
    b2 = ax.bar(x + width / 2, err, width, color=COLORS["gold"], label="Captura de errores diagnósticos")
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=9)
    ax.axhline(20, color="#9AA5AC", ls="--", lw=1.2, label="Selección aleatoria esperada")
    ax.set_xticks(x, names)
    ax.set_ylabel("Casos capturados (%)")
    ax.set_ylim(0, 44)
    ax.set_title("Presupuesto de revisión del 20%: dos objetivos distintos", fontsize=14, weight="bold", color=COLORS["navy"])
    ax.grid(axis="y", color="#DCE3E7", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.01, -0.23, "P0/P1 priorizan baja certeza; B3/B2 concentran algo más los errores del clasificador.", transform=ax.transAxes, color=COLORS["gray"], fontsize=9)
    fig.tight_layout()
    save(fig, "fig05_captura_baja_certeza_vs_error")


def make_class_figure():
    df = pd.read_csv(DATA / "class_capture_summary.csv")
    sub = df[(df["q"] == 0.20) & (df["score"].isin(["B3", "P1"]))]
    pivot = sub.pivot(index="class", columns="score", values="ClassErrorCapture@q_mean")
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    pivot = pivot.loc[classes]
    x = np.arange(len(classes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    b3 = ax.bar(x - width / 2, pivot["B3"] * 100, width, label="B3 umbral", color=MODEL_COLORS["B3"])
    p1 = ax.bar(x + width / 2, pivot["P1"] * 100, width, label="P1 multitarea", color=MODEL_COLORS["P1"])
    ax.bar_label(b3, fmt="%.1f", padding=2, fontsize=8.5)
    ax.bar_label(p1, fmt="%.1f", padding=2, fontsize=8.5)

    diffs = pd.read_csv(DATA / "class_bootstrap_diffs.csv")
    sig = diffs[(diffs["q"] == 0.20) & (diffs["metric"] == "ClassErrorCapture")].set_index("class")
    for i, cls in enumerate(classes):
        if sig.loc[cls, "ci95_low"] > 0 or sig.loc[cls, "ci95_high"] < 0:
            y = max(pivot.loc[cls, "B3"], pivot.loc[cls, "P1"]) * 100 + 4.2
            ax.text(i, y, "*", ha="center", va="center", fontsize=14, weight="bold", color=COLORS["navy"])
    ax.set_xticks(x, classes)
    ax.set_ylabel("Errores de clase capturados (%)")
    ax.set_ylim(0, 50)
    ax.set_title("La captura de errores es dependiente del diagnóstico", fontsize=14, weight="bold", color=COLORS["navy"])
    ax.grid(axis="y", color="#DCE3E7", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.01, -0.23, "q = 20%. * IC95% bootstrap por paciente de P1-B3 no incluye cero.", transform=ax.transAxes, color=COLORS["gray"], fontsize=9)
    fig.tight_layout()
    save(fig, "fig06_captura_error_por_clase")


def main():
    set_style()
    make_design_figure()
    make_calibration_figure()
    make_review_figure()
    make_enrichment_figure()
    make_capture_comparison()
    make_class_figure()
    print(f"Figuras generadas en: {OUT}")


if __name__ == "__main__":
    main()
