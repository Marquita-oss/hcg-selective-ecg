"""
Generates the new, honest figures for the revised manuscripts:
 1. Real attribution heatmap (diag vs certainty), EN + ES, from Integrated Gradients.
 2. Example ECG traces with real attribution overlay (2 cases), EN + ES.
 3. Retained-set purification bar chart (full vs retained AUROC/AUPRC), EN + ES.

Reads the CSV outputs of compute_real_attribution.py / compute_integrated_gradients.py /
compute_retained_set_metrics.py from ATTRIBUTION_OUT_DIR (default ./out; must match the
value used for those scripts) and writes figures back into the same folder, both as
.pdf (for LaTeX) and .png (for Word/CINVIE). See README.md "Reproducing locally".
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from compute_real_attribution import (
    DEVICE, SEED_RUN_MAP, CLASSES, LEAD_NAMES, TIME_SEGMENTS,
    DELIVERY_RUNS, load_p1_model, resolve_record_stem, read_ecg_record,
)
from compute_integrated_gradients import integrated_gradients

SCRATCH = Path(os.environ.get("ATTRIBUTION_OUT_DIR", "./out")).resolve()
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT_DIR = SCRATCH
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

NAVY = "#1b2a4a"
TEAL = "#2a8c82"
CORAL = "#d9544d"
GREY = "#8a8f98"


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------
# 1. Honest attribution heatmap (IG, pooled across 5 seeds)
# ------------------------------------------------------------------
def fig_heatmap(lang="en"):
    df = pd.read_csv(SCRATCH / "ig_exp5_attribution_matrix.csv")
    agg = pd.read_csv(SCRATCH / "ig_attribution_aggregate.csv").set_index("metric")

    leads = LEAD_NAMES
    segs = TIME_SEGMENTS
    diag = df.pivot(index="lead", columns="time_segment", values="diagnostic_attribution").reindex(index=leads, columns=segs).to_numpy()
    cert = df.pivot(index="lead", columns="time_segment", values="certainty_attribution").reindex(index=leads, columns=segs).to_numpy()

    if lang == "en":
        seg_labels = ["P-wave\n0-2.5s", "QRS\n2.5-5.0s", "ST-T\n5.0-7.5s", "T-end\n7.5-10s"]
        title_a, title_b = "A) Diagnostic head", "B) Certainty head"
        suptitle = "Real Integrated-Gradients attribution: no QRS/ST-T dissociation"
        iou = agg.loc["iou_top25pct", "mean"]
        iou_lo, iou_hi = agg.loc["iou_top25pct", "ci95_low"], agg.loc["iou_top25pct", "ci95_high"]
        caption = (f"Top-25% cell overlap (IoU) between heads = {iou:.2f} [{iou_lo:.2f}, {iou_hi:.2f}] across 5 seeds "
                   f"-- well above the ~0.14 expected by chance. Both heads spread attribution close to uniformly\n"
                   f"across time segments (~25% each) and lead groups (~50/50 precordial/peripheral); see Table in Section IV.F.")
        cbar_label = "Normalized attribution"
        fname = "fig4_real_attribution_EN"
    else:
        seg_labels = ["Onda P\n0-2.5s", "QRS\n2.5-5.0s", "ST-T\n5.0-7.5s", "T-final\n7.5-10s"]
        title_a, title_b = "A) Cabeza diagnóstica", "B) Cabeza de certeza"
        suptitle = "Atribución real (Integrated Gradients): sin disociación QRS/ST-T"
        iou = agg.loc["iou_top25pct", "mean"]
        iou_lo, iou_hi = agg.loc["iou_top25pct", "ci95_low"], agg.loc["iou_top25pct", "ci95_high"]
        caption = (f"Solapamiento (IoU) del 25% superior de celdas entre cabezas = {iou:.2f} [{iou_lo:.2f}, {iou_hi:.2f}] en 5 semillas "
                   f"-- muy por encima del ~0,14 esperado al azar. Ambas cabezas reparten la atribución de forma casi uniforme\n"
                   f"entre segmentos temporales (~25% cada uno) y grupos de derivaciones (~50/50 precordial/periférica); ver tabla en Sección IV.F.")
        cbar_label = "Atribución normalizada"
        fname = "fig4_real_attribution_ES"

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.2))
    vmin = min(diag.min(), cert.min())
    vmax = max(diag.max(), cert.max())
    vmid = (vmin + vmax) / 2.0
    for ax, mat, title in zip(axes, [diag, cert], [title_a, title_b]):
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(seg_labels)
        ax.set_yticks(range(12)); ax.set_yticklabels(leads)
        ax.set_title(title, fontweight="bold", color=NAVY)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(12):
            for j in range(4):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < vmid else "black", fontsize=7, fontweight="medium")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(suptitle, fontweight="bold", color=NAVY, y=1.03)
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=7.8, color="#333333", wrap=True)
    save(fig, fname)


# ------------------------------------------------------------------
# 2. Example ECG traces with real attribution overlay
# ------------------------------------------------------------------
def pick_example_cases(seed=42):
    run_name = SEED_RUN_MAP[seed]
    df = pd.read_csv(DELIVERY_RUNS / run_name / "results" / "tables" / "master_predictions_test.csv")
    certain = df[(df["review_binary"] == 0) & (df["P1_MTL_BALANCED_PROPOSED_sample_error"] == 0)].sort_values("P1_MTL_BALANCED_PROPOSED_review_score").iloc[0]
    ambiguous = df[df["review_binary"] == 1].sort_values("P1_MTL_BALANCED_PROPOSED_review_score", ascending=False).iloc[0]
    return certain, ambiguous


def fig_ecg_examples(lang="en"):
    model = load_p1_model(42)
    certain, ambiguous = pick_example_cases(42)
    cases = [("certain", certain), ("ambiguous", ambiguous)]

    if lang == "en":
        labels = {"certain": "Clear, high-certainty case\n(retained, correctly diagnosed)",
                   "ambiguous": "Low human-certainty case\n(referred to review)"}
        row_labels = ["Diagnostic attribution", "Certainty attribution"]
        xlabel = "Time (s)"
        suptitle = "Real Integrated-Gradients attribution on individual test ECGs (lead II shown)"
        fname = "fig_ecg_examples_EN"
    else:
        labels = {"certain": "Caso claro, certeza alta\n(retenido, diagnóstico correcto)",
                  "ambiguous": "Caso de baja certeza humana\n(derivado a revisión)"}
        row_labels = ["Atribución diagnóstica", "Atribución de certeza"]
        xlabel = "Tiempo (s)"
        suptitle = "Atribución real (Integrated Gradients) en ECG individuales de prueba (derivación II)"
        fname = "fig_ecg_examples_ES"

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.0), sharex=True)
    lead_idx = LEAD_NAMES.index("II")
    t = np.arange(1000) / 100.0

    for col, (case_key, row) in enumerate(cases):
        ecg_id = int(row["ecg_id"])
        stem = resolve_record_stem(ecg_id)
        x_np = read_ecg_record(stem)
        y_hard = row[[f"y_{c}" for c in CLASSES]].to_numpy(dtype=np.float32)
        diag_ig, cert_ig, _, _ = integrated_gradients(model, x_np, y_hard)

        sig = x_np[lead_idx]
        for row_i, (attr, rlabel) in enumerate(zip([diag_ig, cert_ig], row_labels)):
            ax = axes[row_i, col]
            a = attr[lead_idx]
            scale = np.max(np.abs(a)) + 1e-8
            ax.plot(t, sig, color="black", lw=0.7)
            ax.fill_between(t, 0, a / scale * 2, color=(CORAL if row_i == 1 else TEAL), alpha=0.4, step="mid")
            if row_i == 0:
                ax.set_title(labels[case_key], fontsize=9, color=NAVY)
            if col == 0:
                ax.set_ylabel(rlabel, fontsize=8)
            ax.set_xlim(0, 10)
            for seg_x in [2.5, 5.0, 7.5]:
                ax.axvline(seg_x, color=GREY, lw=0.5, ls="--", alpha=0.6)
        axes[1, col].set_xlabel(xlabel)

    fig.suptitle(suptitle, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------------------
# 3. Retained-set purification bar chart
# ------------------------------------------------------------------
def fig_retained(lang="en"):
    agg = pd.read_csv(SCRATCH / "retained_set_aggregate_sklearn.csv")
    methods = ["B3", "P0", "P1", "RANDOM"]
    method_labels_en = {"B3": "B3\n(threshold)", "P0": "P0\n(MTL unbal.)", "P1": "P1\n(proposed)", "RANDOM": "Random\n(control)"}
    method_labels_es = {"B3": "B3\n(umbral)", "P0": "P0\n(MTL no bal.)", "P1": "P1\n(propuesto)", "RANDOM": "Aleatorio\n(control)"}

    if lang == "en":
        title = "Does referring the bottom 20% purify the retained 80%?"
        ylabel = "Macro AUPRC"
        full_label, ret_label = "Full test set (100%)", "Retained set (80%, not referred)"
        fname = "fig_retained_purification_EN"
        mlabels = method_labels_en
    else:
        title = "¿Derivar el 20% inferior purifica el 80% retenido?"
        ylabel = "AUPRC macro"
        full_label, ret_label = "Conjunto de prueba completo (100%)", "Conjunto retenido (80%, no derivado)"
        fname = "fig_retained_purification_ES"
        mlabels = method_labels_es

    full_vals, full_lo, full_hi = [], [], []
    ret_vals, ret_lo, ret_hi = [], [], []
    for m in methods:
        f = agg[(agg.method == m) & (agg.metric == "AUPRC_full")].iloc[0]
        r = agg[(agg.method == m) & (agg.metric == "AUPRC_retained")].iloc[0]
        full_vals.append(f["mean"]); full_lo.append(f["mean"] - f["ci95_low"]); full_hi.append(f["ci95_high"] - f["mean"])
        ret_vals.append(r["mean"]); ret_lo.append(r["mean"] - r["ci95_low"]); ret_hi.append(r["ci95_high"] - r["mean"])

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w / 2, full_vals, width=w, yerr=[full_lo, full_hi], capsize=3, color=GREY, label=full_label)
    bars = ax.bar(x + w / 2, ret_vals, width=w, yerr=[ret_lo, ret_hi], capsize=3,
                   color=[CORAL if v < f else TEAL for v, f in zip(ret_vals, full_vals)], label=ret_label)
    for xi, v in zip(x + w / 2, ret_vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([mlabels[m] for m in methods])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.70, 0.82)
    ax.set_title(title, fontweight="bold", color=NAVY)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, fname)


if __name__ == "__main__":
    fig_heatmap("en")
    fig_heatmap("es")
    fig_retained("en")
    fig_retained("es")
    fig_ecg_examples("en")
    fig_ecg_examples("es")
