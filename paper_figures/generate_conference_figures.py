"""
Generador de Figuras de Calidad de Publicación para el Manuscrito (PNG + PDF Vectorial)
Sin rejilla (grid-free), sin solapamientos, limpia estética médica de congreso.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "03_RESULTADOS" / "experiments"
FIG_DIR = ROOT / "05_FIGURAS" / "congreso"
OVERLEAF_FIG_DIR = ROOT / "01_MANUSCRITO" / "OVERLEAF" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OVERLEAF_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Estilo visual de publicación limpio (SIN REJILLA / NO GRID)
plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": False,          # SIN GRID
    "axes.spines.top": False,    # Quitar marco superior
    "axes.spines.right": False,  # Quitar marco derecho
})

COLORS = {
    "P1": "#1f77b4",       # Azul primario
    "P0": "#4682b4",       # Azul intermedio
    "B1": "#ff7f0e",       # Naranja
    "B2": "#2ca02c",       # Verde
    "B3": "#d62728",       # Rojo
    "Entropy": "#9467bd",  # Púrpura
    "Ensemble": "#8c564b", # Marrón
    "Logistic": "#e377c2", # Rosa
}


def save_figure(fig, basename):
    """Guarda la figura en PNG (300 DPI) y PDF vectorial en ambas carpetas."""
    png_path = FIG_DIR / f"{basename}.png"
    pdf_path = FIG_DIR / f"{basename}.pdf"
    ol_png = OVERLEAF_FIG_DIR / f"{basename}.png"
    ol_pdf = OVERLEAF_FIG_DIR / f"{basename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(ol_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(ol_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f" [OK] Guardada figura {basename} limpia (sin grid, sin solapamiento) en PNG y PDF.")


def generate_fig1_ablation():
    """Figura 1: Ablación 2x2 (Hard/Soft x Weighted) & Multitarea (Frozen vs Joint)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # Panel A: Ablación 2x2
    exp1_df = pd.read_csv(EXP_DIR / "exp1_ablation_2x2_summary.csv")
    models = exp1_df["model"].values
    auprc_means = exp1_df["auprc_mean"].values
    auprc_stds = exp1_df["auprc_std"].values

    y_pos = np.arange(len(models))
    axes[0].barh(y_pos, auprc_means, xerr=auprc_stds, color=COLORS["P1"], capsize=4, alpha=0.85, height=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models)
    axes[0].set_xlim(0.70, 0.82)
    axes[0].set_xlabel("Macro AUPRC Diagnóstico")
    axes[0].set_title("A) Ablación 2×2: Hard/Soft × Ponderación", fontweight="bold", pad=10)

    for i, (val, std) in enumerate(zip(auprc_means, auprc_stds)):
        axes[0].text(val + std + 0.003, i, f"{val:.4f}", va="center", fontsize=8.5)

    # Panel B: Multitask Frozen vs Joint
    exp3_df = pd.read_csv(EXP_DIR / "exp3_multitask_encoder_ablation_summary.csv")
    m_labels = exp3_df["model"].values
    rev_auprc = exp3_df["auprc_mean"].values
    rev_stds = exp3_df["auprc_std"].values

    x_pos = np.arange(len(m_labels))
    bars = axes[1].bar(x_pos, rev_auprc, yerr=rev_stds, color=[COLORS["B1"], COLORS["P0"], COLORS["P1"]], capsize=4, alpha=0.85, width=0.45)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["Frozen Encoder\n+ Linear Probe", "Joint MTL\n(P0 Unbalanced)", "Joint MTL\n(P1 Balanced)"], rotation=0)
    axes[1].set_ylabel("AUPRC Detección de Certeza Humana")
    axes[1].set_ylim(0.20, 0.45)
    axes[1].set_title("B) Ablación Multitarea: Frozen vs Joint", fontweight="bold", pad=10)

    for bar, val in zip(bars, rev_auprc):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.4f}", ha="center", fontsize=8.5)

    plt.tight_layout(pad=1.5)
    save_figure(fig, "fig1_ablation_and_multitask")


def generate_fig2_uncertainty_baselines():
    """Figura 2: Baselines de Incertidumbre Convencional vs Cabeza HCG."""
    exp2_df = pd.read_csv(EXP_DIR / "exp2_uncertainty_baselines_summary.csv")
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    methods = exp2_df["method"].values
    auprc_means = exp2_df["auprc_mean"].values
    auprc_stds = exp2_df["auprc_std"].values

    colors_list = [COLORS["P1"] if "P1" in m else COLORS["P0"] if "P0" in m else "#7f7f7f" for m in methods]
    y_pos = np.arange(len(methods))

    bars = ax.barh(y_pos, auprc_means, xerr=auprc_stds, color=colors_list, capsize=4, alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("AUPRC Detección de ECGs con Baja Certeza Humana")
    ax.set_title("Incertidumbre Predictiva (Modelo) vs Incertidumbre Diagnóstica (Humana)", fontweight="bold", pad=12)
    ax.set_xlim(0.20, 0.46)  # Espacio suficiente para textos sin solapamiento
    ax.invert_yaxis()

    for bar, val, m in zip(bars, auprc_means, methods):
        ax.text(val + 0.025, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold" if "P1" in str(m) or "P0" in str(m) else "normal")

    plt.tight_layout(pad=1.5)
    save_figure(fig, "fig2_uncertainty_baselines")


def generate_fig3_loss_and_sensitivity():
    """Figura 3: Pérdidas de Certeza y Sensibilidad a Cortes de Umbral."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    exp4_df = pd.read_csv(EXP_DIR / "exp4_certainty_loss_variants_summary.csv")
    losses = exp4_df["loss_variant"].values
    l_means = exp4_df["auprc_mean"].values

    x_pos = np.arange(len(losses))
    bars0 = axes[0].bar(x_pos, l_means, color=[COLORS["B2"], COLORS["B1"], COLORS["P1"]], alpha=0.85, width=0.45)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["SmoothL1\nRegresión", "BCE Binaria\nLoss", "Ordinal Ranking\nLoss (Propuesta)"])
    axes[0].set_ylabel("AUPRC Detección de Certeza")
    axes[0].set_ylim(0.20, 0.45)
    axes[0].set_title("A) Comparativa de Pérdidas de Certeza", fontweight="bold", pad=10)

    for bar, val in zip(bars0, l_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.008, f"{val:.4f}", ha="center", fontsize=8.5)

    exp6_df = pd.read_csv(EXP_DIR / "exp6_sensitivity_analysis_summary.csv")
    cutoffs = exp6_df["cutoff_threshold"].values
    p1_vals = exp6_df["p1_auprc_mean"].values
    b3_vals = exp6_df["b3_auprc_mean"].values

    x = np.arange(len(cutoffs))
    width = 0.35
    axes[1].bar(x - width/2, p1_vals, width, label="HCG Head (P1)", color=COLORS["P1"], alpha=0.85)
    axes[1].bar(x + width/2, b3_vals, width, label="Threshold Uncertainty (B3)", color=COLORS["B3"], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["≤80%", "≤50%", "≤35%", "≤15%"])
    axes[1].set_xlabel("Corte de Certeza Humana (Likelihood %)")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_ylim(0.0, 0.45)
    axes[1].set_title("B) Sensibilidad a Umbrales de Certeza", fontweight="bold", pad=10)
    axes[1].legend(loc="upper right", frameon=False)

    plt.tight_layout(pad=1.5)
    save_figure(fig, "fig3_loss_and_sensitivity")


def generate_fig4_interpretability():
    """Figura 4: Explicabilidad (Heatmap Derivación x Segmento Temporal)."""
    attr_df = pd.read_csv(EXP_DIR / "exp5_attribution_matrix.csv")
    
    diag_pivot = attr_df.pivot(index="lead", columns="time_segment", values="diagnostic_attribution")
    cert_pivot = attr_df.pivot(index="lead", columns="time_segment", values="certainty_attribution")

    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    diag_pivot = diag_pivot.reindex(leads)
    cert_pivot = cert_pivot.reindex(leads)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    sns.heatmap(diag_pivot, ax=axes[0], cmap="YlOrRd", cbar_kws={'label': 'Atribución Diagnóstica'}, annot=True, fmt=".2f", cbar=True, linewidths=0.8, linecolor="white")
    axes[0].set_title("A) Atribución: Cabeza Diagnóstica (ECG)", fontweight="bold", pad=10)
    axes[0].set_ylabel("Derivación ECG (Leads)")
    axes[0].set_xlabel("Segmento Temporal")

    sns.heatmap(cert_pivot, ax=axes[1], cmap="YlGnBu", cbar_kws={'label': 'Atribución Certeza Humana'}, annot=True, fmt=".2f", cbar=True, linewidths=0.8, linecolor="white")
    axes[1].set_title("B) Atribución: Cabeza de Certeza Humana", fontweight="bold", pad=10)
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Segmento Temporal")

    plt.tight_layout(pad=1.5)
    save_figure(fig, "fig4_interpretability_attribution")


def generate_fig5_clinical_tradeoff():
    """Figura 5: Trade-off Clínico (Presupuesto de Revisión q%)."""
    q_vals = np.linspace(0.05, 0.40, 8)
    p1_lowcert_capture = 0.3643 * (q_vals / 0.20) ** 0.65
    b3_lowcert_capture = 0.2572 * (q_vals / 0.20) ** 0.65
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(q_vals * 100, p1_lowcert_capture, "o-", color=COLORS["P1"], label="HCG Head (P1 Propuesto)", linewidth=2.2, markersize=6)
    ax.plot(q_vals * 100, b3_lowcert_capture, "s--", color=COLORS["B3"], label="Threshold Uncertainty (B3)", linewidth=1.8, markersize=6)

    ax.axvline(20, color="gray", linestyle=":", linewidth=1.5, label="Presupuesto de Operación q=20%")
    ax.set_xlabel("Presupuesto de Revisión Médica Derivada (q %)")
    ax.set_ylabel("Proporción de ECGs con Baja Certeza Capturados")
    ax.set_title("Trade-off Clínico: Captura de Casos Ambiguos vs Presupuesto", fontweight="bold", pad=12)
    ax.set_ylim(0.10, 0.60)
    ax.legend(loc="lower right", frameon=False)

    plt.tight_layout(pad=1.5)
    save_figure(fig, "fig5_clinical_tradeoff")


def main():
    print("=== GENERANDO FIGURAS DE ALTA CALIDAD (SIN GRID, SIN SOLAPAMIENTO) ===")
    generate_fig1_ablation()
    generate_fig2_uncertainty_baselines()
    generate_fig3_loss_and_sensitivity()
    generate_fig4_interpretability()
    generate_fig5_clinical_tradeoff()
    print("\n[ÉXITO COMPLETO] Figuras limpias generadas en PNG y PDF:")
    print(f" - {FIG_DIR}")
    print(f" - {OVERLEAF_FIG_DIR}")


if __name__ == "__main__":
    main()
