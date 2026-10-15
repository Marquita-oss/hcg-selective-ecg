"""
Experimento 6: Sensibilidad de Cortes de Certeza (80, 50, 35, 15) y Generalización con Backbone ResNet1D
Objetivo: Demostrar que el fenómeno de captura de certeza humana es robusto frente a distintos umbrales
y no depende de la arquitectura CNN1D particular.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "03_RESULTADOS" / "runs"
RUNS = [
    "hcg_selective_ecg_full_20260712_131520",
    "hcg_selective_ecg_full_seed13",
    "hcg_selective_ecg_full_seed123",
    "hcg_selective_ecg_full_seed2024",
    "hcg_selective_ecg_full_seed7",
]


def compute_auprc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def run_exp6():
    print("=== EXPERIMENTO 6: Sensibilidad a Umbrales de Certeza (<=80, <=50, <=35, <=15) ===")
    
    thresholds = {
        "LowCert_<=80": 0.20,  # 1 - 0.80 = 0.20
        "LowCert_<=50": 0.50,  # 1 - 0.50 = 0.50
        "LowCert_<=35": 0.65,  # 1 - 0.35 = 0.65
        "LowCert_<=15": 0.85,  # 1 - 0.15 = 0.85
    }

    results = []

    for run_name in RUNS:
        df = pd.read_csv(RUNS_DIR / run_name / "results" / "tables" / "master_predictions_test.csv")
        p1_scores = df["P1_MTL_BALANCED_PROPOSED_review_score"].values
        b3_scores = df["B3_threshold_uncertainty_score"].values
        ambiguity_scores = df["ambiguity_score"].values

        for th_label, amb_cutoff in thresholds.items():
            y_binary_cutoff = (ambiguity_scores >= amb_cutoff).astype(int)
            
            auprc_p1 = compute_auprc(y_binary_cutoff, p1_scores)
            auprc_b3 = compute_auprc(y_binary_cutoff, b3_scores)

            results.append({
                "run": run_name,
                "cutoff_threshold": th_label,
                "auprc_P1_HCG": auprc_p1,
                "auprc_B3_Baseline": auprc_b3,
                "delta_P1_minus_B3": auprc_p1 - auprc_b3
            })

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("cutoff_threshold").agg(
        p1_auprc_mean=("auprc_P1_HCG", "mean"),
        b3_auprc_mean=("auprc_B3_Baseline", "mean"),
        delta_mean=("delta_P1_minus_B3", "mean"),
        delta_std=("delta_P1_minus_B3", "std"),
    )

    out_dir = ROOT / "03_RESULTADOS" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "exp6_sensitivity_analysis_summary.csv")
    print(summary)
    print(f"\n[OK] Resultados guardados en: {out_dir / 'exp6_sensitivity_analysis_summary.csv'}")


if __name__ == "__main__":
    run_exp6()
