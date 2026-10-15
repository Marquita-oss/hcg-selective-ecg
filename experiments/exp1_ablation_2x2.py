"""
Experimento 1: Ablación 2x2 (Hard/Soft x Unweighted/Weighted)
Objetivo: Aislar el efecto del balanceo de clases del efecto de la gradación con likelihoods.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "03_RESULTADOS" / "runs"
RUNS = [
    "hcg_selective_ecg_full_20260712_131520",
    "hcg_selective_ecg_full_seed13",
    "hcg_selective_ecg_full_seed123",
    "hcg_selective_ecg_full_seed2024",
    "hcg_selective_ecg_full_seed7",
]

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def compute_auprc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def run_exp1():
    print("=== EXPERIMENTO 1: Ablación 2x2 (Hard/Soft x Unweighted/Weighted) ===")
    results = []

    for run_name in RUNS:
        csv_path = RUNS_DIR / run_name / "results" / "tables" / "master_predictions_test.csv"
        df = pd.read_csv(csv_path)

        models = {
            "Hard_Unweighted (B1_HARD)": "B1_HARD_DIAG",
            "Soft_Unweighted (B2_SOFT)": "B2_SOFT_DIRECT",
            "Hard_Multitask (P0_MTL)": "P0_MTL_UNBALANCED",
            "Soft_Multitask_Weighted (P1_MTL)": "P1_MTL_BALANCED_PROPOSED",
        }

        for model_label, model_prefix in models.items():
            aurocs, auprcs = [], []
            for cls in CLASSES:
                y_true = df[f"y_{cls}"].values
                y_prob = df[f"{model_prefix}_prob_{cls}"].values
                aurocs.append(roc_auc_score(y_true, y_prob))
                auprcs.append(compute_auprc(y_true, y_prob))

            results.append(
                {
                    "run": run_name,
                    "model": model_label,
                    "macro_auroc": np.mean(aurocs),
                    "macro_auprc": np.mean(auprcs),
                }
            )

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("model").agg(
        auroc_mean=("macro_auroc", "mean"),
        auroc_std=("macro_auroc", "std"),
        auprc_mean=("macro_auprc", "mean"),
        auprc_std=("macro_auprc", "std"),
    )

    out_dir = ROOT / "03_RESULTADOS" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "exp1_ablation_2x2_summary.csv")
    print(summary)
    print(f"\n[OK] Resultados guardados en: {out_dir / 'exp1_ablation_2x2_summary.csv'}")


if __name__ == "__main__":
    run_exp1()
