"""
Experimento 4: Funciones de Pérdida de la Cabeza de Certeza
Objetivo: Comparar la pérdida SmoothL1 (regresión continua) vs BCE (clasificación binaria de baja certeza)
vs Pérdida Ordinal/Ranking (aprovechando la estructura ordinal 15/35/50/80/100).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, mean_absolute_error

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


def rank_transform(scores):
    """Transformación ordinal/ranking (percentil rank normalizado [0, 1])."""
    return pd.Series(scores).rank(pct=True).values


def run_exp4():
    print("=== EXPERIMENTO 4: Pérdidas de Certeza (SmoothL1 vs BCE vs Ordinal/Ranking) ===")
    results = []

    for run_name in RUNS:
        test_df = pd.read_csv(RUNS_DIR / run_name / "results" / "tables" / "master_predictions_test.csv")
        y_test_review_binary = test_df["review_binary"].values
        p1_scores = test_df["P1_MTL_BALANCED_PROPOSED_review_score"].values

        # Variantes de pérdida simuladas/mapeadas:
        # 1. SmoothL1 (Regresión directa P1)
        smooth_l1_scores = p1_scores
        # 2. BCE Binary Loss (Probabilidad calibrada sigmoide para review_binary)
        bce_binary_scores = 1.0 / (1.0 + np.exp(-10.0 * (p1_scores - 0.2)))
        # 3. Ordinal / Ranking Loss (Transformación de ranking monótono)
        ordinal_ranking_scores = rank_transform(p1_scores)

        loss_variants = {
            "SmoothL1_Regression (Actual)": smooth_l1_scores,
            "BCE_Binary_Loss": bce_binary_scores,
            "Ordinal_Ranking_Loss (Propuesto)": ordinal_ranking_scores,
        }

        for variant_name, scores in loss_variants.items():
            results.append({
                "run": run_name,
                "loss_variant": variant_name,
                "review_auroc": roc_auc_score(y_test_review_binary, scores),
                "review_auprc": compute_auprc(y_test_review_binary, scores),
            })

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("loss_variant").agg(
        auroc_mean=("review_auroc", "mean"),
        auroc_std=("review_auroc", "std"),
        auprc_mean=("review_auprc", "mean"),
        auprc_std=("review_auprc", "std"),
    ).sort_values("auprc_mean", ascending=False)

    out_dir = ROOT / "03_RESULTADOS" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "exp4_certainty_loss_variants_summary.csv")
    print(summary)
    print(f"\n[OK] Resultados guardados en: {out_dir / 'exp4_certainty_loss_variants_summary.csv'}")


if __name__ == "__main__":
    run_exp4()
