"""
Experimento 3: Ablación Multitarea (Frozen Encoder B1 + Certainty Head vs Joint P0/P1)
Objetivo: Demostrar si la supervisión explícita de certeza modifica la representación aprendida del ECG
o si la certeza humana se puede extraer fácilmente del embedding de un encoder congelado (B1).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, mean_absolute_error
from sklearn.linear_model import Ridge

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


def run_exp3():
    print("=== EXPERIMENTO 3: Ablación Multitarea (Frozen Encoder vs Joint P0/P1) ===")
    results = []

    for run_name in RUNS:
        val_df = pd.read_csv(RUNS_DIR / run_name / "results" / "tables" / "master_predictions_val.csv")
        test_df = pd.read_csv(RUNS_DIR / run_name / "results" / "tables" / "master_predictions_test.csv")

        # Extraer probabilidades B1 como representación intermedia (frozen diagnostic features)
        b1_val_features = val_df[[f"B1_HARD_DIAG_prob_{c}" for c in CLASSES]].values
        b1_test_features = test_df[[f"B1_HARD_DIAG_prob_{c}" for c in CLASSES]].values

        y_val_ambiguity = val_df["ambiguity_score"].values
        y_test_ambiguity = test_df["ambiguity_score"].values
        y_test_review_binary = test_df["review_binary"].values

        # Frozen Encoder + Certainty Head (Ridge probe on B1 features)
        probe = Ridge(alpha=1.0)
        probe.fit(b1_val_features, y_val_ambiguity)
        frozen_test_preds = probe.predict(b1_test_features)

        # Evaluaciones
        models = {
            "Frozen_Encoder_B1 + Linear_Head": frozen_test_preds,
            "Joint_Multitask_P0 (Unbalanced)": test_df["P0_MTL_UNBALANCED_review_score"].values,
            "Joint_Multitask_P1 (Balanced_Propuesto)": test_df["P1_MTL_BALANCED_PROPOSED_review_score"].values,
        }

        for model_label, preds in models.items():
            results.append({
                "run": run_name,
                "model": model_label,
                "review_auroc": roc_auc_score(y_test_review_binary, preds),
                "review_auprc": compute_auprc(y_test_review_binary, preds),
                "mae_ambiguity": mean_absolute_error(y_test_ambiguity, preds),
            })

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("model").agg(
        auroc_mean=("review_auroc", "mean"),
        auroc_std=("review_auroc", "std"),
        auprc_mean=("review_auprc", "mean"),
        auprc_std=("review_auprc", "std"),
        mae_mean=("mae_ambiguity", "mean"),
        mae_std=("mae_ambiguity", "std"),
    ).sort_values("auprc_mean", ascending=False)

    out_dir = ROOT / "03_RESULTADOS" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "exp3_multitask_encoder_ablation_summary.csv")
    print(summary)
    print(f"\n[OK] Resultados guardados en: {out_dir / 'exp3_multitask_encoder_ablation_summary.csv'}")


if __name__ == "__main__":
    run_exp3()
