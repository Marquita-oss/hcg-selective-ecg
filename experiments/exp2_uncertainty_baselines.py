"""
Experimento 2: Baselines de Incertidumbre Convencional vs Cabeza de Certeza Humana (HCG)
Objetivo: Demostrar que la incertidumbre predictiva del modelo (Entropía, Ensemble, Confianza)
NO equivale a la incertidumbre diagnóstica humana, y que la cabeza supervisada HCG (P1)
la captura sustancialmente mejor.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression

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


def calculate_entropy(probs):
    # Probs shape: (N, num_classes)
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    entropy_per_class = - (probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
    return np.mean(entropy_per_class, axis=1)


def run_exp2():
    print("=== EXPERIMENTO 2: Baselines de Incertidumbre Convencional vs Cabeza HCG ===")
    
    # 1. Cargar todas las predicciones para Deep Ensemble
    dfs = [pd.read_csv(RUNS_DIR / run / "results" / "tables" / "master_predictions_test.csv") for run in RUNS]
    n_samples = len(dfs[0])

    # 2. Calcular Deep Ensemble Variance across seeds
    ensemble_probs_b1 = np.zeros((n_samples, len(CLASSES), len(RUNS)))
    for idx_run, df_run in enumerate(dfs):
        for idx_cls, cls in enumerate(CLASSES):
            ensemble_probs_b1[:, idx_cls, idx_run] = df_run[f"B1_HARD_DIAG_prob_{cls}"].values

    # Varianza entre semillas por muestra y clase, luego promedio entre clases
    ensemble_variance = np.mean(np.var(ensemble_probs_b1, axis=2), axis=1)

    # 3. Evaluar cada corrida para las métricas individuales y promediar entre semillas
    seed_results = []
    
    for idx_run, df in enumerate(dfs):
        y_low_cert = df["review_binary"].values
        probs_b1 = df[[f"B1_HARD_DIAG_prob_{c}" for c in CLASSES]].values

        # Baseline A: Shannon Entropy H(p)
        entropy_scores = calculate_entropy(probs_b1)

        # Baseline B: Margin / Max Softmax Probability (1 - max(p))
        msp_scores = 1.0 - np.max(probs_b1, axis=1)

        # Baseline C: Threshold Uncertainty (B3)
        b3_scores = df["B3_threshold_uncertainty_score"].values

        # Baseline D: Regresión Logística basada únicamente en probabilidades diagnósticas
        lr = LogisticRegression()
        # Train on val if available, or 5-fold cross fit
        val_df = pd.read_csv(RUNS_DIR / RUNS[idx_run] / "results" / "tables" / "master_predictions_val.csv")
        val_probs = val_df[[f"B1_HARD_DIAG_prob_{c}" for c in CLASSES]].values
        val_target = val_df["review_binary"].values
        lr.fit(val_probs, val_target)
        lr_probs = lr.predict_proba(probs_b1)[:, 1]

        # Modelo Propuesto HCG (P1)
        p1_review_scores = df["P1_MTL_BALANCED_PROPOSED_review_score"].values
        p0_review_scores = df["P0_MTL_UNBALANCED_review_score"].values

        methods = {
            "Entropy_H(p)": entropy_scores,
            "1-MSP (Confianza)": msp_scores,
            "Threshold_Uncertainty (B3)": b3_scores,
            "Logistic_Regression (Probs-only)": lr_probs,
            "Deep_Ensemble_Variance": ensemble_variance,
            "P0_MTL_Unbalanced_Head": p0_review_scores,
            "P1_MTL_Balanced_Head (Propuesto)": p1_review_scores,
        }

        for method_name, scores in methods.items():
            auroc = roc_auc_score(y_low_cert, scores)
            auprc = compute_auprc(y_low_cert, scores)
            seed_results.append({
                "run": RUNS[idx_run],
                "method": method_name,
                "auroc_low_cert": auroc,
                "auprc_low_cert": auprc
            })

    res_df = pd.DataFrame(seed_results)
    summary = res_df.groupby("method").agg(
        auroc_mean=("auroc_low_cert", "mean"),
        auroc_std=("auroc_low_cert", "std"),
        auprc_mean=("auprc_low_cert", "mean"),
        auprc_std=("auprc_low_cert", "std")
    ).sort_values("auprc_mean", ascending=False)

    out_dir = ROOT / "03_RESULTADOS" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "exp2_uncertainty_baselines_summary.csv")
    print(summary)
    print(f"\n[OK] Resultados guardados en: {out_dir / 'exp2_uncertainty_baselines_summary.csv'}")


if __name__ == "__main__":
    run_exp2()
