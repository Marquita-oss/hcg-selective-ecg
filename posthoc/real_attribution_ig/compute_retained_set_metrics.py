"""
Retained-set (purification) analysis: does referring the lowest-certainty q% improve
or degrade the diagnostic quality of the remaining (1-q)% for each referral method?

Env vars: RESULTS_ROOT (folder containing 03_RESULTADOS/runs/...), ATTRIBUTION_OUT_DIR
(output folder, default ./out). See README.md "Reproducing locally".
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "..")).resolve()
DELIVERY_RUNS = RESULTS_ROOT / "03_RESULTADOS" / "runs"
SEED_MAP = {
    42: "hcg_selective_ecg_full_20260712_131520",
    13: "hcg_selective_ecg_full_seed13",
    123: "hcg_selective_ecg_full_seed123",
    2024: "hcg_selective_ecg_full_seed2024",
    7: "hcg_selective_ecg_full_seed7",
}
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
METHODS = [
    ("B3", "B1_HARD_DIAG", "B3_THRESHOLD_UNCERTAINTY_referred_top20"),
    ("P0", "P0_MTL_UNBALANCED", "P0_MTL_UNBALANCED_referred_top20"),
    ("P1", "P1_MTL_BALANCED_PROPOSED", "P1_MTL_BALANCED_PROPOSED_referred_top20"),
    ("RANDOM", "B1_HARD_DIAG", "RANDOM_REFERRAL_referred_top20"),
]

rows = []
for seed, run_name in SEED_MAP.items():
    df = pd.read_csv(DELIVERY_RUNS / run_name / "results" / "tables" / "master_predictions_test.csv")
    for name, prob_prefix, refer_col in METHODS:
        y = df[[f"y_{c}" for c in CLASSES]].to_numpy()
        probs = df[[f"{prob_prefix}_prob_{c}" for c in CLASSES]].to_numpy()

        auroc_full = np.mean([roc_auc_score(y[:, k], probs[:, k]) for k in range(5)])
        auprc_full = np.mean([average_precision_score(y[:, k], probs[:, k]) for k in range(5)])

        mask = df[refer_col].to_numpy() == 0
        y_ret, probs_ret = y[mask], probs[mask]
        auroc_ret = np.mean([roc_auc_score(y_ret[:, k], probs_ret[:, k]) for k in range(5)])
        auprc_ret = np.mean([average_precision_score(y_ret[:, k], probs_ret[:, k]) for k in range(5)])

        rows.append({
            "seed": seed, "method": name, "n_total": len(df), "n_retained": int(mask.sum()),
            "AUROC_full": auroc_full, "AUPRC_full": auprc_full,
            "AUROC_retained": auroc_ret, "AUPRC_retained": auprc_ret,
            "AUROC_delta": auroc_ret - auroc_full, "AUPRC_delta": auprc_ret - auprc_full,
        })

per_seed = pd.DataFrame(rows)
out_dir = Path(os.environ.get("ATTRIBUTION_OUT_DIR", "./out")).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
per_seed.to_csv(out_dir / "retained_set_per_seed_sklearn.csv", index=False)

agg_rows = []
for name, *_ in METHODS:
    sub = per_seed[per_seed.method == name]
    for col in ["AUROC_full", "AUPRC_full", "AUROC_retained", "AUPRC_retained", "AUROC_delta", "AUPRC_delta"]:
        vals = sub[col].to_numpy()
        n = len(vals)
        mean = vals.mean()
        sd = vals.std(ddof=1)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        half = tcrit * sd / np.sqrt(n)
        agg_rows.append({"method": name, "metric": col, "n": n, "mean": mean, "sd": sd,
                          "ci95_low": mean - half, "ci95_high": mean + half})
agg = pd.DataFrame(agg_rows)
agg.to_csv(out_dir / "retained_set_aggregate_sklearn.csv", index=False)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(agg.pivot(index="method", columns="metric", values="mean").to_string())
print()
print("=== deltas (retained - full) with 95% CI ===")
delta_view = agg[agg.metric.isin(["AUROC_delta", "AUPRC_delta"])]
print(delta_view[["method", "metric", "mean", "ci95_low", "ci95_high"]].to_string(index=False))
