"""
Integrated Gradients attribution (Sundararajan et al. 2017) for the HCG-SelectiveECG
P1 model, on the real trained checkpoints (5 seeds) and real PTB-XL test signals.

More robust than plain gradient x input: satisfies the completeness axiom (sum of
attributions ~= F(x) - F(baseline)), which we check as a sanity test, and averages
gradients along a path instead of relying on a single noisy local gradient.

Uses the same PTBXL_PROJECT_ROOT / RESULTS_ROOT / ATTRIBUTION_OUT_DIR environment
variables as compute_real_attribution.py (see README.md "Reproducing locally").

Run with: python -I compute_integrated_gradients.py
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from compute_real_attribution import (
    DEVICE, SEED_RUN_MAP, CLASSES, LEAD_NAMES, TIME_SEGMENTS, PRECORDIAL, PERIPHERAL,
    DELIVERY_RUNS, load_p1_model, resolve_record_stem, read_ecg_record,
    segment_pool, iou_top_k, segment_share, lead_group_share,
)

OUT_DIR = Path(os.environ.get("ATTRIBUTION_OUT_DIR", "./out")).resolve()
N_STEPS = 50


def integrated_gradients(model, x_np, y_hard_row, n_steps=N_STEPS):
    """Returns (diag_ig, cert_ig, diag_convergence_err, cert_convergence_err)."""
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    baseline = torch.zeros_like(x)

    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)  # (n_steps,12,1000)
    path.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    out = model(path)

    pos_idx = np.where(y_hard_row > 0.5)[0]
    if len(pos_idx) == 0:
        pos_idx = np.array([0])
    diag_target = out["diag_logits"][:, pos_idx].sum(dim=1)  # (n_steps,)
    diag_grads = torch.autograd.grad(diag_target.sum(), path, retain_graph=True)[0]
    avg_diag_grad = diag_grads.mean(dim=0)
    diag_ig = ((x - baseline) * avg_diag_grad).detach().cpu().numpy()

    path.grad = None
    model.zero_grad(set_to_none=True)
    out2 = model(path)
    cert_target = torch.sigmoid(out2["review_logit"])[:, 0]
    cert_grads = torch.autograd.grad(cert_target.sum(), path)[0]
    avg_cert_grad = cert_grads.mean(dim=0)
    cert_ig = ((x - baseline) * avg_cert_grad).detach().cpu().numpy()

    with torch.no_grad():
        f_x = model(x.unsqueeze(0))
        f_base = model(baseline.unsqueeze(0))
        diag_f_diff = (f_x["diag_logits"][0, pos_idx].sum() - f_base["diag_logits"][0, pos_idx].sum()).item()
        cert_f_diff = (torch.sigmoid(f_x["review_logit"])[0, 0] - torch.sigmoid(f_base["review_logit"])[0, 0]).item()

    diag_conv_err = diag_ig.sum() - diag_f_diff
    cert_conv_err = cert_ig.sum() - cert_f_diff
    return diag_ig, cert_ig, diag_conv_err, cert_conv_err


def main():
    per_seed_rows = []
    per_seed_matrices = {}
    convergence_rows = []

    for seed, run_name in SEED_RUN_MAP.items():
        print(f"=== seed {seed} ({run_name}) IG ===", flush=True)
        model = load_p1_model(seed)
        master_csv = DELIVERY_RUNS / run_name / "results" / "tables" / "master_predictions_test.csv"
        df = pd.read_csv(master_csv)
        y_cols = [f"y_{c}" for c in CLASSES]

        diag_acc = np.zeros((12, 4), dtype=np.float64)
        cert_acc = np.zeros((12, 4), dtype=np.float64)
        n_ok = 0
        diag_conv_errs = []
        cert_conv_errs = []

        for i, row in df.iterrows():
            ecg_id = int(row["ecg_id"])
            try:
                stem = resolve_record_stem(ecg_id)
                x_np = read_ecg_record(stem)
            except Exception:
                continue
            y_hard = row[y_cols].to_numpy(dtype=np.float32)
            diag_ig, cert_ig, d_err, c_err = integrated_gradients(model, x_np, y_hard)
            diag_acc += segment_pool(diag_ig)
            cert_acc += segment_pool(cert_ig)
            diag_conv_errs.append(d_err)
            cert_conv_errs.append(c_err)
            n_ok += 1
            if n_ok % 500 == 0:
                print(f"  seed {seed}: {n_ok} samples processed", flush=True)

        diag_mat = diag_acc / max(n_ok, 1)
        cert_mat = cert_acc / max(n_ok, 1)
        diag_norm = diag_mat / diag_mat.max()
        cert_norm = cert_mat / cert_mat.max()

        iou25 = iou_top_k(diag_norm, cert_norm, frac=0.25)
        qrs_idx = TIME_SEGMENTS.index("QRS (2.5-5.0s)")
        stt_idx = TIME_SEGMENTS.index("ST-T (5.0-7.5s)")

        per_seed_rows.append({
            "seed": seed, "n_samples": n_ok,
            "iou_top25pct": iou25,
            "diag_qrs_share": segment_share(diag_mat, qrs_idx),
            "diag_stt_share": segment_share(diag_mat, stt_idx),
            "cert_qrs_share": segment_share(cert_mat, qrs_idx),
            "cert_stt_share": segment_share(cert_mat, stt_idx),
            "diag_precordial_share": lead_group_share(diag_mat, LEAD_NAMES, PRECORDIAL),
            "diag_peripheral_share": lead_group_share(diag_mat, LEAD_NAMES, PERIPHERAL),
            "cert_precordial_share": lead_group_share(cert_mat, LEAD_NAMES, PRECORDIAL),
            "cert_peripheral_share": lead_group_share(cert_mat, LEAD_NAMES, PERIPHERAL),
        })
        per_seed_matrices[seed] = {"diag": diag_mat.tolist(), "cert": cert_mat.tolist()}

        d_mean_abs_err = float(np.mean(np.abs(diag_conv_errs)))
        c_mean_abs_err = float(np.mean(np.abs(cert_conv_errs)))
        d_mean_f = float(np.mean(np.abs(diag_acc)))  # rough scale reference
        convergence_rows.append({"seed": seed, "diag_mean_abs_convergence_err": d_mean_abs_err,
                                  "cert_mean_abs_convergence_err": c_mean_abs_err})
        print(f"  seed {seed} DONE n_ok={n_ok} iou25={iou25:.4f} diag_conv_err={d_mean_abs_err:.5f} cert_conv_err={c_mean_abs_err:.5f}", flush=True)

    summary_df = pd.DataFrame(per_seed_rows)
    summary_df.to_csv(OUT_DIR / "ig_attribution_per_seed.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(OUT_DIR / "ig_convergence_check.csv", index=False)

    with open(OUT_DIR / "ig_attribution_matrices.json", "w", encoding="utf-8") as f:
        json.dump({"leads": LEAD_NAMES, "segments": TIME_SEGMENTS, "per_seed": per_seed_matrices}, f, indent=2)

    t_crit = 2.776
    agg_rows = []
    for col in summary_df.columns:
        if col in ("seed", "n_samples"):
            continue
        vals = summary_df[col].to_numpy(dtype=np.float64)
        n = len(vals)
        mean = vals.mean()
        sd = vals.std(ddof=1)
        half = t_crit * sd / math.sqrt(n)
        agg_rows.append({"metric": col, "n_seeds": n, "mean": mean, "sd": sd,
                          "ci95_low": mean - half, "ci95_high": mean + half})
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "ig_attribution_aggregate.csv", index=False)

    diag_pooled = np.mean([np.array(per_seed_matrices[s]["diag"]) for s in SEED_RUN_MAP], axis=0)
    cert_pooled = np.mean([np.array(per_seed_matrices[s]["cert"]) for s in SEED_RUN_MAP], axis=0)
    diag_pooled_norm = diag_pooled / diag_pooled.max()
    cert_pooled_norm = cert_pooled / cert_pooled.max()
    rows = []
    for l_idx, lead in enumerate(LEAD_NAMES):
        for t_idx, seg in enumerate(TIME_SEGMENTS):
            rows.append({"lead": lead, "time_segment": seg,
                         "diagnostic_attribution": float(diag_pooled_norm[l_idx, t_idx]),
                         "certainty_attribution": float(cert_pooled_norm[l_idx, t_idx])})
    pd.DataFrame(rows).to_csv(OUT_DIR / "ig_exp5_attribution_matrix.csv", index=False)

    print("\n=== IG SUMMARY ACROSS SEEDS ===")
    print(agg_df.to_string(index=False))
    print("\n=== Convergence check (should be small relative to logit scale) ===")
    print(pd.DataFrame(convergence_rows).to_string(index=False))


if __name__ == "__main__":
    main()
