"""
Real gradient x input attribution for the HCG-SelectiveECG P1 model, computed on the
actual trained checkpoints (5 seeds) and the real PTB-XL records100 test signals.

Replaces the synthetic exp5_interpretability_attribution.py matrix with a genuine
per-seed, per-sample computation, aggregated into a 12-lead x 4-time-segment grid for
the diagnostic head and the certainty (review) head, plus quantitative dissociation
metrics (IoU of top attribution cells, % attribution by time segment, % attribution
by lead group) per seed, aggregated across seeds with mean/std/95% CI.

Environment variables (see README.md "Reproducing locally"):
  PTBXL_PROJECT_ROOT  -- folder containing data/raw/ptbxl_base/records100 (PTB-XL
                         records100 signals, downloaded separately from PhysioNet)
                         and runs_hcg_selective/<run>/models/<model>/checkpoints/best.pt
  RESULTS_ROOT         -- folder containing 03_RESULTADOS/runs/<run>/results/tables/
                         master_predictions_test.csv (released results package)
  ATTRIBUTION_OUT_DIR  -- output directory for CSV/JSON results (default: ./out)

Run with: python -I compute_real_attribution.py
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

PROJECT_ROOT = Path(os.environ.get("PTBXL_PROJECT_ROOT", "../ptbxl_likelihood_calibration")).resolve()
DELIVERY_ROOT = Path(os.environ.get("RESULTS_ROOT", "..")).resolve()
RECORDS100 = PROJECT_ROOT / "data" / "raw" / "ptbxl_base" / "records100"
RUNS_DIR = PROJECT_ROOT / "runs_hcg_selective"
DELIVERY_RUNS = DELIVERY_ROOT / "03_RESULTADOS" / "runs"
OUT_DIR = Path(os.environ.get("ATTRIBUTION_OUT_DIR", "./out")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TIME_SEGMENTS = ["P-wave (0-2.5s)", "QRS (2.5-5.0s)", "ST-T (5.0-7.5s)", "T-end (7.5-10s)"]
PRECORDIAL = {"V1", "V2", "V3", "V4", "V5", "V6"}
PERIPHERAL = {"I", "II", "III", "aVR", "aVL", "aVF"}

SEED_RUN_MAP = {
    42: "hcg_selective_ecg_full_20260712_131520",
    13: "hcg_selective_ecg_full_seed13",
    123: "hcg_selective_ecg_full_seed123",
    2024: "hcg_selective_ecg_full_seed2024",
    7: "hcg_selective_ecg_full_seed7",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dropout=0.0, pool=True):
        super().__init__()
        pad = kernel // 2
        layers = [nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=pad), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool1d(2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1DBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(12, 32, 7, dropout=0.20, pool=True),
            ConvBlock(32, 64, 7, dropout=0.25, pool=True),
            ConvBlock(64, 128, 5, dropout=0.25, pool=True),
            ConvBlock(128, 256, 5, dropout=0.0, pool=False),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HCGSelectiveECGModel(nn.Module):
    def __init__(self, multitask=True, n_classes=5):
        super().__init__()
        self.multitask = multitask
        self.backbone = CNN1DBackbone()
        self.dropout = nn.Dropout(0.30)
        self.diag_head = nn.Linear(256, n_classes)
        self.review_head = nn.Linear(256, 1) if multitask else None

    def forward(self, x):
        z = self.dropout(self.backbone(x))
        diag_logits = self.diag_head(z)
        if self.multitask:
            review_logit = self.review_head(z)
            return {"diag_logits": diag_logits, "review_logit": review_logit}
        return {"diag_logits": diag_logits}


def resolve_record_stem(ecg_id: int) -> Path:
    folder = f"{(ecg_id // 1000) * 1000:05d}"
    fname = f"{ecg_id:05d}_lr"
    return RECORDS100 / folder / fname


def read_ecg_record(stem: Path) -> np.ndarray:
    signal, _fields = wfdb.rdsamp(str(stem))
    x = signal.astype(np.float32)
    if x.shape == (1000, 12):
        x = x.T
    elif x.shape == (12, 1000):
        pass
    else:
        raise ValueError(f"Unexpected ECG shape {x.shape} for {stem}")
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x = (x - mean) / (std + 1e-8)
    return x.astype(np.float32)


def load_p1_model(seed: int) -> HCGSelectiveECGModel:
    run_name = SEED_RUN_MAP[seed]
    ckpt_path = RUNS_DIR / run_name / "models" / "P1_MTL_BALANCED_PROPOSED" / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HCGSelectiveECGModel(multitask=True, n_classes=5)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    model.to(DEVICE)
    return model


def segment_pool(attr_1000: np.ndarray) -> np.ndarray:
    """attr_1000: (12, 1000) -> (12, 4) mean |attr| per 250-sample segment."""
    seg = attr_1000.reshape(12, 4, 250)
    return np.mean(np.abs(seg), axis=2)


def compute_grad_attribution(model, x_np, y_hard_row):
    """x_np: (12,1000) normalized signal. y_hard_row: (5,) binary true labels.
    Returns (diag_attr_12x1000, cert_attr_12x1000) raw grad*input (signed)."""
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    x.requires_grad_(True)

    # --- diagnostic attribution: sum of logits over true positive classes ---
    model.zero_grad(set_to_none=True)
    out = model(x)
    pos_idx = np.where(y_hard_row > 0.5)[0]
    if len(pos_idx) == 0:
        pos_idx = np.array([0])  # fall back to NORM if no positive label
    diag_score = out["diag_logits"][0, pos_idx].sum()
    diag_score.backward(retain_graph=True)
    diag_attr = (x.grad.detach() * x.detach()).squeeze(0).cpu().numpy()

    # --- certainty attribution: sigmoid(review_logit) ---
    x.grad = None
    model.zero_grad(set_to_none=True)
    out2 = model(x)
    cert_score = torch.sigmoid(out2["review_logit"])[0, 0]
    cert_score.backward()
    cert_attr = (x.grad.detach() * x.detach()).squeeze(0).cpu().numpy()

    return diag_attr, cert_attr


def iou_top_k(mat_a: np.ndarray, mat_b: np.ndarray, frac=0.25):
    """IoU of the top `frac` cells (by value) between two 12x4 matrices."""
    n = mat_a.size
    k = max(1, round(n * frac))
    top_a = set(np.argsort(mat_a.ravel())[-k:])
    top_b = set(np.argsort(mat_b.ravel())[-k:])
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union > 0 else float("nan")


def segment_share(mat: np.ndarray, seg_idx: int) -> float:
    """Share of total |attribution| mass in a given time segment, summed over leads."""
    total = mat.sum()
    return float(mat[:, seg_idx].sum() / total) if total > 0 else float("nan")


def lead_group_share(mat: np.ndarray, lead_names, group: set) -> float:
    total = mat.sum()
    idx = [i for i, l in enumerate(lead_names) if l in group]
    return float(mat[idx, :].sum() / total) if total > 0 else float("nan")


def main():
    per_seed_rows = []
    per_seed_matrices = {}

    for seed, run_name in SEED_RUN_MAP.items():
        print(f"=== seed {seed} ({run_name}) ===", flush=True)
        model = load_p1_model(seed)
        master_csv = DELIVERY_RUNS / run_name / "results" / "tables" / "master_predictions_test.csv"
        df = pd.read_csv(master_csv)
        y_cols = [f"y_{c}" for c in CLASSES]

        diag_acc = np.zeros((12, 4), dtype=np.float64)
        cert_acc = np.zeros((12, 4), dtype=np.float64)
        n_ok = 0
        n_fail = 0

        for i, row in df.iterrows():
            ecg_id = int(row["ecg_id"])
            try:
                stem = resolve_record_stem(ecg_id)
                x_np = read_ecg_record(stem)
            except Exception as e:
                n_fail += 1
                continue
            y_hard = row[y_cols].to_numpy(dtype=np.float32)
            diag_attr, cert_attr = compute_grad_attribution(model, x_np, y_hard)
            diag_acc += segment_pool(diag_attr)
            cert_acc += segment_pool(cert_attr)
            n_ok += 1
            if n_ok % 500 == 0:
                print(f"  seed {seed}: {n_ok} samples processed, {n_fail} failed", flush=True)

        diag_mat = diag_acc / max(n_ok, 1)
        cert_mat = cert_acc / max(n_ok, 1)

        diag_norm = diag_mat / diag_mat.max()
        cert_norm = cert_mat / cert_mat.max()

        iou25 = iou_top_k(diag_norm, cert_norm, frac=0.25)

        qrs_idx = TIME_SEGMENTS.index("QRS (2.5-5.0s)")
        stt_idx = TIME_SEGMENTS.index("ST-T (5.0-7.5s)")
        diag_qrs_share = segment_share(diag_mat, qrs_idx)
        diag_stt_share = segment_share(diag_mat, stt_idx)
        cert_qrs_share = segment_share(cert_mat, qrs_idx)
        cert_stt_share = segment_share(cert_mat, stt_idx)

        diag_precordial_share = lead_group_share(diag_mat, LEAD_NAMES, PRECORDIAL)
        diag_peripheral_share = lead_group_share(diag_mat, LEAD_NAMES, PERIPHERAL)
        cert_precordial_share = lead_group_share(cert_mat, LEAD_NAMES, PRECORDIAL)
        cert_peripheral_share = lead_group_share(cert_mat, LEAD_NAMES, PERIPHERAL)

        per_seed_rows.append({
            "seed": seed, "n_samples": n_ok, "n_failed": n_fail,
            "iou_top25pct": iou25,
            "diag_qrs_share": diag_qrs_share, "diag_stt_share": diag_stt_share,
            "cert_qrs_share": cert_qrs_share, "cert_stt_share": cert_stt_share,
            "diag_precordial_share": diag_precordial_share, "diag_peripheral_share": diag_peripheral_share,
            "cert_precordial_share": cert_precordial_share, "cert_peripheral_share": cert_peripheral_share,
        })
        per_seed_matrices[seed] = {"diag": diag_mat.tolist(), "cert": cert_mat.tolist()}
        print(f"  seed {seed} DONE: n_ok={n_ok} n_fail={n_fail} iou25={iou25:.4f}", flush=True)

    summary_df = pd.DataFrame(per_seed_rows)
    summary_df.to_csv(OUT_DIR / "real_attribution_per_seed.csv", index=False)

    with open(OUT_DIR / "real_attribution_matrices.json", "w", encoding="utf-8") as f:
        json.dump({"leads": LEAD_NAMES, "segments": TIME_SEGMENTS, "per_seed": per_seed_matrices}, f, indent=2)

    # aggregate across seeds: mean, std, 95% CI (t, df=4 -> t_crit=2.776)
    t_crit = 2.776
    agg_rows = []
    for col in summary_df.columns:
        if col in ("seed", "n_samples", "n_failed"):
            continue
        vals = summary_df[col].to_numpy(dtype=np.float64)
        n = len(vals)
        mean = vals.mean()
        sd = vals.std(ddof=1)
        half = t_crit * sd / math.sqrt(n)
        agg_rows.append({"metric": col, "n_seeds": n, "mean": mean, "sd": sd,
                          "ci95_low": mean - half, "ci95_high": mean + half})
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "real_attribution_aggregate.csv", index=False)

    # pooled (mean-of-seeds) matrices for the figure
    diag_pooled = np.mean([np.array(per_seed_matrices[s]["diag"]) for s in SEED_RUN_MAP], axis=0)
    cert_pooled = np.mean([np.array(per_seed_matrices[s]["cert"]) for s in SEED_RUN_MAP], axis=0)
    diag_pooled_norm = diag_pooled / diag_pooled.max()
    cert_pooled_norm = cert_pooled / cert_pooled.max()

    rows = []
    for l_idx, lead in enumerate(LEAD_NAMES):
        for t_idx, seg in enumerate(TIME_SEGMENTS):
            rows.append({
                "lead": lead, "time_segment": seg,
                "diagnostic_attribution": float(diag_pooled_norm[l_idx, t_idx]),
                "certainty_attribution": float(cert_pooled_norm[l_idx, t_idx]),
            })
    pooled_df = pd.DataFrame(rows)
    pooled_df.to_csv(OUT_DIR / "real_exp5_attribution_matrix.csv", index=False)

    print("\n=== SUMMARY ACROSS SEEDS ===")
    print(agg_df.to_string(index=False))
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
