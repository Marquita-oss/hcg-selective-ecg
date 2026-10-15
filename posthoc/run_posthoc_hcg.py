from __future__ import annotations

import argparse
import io
import json
import math
import re
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False
    from PIL import Image, ImageDraw, ImageFont

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
QS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CI_MULTIPLIER_5_SEEDS = 2.776 / math.sqrt(5)
RUN_ROOT = "ptbxl_likelihood_calibration/runs_hcg_selective"


@dataclass(frozen=True)
class ScoreSpec:
    key: str
    label: str
    score_col: str | None
    diag_model: str


SCORES = [
    ScoreSpec("B3", "B3 umbral", "B3_threshold_uncertainty_score", "B1_HARD_DIAG"),
    ScoreSpec("P0", "P0 multitarea", "P0_MTL_UNBALANCED_review_score", "P0_MTL_UNBALANCED"),
    ScoreSpec(
        "P1",
        "P1 multitarea balanceado",
        "P1_MTL_BALANCED_PROPOSED_review_score",
        "P1_MTL_BALANCED_PROPOSED",
    ),
    ScoreSpec("B2prob", "B2 incertidumbre prob.", None, "B2_SOFT_DIRECT"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisis post-hoc HCG-SelectiveECG desde ZIP de resultados."
    )
    parser.add_argument(
        "--zip",
        required=True,
        help="Ruta al ZIP ptbxl_likelihood_calibration.zip recibido.",
    )
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1]
            / "03_RESULTADOS"
            / "posthoc_recalculado"
        ),
        help="Carpeta de salida.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def ensure_dirs(out: Path) -> dict[str, Path]:
    dirs = {
        "root": out,
        "tables": out / "tables",
        "figures": out / "figures",
        "reports": out / "reports",
        "latex": out / "latex_tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def zip_names(zf: zipfile.ZipFile) -> set[str]:
    return set(zf.namelist())


def discover_runs(names: set[str]) -> list[str]:
    runs = []
    pat = re.compile(rf"{re.escape(RUN_ROOT)}/([^/]+)/")
    for name in names:
        m = pat.search(name)
        if not m:
            continue
        run = m.group(1)
        if run.startswith("_aggregate"):
            continue
        if run not in runs:
            runs.append(run)
    return sorted(runs)


def read_csv(zf: zipfile.ZipFile, rel_path: str) -> pd.DataFrame:
    with zf.open(rel_path) as handle:
        return pd.read_csv(handle)


def read_json(zf: zipfile.ZipFile, rel_path: str) -> dict:
    with zf.open(rel_path) as handle:
        return json.load(handle)


def run_path(run: str, rel: str) -> str:
    return f"{RUN_ROOT}/{run}/{rel}"


def agg_path(rel: str) -> str:
    return f"{RUN_ROOT}/_aggregate_multiseed/{rel}"


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean_ci(values: list[float] | pd.Series | np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) == 5:
        half = CI_MULTIPLIER_5_SEEDS * sd
    elif len(arr) > 1:
        half = 1.96 * sd / math.sqrt(len(arr))
    else:
        half = 0.0
    return mean, sd, mean - half, mean + half


def fmt(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{nd}f}"


def write_text(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def add_b2_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [f"B2_SOFT_DIRECT_prob_{c}" for c in CLASSES]
    if all(c in df.columns for c in prob_cols):
        probs = df[prob_cols].to_numpy(dtype=float)
        df = df.copy()
        df["B2_prob_uncertainty_score"] = np.max(1.0 - np.abs(2.0 * probs - 1.0), axis=1)
    return df


def referred_mask(df: pd.DataFrame, spec: ScoreSpec, q: float) -> np.ndarray:
    if spec.key == "B2prob":
        col = "B2_prob_uncertainty_score"
    else:
        col = spec.score_col
    if col not in df.columns:
        raise KeyError(f"Falta score {col}")
    n_ref = max(1, int(round(q * len(df))))
    scores = df[col].to_numpy(dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(len(df), dtype=bool)
    mask[order[:n_ref]] = True
    return mask


def referral_metrics(df: pd.DataFrame, spec: ScoreSpec, q: float) -> dict[str, float]:
    mask = referred_mask(df, spec, q)
    low = df["review_binary"].to_numpy(dtype=int).astype(bool)
    err = df[f"{spec.diag_model}_sample_error"].to_numpy(dtype=int).astype(bool)
    n_ref = int(mask.sum())
    low_total = int(low.sum())
    err_total = int(err.sum())
    low_in_ref = int((mask & low).sum())
    err_in_ref = int((mask & err).sum())
    low_prev = float(low.mean())
    precision_low = low_in_ref / n_ref if n_ref else np.nan
    lift_low = precision_low / low_prev if low_prev > 0 else np.nan
    nnr_low = 1.0 / precision_low if precision_low > 0 else np.inf
    return {
        "n": len(df),
        "q": q,
        "n_referred": n_ref,
        "low_cert_prevalence": low_prev,
        "LowCertCapture@q": low_in_ref / low_total if low_total else np.nan,
        "LowCertPrecision@q": precision_low,
        "LowCertLift@q": lift_low,
        "NNR_low_cert@q": nnr_low,
        "ErrorCapture@q": err_in_ref / err_total if err_total else np.nan,
        "ErrorPrecision@q": err_in_ref / n_ref if n_ref else np.nan,
    }


def aggregate_by_seed(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = g["seed"].nunique()
        for metric in metrics:
            mean, sd, lo, hi = mean_ci(g[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_seed_mean_diff(
    dfs_by_seed: dict[str, pd.DataFrame],
    spec_a: ScoreSpec,
    spec_b: ScoreSpec,
    q: float,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    base = next(iter(dfs_by_seed.values()))
    patient_ids = pd.Series(base["patient_id"].dropna().unique()).to_numpy()
    boot_diffs = []
    groups_by_seed = {}
    for seed, df in dfs_by_seed.items():
        groups = {
            pid: idx.to_numpy()
            for pid, idx in df.groupby("patient_id").groups.items()
            if not pd.isna(pid)
        }
        groups_by_seed[seed] = groups
    for _ in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        seed_diffs = []
        for seed, df in dfs_by_seed.items():
            groups = groups_by_seed[seed]
            idx_parts = [groups[pid] for pid in sampled if pid in groups]
            if not idx_parts:
                continue
            idx = np.concatenate(idx_parts)
            sample_df = df.iloc[idx].reset_index(drop=True)
            ma = referral_metrics(sample_df, spec_a, q)[metric]
            mb = referral_metrics(sample_df, spec_b, q)[metric]
            seed_diffs.append(ma - mb)
        if seed_diffs:
            boot_diffs.append(float(np.mean(seed_diffs)))
    arr = np.asarray(boot_diffs, dtype=float)
    obs_diffs = []
    for df in dfs_by_seed.values():
        obs_diffs.append(referral_metrics(df, spec_a, q)[metric] - referral_metrics(df, spec_b, q)[metric])
    obs = float(np.mean(obs_diffs))
    lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
    p = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))) if len(arr) else np.nan
    return {
        "comparison": f"{spec_a.key}-{spec_b.key}",
        "q": q,
        "metric": metric,
        "observed_diff_seed_mean": obs,
        "ci95_low_patient_bootstrap": float(lo),
        "ci95_high_patient_bootstrap": float(hi),
        "p_bootstrap_two_sided": min(p, 1.0) if np.isfinite(p) else np.nan,
        "n_bootstrap": n_boot,
        "bootstrap_unit": "patient",
    }


def _score_col_for_spec(spec: ScoreSpec) -> str:
    if spec.key == "B2prob":
        return "B2_prob_uncertainty_score"
    if spec.score_col is None:
        raise KeyError(spec.key)
    return spec.score_col


def _metric_bundle_from_arrays(
    scores: np.ndarray,
    low: np.ndarray,
    err: np.ndarray,
    idx: np.ndarray,
    q: float,
) -> dict[str, float]:
    n = len(idx)
    n_ref = max(1, int(round(q * n)))
    idx_scores = scores[idx]
    if n_ref >= n:
        sel_local = np.arange(n)
    else:
        sel_local = np.argpartition(-idx_scores, n_ref - 1)[:n_ref]
    selected_idx = idx[sel_local]
    low_idx = low[idx]
    err_idx = err[idx]
    low_sel = low[selected_idx]
    err_sel = err[selected_idx]
    low_total = int(low_idx.sum())
    err_total = int(err_idx.sum())
    precision_low = float(low_sel.sum() / n_ref) if n_ref else np.nan
    low_prev = float(low_idx.mean()) if n else np.nan
    return {
        "LowCertCapture@q": float(low_sel.sum() / low_total) if low_total else np.nan,
        "LowCertPrecision@q": precision_low,
        "LowCertLift@q": float(precision_low / low_prev) if low_prev > 0 else np.nan,
        "ErrorCapture@q": float(err_sel.sum() / err_total) if err_total else np.nan,
    }


def bootstrap_seed_mean_diffs_multi(
    dfs_by_seed: dict[str, pd.DataFrame],
    spec_a: ScoreSpec,
    spec_b: ScoreSpec,
    q: float,
    metrics: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    first_df = next(iter(dfs_by_seed.values()))
    patient_ids = pd.Series(first_df["patient_id"].dropna().unique()).to_numpy()
    prepared = {}
    for seed, df in dfs_by_seed.items():
        groups = {
            pid: np.asarray(idx, dtype=int)
            for pid, idx in df.groupby("patient_id").groups.items()
            if not pd.isna(pid)
        }
        prepared[seed] = {
            "groups": groups,
            "low": df["review_binary"].to_numpy(dtype=int).astype(bool),
            "score_a": df[_score_col_for_spec(spec_a)].to_numpy(dtype=float),
            "score_b": df[_score_col_for_spec(spec_b)].to_numpy(dtype=float),
            "err_a": df[f"{spec_a.diag_model}_sample_error"].to_numpy(dtype=int).astype(bool),
            "err_b": df[f"{spec_b.diag_model}_sample_error"].to_numpy(dtype=int).astype(bool),
        }

    boot_by_metric = {metric: [] for metric in metrics}
    for _ in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        seed_diffs_by_metric = {metric: [] for metric in metrics}
        for prep in prepared.values():
            idx_parts = [prep["groups"][pid] for pid in sampled if pid in prep["groups"]]
            if not idx_parts:
                continue
            idx = np.concatenate(idx_parts)
            ma = _metric_bundle_from_arrays(prep["score_a"], prep["low"], prep["err_a"], idx, q)
            mb = _metric_bundle_from_arrays(prep["score_b"], prep["low"], prep["err_b"], idx, q)
            for metric in metrics:
                seed_diffs_by_metric[metric].append(ma[metric] - mb[metric])
        for metric in metrics:
            if seed_diffs_by_metric[metric]:
                boot_by_metric[metric].append(float(np.mean(seed_diffs_by_metric[metric])))

    obs_by_metric = {metric: [] for metric in metrics}
    for df in dfs_by_seed.values():
        ma = referral_metrics(df, spec_a, q)
        mb = referral_metrics(df, spec_b, q)
        for metric in metrics:
            obs_by_metric[metric].append(ma[metric] - mb[metric])

    rows = []
    for metric in metrics:
        arr = np.asarray(boot_by_metric[metric], dtype=float)
        lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
        p = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))) if len(arr) else np.nan
        rows.append(
            {
                "comparison": f"{spec_a.key}-{spec_b.key}",
                "q": q,
                "metric": metric,
                "observed_diff_seed_mean": float(np.mean(obs_by_metric[metric])),
                "ci95_low_patient_bootstrap": float(lo),
                "ci95_high_patient_bootstrap": float(hi),
                "p_bootstrap_two_sided": min(p, 1.0) if np.isfinite(p) else np.nan,
                "n_bootstrap": n_boot,
                "bootstrap_unit": "patient",
            }
        )
    return pd.DataFrame(rows)


def brier_macro(df: pd.DataFrame, model: str) -> float:
    vals = []
    for cls in CLASSES:
        p = df[f"{model}_prob_{cls}"].to_numpy(dtype=float)
        y = df[f"y_{cls}"].to_numpy(dtype=float)
        vals.append(np.mean((p - y) ** 2))
    return float(np.mean(vals))


def ece_binary(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out += (mask.sum() / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(out)


def ece_macro(df: pd.DataFrame, model: str, n_bins: int = 10) -> float:
    vals = []
    for cls in CLASSES:
        p = df[f"{model}_prob_{cls}"].to_numpy(dtype=float)
        y = df[f"y_{cls}"].to_numpy(dtype=float)
        vals.append(ece_binary(y, p, n_bins=n_bins))
    return float(np.mean(vals))


def logloss_macro(df: pd.DataFrame, model: str) -> float:
    vals = []
    eps = 1e-7
    for cls in CLASSES:
        p = np.clip(df[f"{model}_prob_{cls}"].to_numpy(dtype=float), eps, 1 - eps)
        y = df[f"y_{cls}"].to_numpy(dtype=float)
        vals.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean(vals))


def diagnostic_discrimination(df: pd.DataFrame, model: str) -> dict[str, float]:
    y = df[[f"y_{c}" for c in CLASSES]].to_numpy(dtype=int)
    p = df[[f"{model}_prob_{c}" for c in CLASSES]].to_numpy(dtype=float)
    if roc_auc_score is None or average_precision_score is None:
        return {"AUROC_macro": np.nan, "AUPRC_macro": np.nan}
    try:
        auroc = float(roc_auc_score(y, p, average="macro"))
    except Exception:
        auroc = np.nan
    try:
        auprc = float(average_precision_score(y, p, average="macro"))
    except Exception:
        auprc = np.nan
    return {"AUROC_macro": auroc, "AUPRC_macro": auprc}


def calibration_table(dfs_by_seed: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = ["B1_HARD_DIAG", "B2_SOFT_DIRECT"]
    rows = []
    per_class_rows = []
    for seed, df in dfs_by_seed.items():
        for model in models:
            disc = diagnostic_discrimination(df, model)
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "Brier_macro": brier_macro(df, model),
                    "ECE_macro_10bins": ece_macro(df, model, n_bins=10),
                    "ECE_macro_15bins": ece_macro(df, model, n_bins=15),
                    "LogLoss_macro": logloss_macro(df, model),
                    **disc,
                }
            )
            for cls in CLASSES:
                y = df[f"y_{cls}"].to_numpy(dtype=float)
                p = df[f"{model}_prob_{cls}"].to_numpy(dtype=float)
                per_class_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "class": cls,
                        "Brier": float(np.mean((p - y) ** 2)),
                        "ECE_10bins": ece_binary(y, p, n_bins=10),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(per_class_rows)


def paired_diff_table(metric_df: pd.DataFrame, model_a: str, model_b: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    pivot = metric_df.pivot(index="seed", columns="model", values=metrics)
    for metric in metrics:
        a = pivot[(metric, model_a)]
        b = pivot[(metric, model_b)]
        diff = a - b
        mean, sd, lo, hi = mean_ci(diff)
        rows.append(
            {
                "comparison": f"{model_a} - {model_b}",
                "metric": metric,
                "diff_mean": mean,
                "diff_sd": sd,
                "diff_ci95_low": lo,
                "diff_ci95_high": hi,
                "n_seeds": len(diff),
                "seeds_improved_for_a": int((diff < 0).sum()) if metric in ["Brier_macro", "ECE_macro_10bins", "ECE_macro_15bins", "LogLoss_macro"] else int((diff > 0).sum()),
                "direction_note": "negative favors first model" if metric in ["Brier_macro", "ECE_macro_10bins", "ECE_macro_15bins", "LogLoss_macro"] else "positive favors first model",
            }
        )
    return pd.DataFrame(rows)


def class_capture(df: pd.DataFrame, spec: ScoreSpec, q: float) -> list[dict[str, float | str]]:
    mask = referred_mask(df, spec, q)
    rows = []
    for cls in CLASSES:
        y = df[f"y_{cls}"].to_numpy(dtype=int).astype(bool)
        pred = df[f"{spec.diag_model}_pred_{cls}_thrval"].to_numpy(dtype=int).astype(bool)
        fn = y & ~pred
        fp = ~y & pred
        ce = fn | fp
        rows.append(
            {
                "score": spec.key,
                "score_label": spec.label,
                "diag_model": spec.diag_model,
                "q": q,
                "class": cls,
                "n_class_error": int(ce.sum()),
                "n_false_negative": int(fn.sum()),
                "n_false_positive": int(fp.sum()),
                "ClassErrorCapture@q": float((mask & ce).sum() / ce.sum()) if ce.sum() else np.nan,
                "FalseNegativeCapture@q": float((mask & fn).sum() / fn.sum()) if fn.sum() else np.nan,
                "FalsePositiveCapture@q": float((mask & fp).sum() / fp.sum()) if fp.sum() else np.nan,
            }
        )
    return rows


def _rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.strip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _font(size: int, bold: bool = False):
    if HAS_MPL:
        return None
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _save_pil(img, base: Path) -> None:
    img.save(base.with_suffix(".png"), dpi=(300, 300))
    rgb = img.convert("RGB")
    rgb.save(base.with_suffix(".pdf"), "PDF", resolution=300.0)


def _draw_line(draw, points, fill, width=3):
    if len(points) > 1:
        draw.line(points, fill=fill, width=width, joint="curve")
    for x, y in points:
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255), width=1)


def _panel_map(x, y, rect, xmin, xmax, ymin, ymax):
    x0, y0, x1, y1 = rect
    px = x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
    py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
    return float(px), float(py)


def _draw_axes(draw, rect, title, xlabel, ylabel=None, xticks=None, yticks=None):
    x0, y0, x1, y1 = rect
    draw.rectangle((x0, y0, x1, y1), outline=(55, 55, 55), width=1)
    for t in yticks or []:
        yy = y1 - t * (y1 - y0)
        draw.line((x0, yy, x1, yy), fill=(225, 225, 225), width=1)
    draw.text((x0, y0 - 30), title, fill=(20, 20, 20), font=_font(16, bold=True))
    draw.text(((x0 + x1) / 2 - 70, y1 + 34), xlabel, fill=(45, 45, 45), font=_font(13))
    if ylabel:
        draw.text((x0 - 55, y0 - 10), ylabel, fill=(45, 45, 45), font=_font(13))
    if xticks:
        for x, label in xticks:
            draw.line((x, y1, x, y1 + 5), fill=(55, 55, 55), width=1)
            draw.text((x - 16, y1 + 9), label, fill=(60, 60, 60), font=_font(11))


def plot_enrichment_pil(summary: pd.DataFrame, fig_dir: Path) -> None:
    W, H = 1800, 620
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 22), "Enriquecimiento de ECG de baja certeza humana por presupuesto de revision", fill=(20, 20, 20), font=_font(26, True))
    metrics = [
        ("LowCertCapture@q", "Captura baja certeza"),
        ("LowCertPrecision@q", "Precision derivados"),
        ("LowCertLift@q", "Lift vs azar"),
    ]
    colors = {"B3": "#5B6470", "B2prob": "#8770A8", "P0": "#2B8C7E", "P1": "#D17A22"}
    labels = {"B3": "B3", "B2prob": "B2-prob", "P0": "P0", "P1": "P1"}
    rects = [(90, 110, 560, 460), (680, 110, 1150, 460), (1270, 110, 1740, 460)]
    for rect, (metric, title) in zip(rects, metrics):
        ymax = max(1.0, float(summary[f"{metric}_ci95_high"].max()) * 1.08)
        ymin = 0.0
        _draw_axes(
            draw,
            rect,
            title,
            "q",
            "valor",
            xticks=[(_panel_map(q, ymin, rect, 0.05, 0.30, ymin, ymax)[0], f"{q:.2f}") for q in QS],
            yticks=[0.25, 0.5, 0.75],
        )
        x0, y0, x1, y1 = rect
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = ymin + frac * (ymax - ymin)
            yy = y1 - frac * (y1 - y0)
            draw.text((x0 - 45, yy - 7), f"{val:.1f}", fill=(70, 70, 70), font=_font(10))
        if metric == "LowCertCapture@q":
            pts = [_panel_map(q, q, rect, 0.05, 0.30, ymin, ymax) for q in QS]
            draw.line(pts, fill=(140, 140, 140), width=2)
        if metric == "LowCertLift@q":
            y = 1.0
            draw.line((_panel_map(0.05, y, rect, 0.05, 0.30, ymin, ymax), _panel_map(0.30, y, rect, 0.05, 0.30, ymin, ymax)), fill=(140, 140, 140), width=2)
        for key in ["B3", "B2prob", "P0", "P1"]:
            s = summary[summary["score"] == key].sort_values("q")
            pts = [
                _panel_map(float(row["q"]), float(row[f"{metric}_mean"]), rect, 0.05, 0.30, ymin, ymax)
                for _, row in s.iterrows()
            ]
            _draw_line(draw, pts, _rgb(colors[key]), width=4)
    lx, ly = 1320, 500
    for i, key in enumerate(["B3", "B2prob", "P0", "P1"]):
        y = ly + i * 26
        draw.line((lx, y + 8, lx + 38, y + 8), fill=_rgb(colors[key]), width=5)
        draw.text((lx + 48, y), labels[key], fill=(30, 30, 30), font=_font(13))
    _save_pil(img, fig_dir / "fig_posthoc_low_cert_enrichment")


def plot_pareto_pil(summary: pd.DataFrame, fig_dir: Path, q: float = 0.20) -> None:
    W, H = 950, 650
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    rect = (110, 90, 850, 520)
    s = summary[np.isclose(summary["q"], q)].copy()
    xmin, xmax = 0.15, max(0.50, float(s["ErrorCapture@q_ci95_high"].max()) * 1.08)
    ymin, ymax = 0.15, max(0.45, float(s["LowCertCapture@q_ci95_high"].max()) * 1.08)
    colors = {"B3": "#5B6470", "B2prob": "#8770A8", "P0": "#2B8C7E", "P1": "#D17A22"}
    _draw_axes(draw, rect, f"Objetivos de derivacion con q={q:.2f}", "Captura de errores", "Baja certeza")
    for frac in np.linspace(0, 1, 5):
        yy = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], yy, rect[2], yy), fill=(230, 230, 230), width=1)
    px0, py0 = _panel_map(q, ymin, rect, xmin, xmax, ymin, ymax)
    px1, py1 = _panel_map(q, ymax, rect, xmin, xmax, ymin, ymax)
    draw.line((px0, py0, px1, py1), fill=(160, 160, 160), width=2)
    px0, py0 = _panel_map(xmin, q, rect, xmin, xmax, ymin, ymax)
    px1, py1 = _panel_map(xmax, q, rect, xmin, xmax, ymin, ymax)
    draw.line((px0, py0, px1, py1), fill=(160, 160, 160), width=2)
    for _, row in s.iterrows():
        key = row["score"]
        x = float(row["ErrorCapture@q_mean"])
        y = float(row["LowCertCapture@q_mean"])
        px, py = _panel_map(x, y, rect, xmin, xmax, ymin, ymax)
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=_rgb(colors.get(key, "#333333")), outline=(255, 255, 255), width=2)
        draw.text((px + 12, py - 8), key, fill=(25, 25, 25), font=_font(13, True))
    draw.text((45, 20), "Trade-off entre captura de errores y baja certeza humana", fill=(20, 20, 20), font=_font(22, True))
    _save_pil(img, fig_dir / f"fig_posthoc_pareto_error_vs_lowcert_q{int(q*100):02d}")


def plot_calibration_diffs_pil(diff: pd.DataFrame, fig_dir: Path) -> None:
    keep = diff[diff["metric"].isin(["Brier_macro", "ECE_macro_10bins", "AUROC_macro", "AUPRC_macro"])].copy()
    labels = {
        "Brier_macro": "Brier macro",
        "ECE_macro_10bins": "ECE macro",
        "AUROC_macro": "AUROC macro",
        "AUPRC_macro": "AUPRC macro",
    }
    W, H = 950, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((45, 25), "Cambio asociado a etiquetas suaves likelihood/100", fill=(20, 20, 20), font=_font(22, True))
    x0, y0, x1, y1 = 250, 100, 850, 420
    vals = keep["diff_mean"].fillna(0).to_numpy(dtype=float)
    lim = max(0.02, float(np.nanmax(np.abs([keep["diff_ci95_low"], keep["diff_ci95_high"]]))) * 1.25)
    zero_x = x0 + (0 + lim) / (2 * lim) * (x1 - x0)
    draw.line((zero_x, y0 - 10, zero_x, y1 + 10), fill=(40, 40, 40), width=2)
    row_h = (y1 - y0) / len(keep)
    for i, (_, row) in enumerate(keep.iterrows()):
        cy = y0 + row_h * (i + 0.5)
        val = float(row["diff_mean"]) if pd.notna(row["diff_mean"]) else 0.0
        end_x = x0 + (val + lim) / (2 * lim) * (x1 - x0)
        color = _rgb("#2B8C7E" if row["metric"] in ["Brier_macro", "ECE_macro_10bins"] and val < 0 else "#8770A8")
        draw.rectangle((min(zero_x, end_x), cy - 16, max(zero_x, end_x), cy + 16), fill=color)
        lo = float(row["diff_ci95_low"]) if pd.notna(row["diff_ci95_low"]) else val
        hi = float(row["diff_ci95_high"]) if pd.notna(row["diff_ci95_high"]) else val
        lx = x0 + (lo + lim) / (2 * lim) * (x1 - x0)
        hx = x0 + (hi + lim) / (2 * lim) * (x1 - x0)
        draw.line((lx, cy, hx, cy), fill=(35, 35, 35), width=2)
        draw.line((lx, cy - 7, lx, cy + 7), fill=(35, 35, 35), width=2)
        draw.line((hx, cy - 7, hx, cy + 7), fill=(35, 35, 35), width=2)
        draw.text((45, cy - 10), labels.get(row["metric"], row["metric"]), fill=(35, 35, 35), font=_font(14))
        draw.text((end_x + (8 if val >= 0 else -75), cy - 10), f"{val:+.3f}", fill=(35, 35, 35), font=_font(13))
    draw.text((x0, y1 + 35), "Diferencia B2 - B1 (en Brier/ECE, negativo favorece B2)", fill=(45, 45, 45), font=_font(13))
    _save_pil(img, fig_dir / "fig_posthoc_b2_vs_b1_calibration")


def plot_quadrants_pil(quad_summary: pd.DataFrame, fig_dir: Path) -> None:
    W, H = 900, 680
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 25), "Cuadrantes: ambiguedad humana aprendida vs incertidumbre del modelo", fill=(20, 20, 20), font=_font(22, True))
    vals = quad_summary.set_index("quadrant")
    order = [
        ("alta_humana_alta_modelo", "Humana alta\nModelo alta", 120, 120),
        ("alta_humana_baja_modelo", "Humana alta\nModelo baja", 470, 120),
        ("baja_humana_alta_modelo", "Humana baja\nModelo alta", 120, 370),
        ("baja_humana_baja_modelo", "Humana baja\nModelo baja", 470, 370),
    ]
    max_low = float(vals["low_cert_rate_mean"].max()) if not vals.empty else 1.0
    for key, label, x, y in order:
        row = vals.loc[key]
        intensity = row["low_cert_rate_mean"] / max_low if max_low else 0
        base = np.array(_rgb("#F3E2B7"))
        hot = np.array(_rgb("#D17A22"))
        color = tuple((base * (1 - intensity) + hot * intensity).astype(int))
        draw.rectangle((x, y, x + 300, y + 190), fill=color, outline=(55, 55, 55), width=2)
        lines = [
            label,
            f"N={row['N_mean']:.0f} ({100*row['pct_test_mean']:.1f}%)",
            f"Baja certeza={100*row['low_cert_rate_mean']:.1f}%",
            f"Error B1={100*row['B1_error_rate_mean']:.1f}%",
        ]
        for j, line in enumerate(lines):
            draw.text((x + 25, y + 25 + j * 34), line, fill=(20, 20, 20), font=_font(17 if j == 0 else 15, j == 0))
    draw.text((205, 85), "Modelo alto", fill=(50, 50, 50), font=_font(14, True))
    draw.text((555, 85), "Modelo bajo", fill=(50, 50, 50), font=_font(14, True))
    draw.text((35, 185), "Humana alta", fill=(50, 50, 50), font=_font(14, True))
    draw.text((35, 435), "Humana baja", fill=(50, 50, 50), font=_font(14, True))
    _save_pil(img, fig_dir / "fig_posthoc_quadrants_human_model")


def plot_class_capture_pil(class_summary: pd.DataFrame, fig_dir: Path, q: float = 0.20) -> None:
    s = class_summary[(class_summary["score"].isin(["B3", "P1"])) & np.isclose(class_summary["q"], q)].copy()
    W, H = 1000, 600
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((45, 25), f"Captura de errores por clase con q={q:.2f}", fill=(20, 20, 20), font=_font(22, True))
    x0, y0, x1, y1 = 90, 100, 910, 470
    draw.rectangle((x0, y0, x1, y1), outline=(55, 55, 55), width=1)
    ymax = max(0.50, float(s["ClassErrorCapture@q_ci95_high"].max()) * 1.1)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        yy = y1 - frac * (y1 - y0)
        draw.line((x0, yy, x1, yy), fill=(230, 230, 230), width=1)
        draw.text((x0 - 48, yy - 7), f"{frac*ymax:.2f}", fill=(70, 70, 70), font=_font(10))
    bw = 42
    group_w = (x1 - x0) / len(CLASSES)
    colors = {"B3": "#5B6470", "P1": "#D17A22"}
    for i, cls in enumerate(CLASSES):
        cx = x0 + group_w * (i + 0.5)
        draw.text((cx - 18, y1 + 16), cls, fill=(35, 35, 35), font=_font(13, True))
        for key, dx in [("B3", -26), ("P1", 26)]:
            row = s[(s["score"] == key) & (s["class"] == cls)].iloc[0]
            val = float(row["ClassErrorCapture@q_mean"])
            bh = val / ymax * (y1 - y0)
            draw.rectangle((cx + dx - bw / 2, y1 - bh, cx + dx + bw / 2, y1), fill=_rgb(colors[key]))
    draw.line((x0, y1 - q / ymax * (y1 - y0), x1, y1 - q / ymax * (y1 - y0)), fill=(140, 140, 140), width=2)
    draw.text((x1 - 90, y1 - q / ymax * (y1 - y0) - 18), "azar", fill=(90, 90, 90), font=_font(11))
    draw.rectangle((710, 505, 730, 525), fill=_rgb(colors["B3"]))
    draw.text((738, 503), "B3", fill=(35, 35, 35), font=_font(13))
    draw.rectangle((790, 505, 810, 525), fill=_rgb(colors["P1"]))
    draw.text((818, 503), "P1", fill=(35, 35, 35), font=_font(13))
    _save_pil(img, fig_dir / f"fig_posthoc_class_error_capture_q{int(q*100):02d}")


def quadrant_analysis(dfs_val: dict[str, pd.DataFrame], dfs_test: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for seed, val in dfs_val.items():
        test = dfs_test[seed]
        h_thr = float(np.quantile(val["P1_MTL_BALANCED_PROPOSED_review_score"], 0.80))
        m_thr = float(np.quantile(val["B3_threshold_uncertainty_score"], 0.80))
        h_high = test["P1_MTL_BALANCED_PROPOSED_review_score"].to_numpy(dtype=float) >= h_thr
        m_high = test["B3_threshold_uncertainty_score"].to_numpy(dtype=float) >= m_thr
        categories = {
            "alta_humana_alta_modelo": h_high & m_high,
            "alta_humana_baja_modelo": h_high & ~m_high,
            "baja_humana_alta_modelo": ~h_high & m_high,
            "baja_humana_baja_modelo": ~h_high & ~m_high,
        }
        for quad, mask in categories.items():
            sub = test.loc[mask].copy()
            if len(sub) == 0:
                continue
            y_mat = sub[[f"y_{c}" for c in CLASSES]].to_numpy(dtype=int)
            row = {
                "seed": seed,
                "quadrant": quad,
                "human_threshold_p80_val": h_thr,
                "model_threshold_p80_val": m_thr,
                "N": len(sub),
                "pct_test": len(sub) / len(test),
                "low_cert_rate": float(sub["review_binary"].mean()),
                "mean_ambiguity": float(sub["ambiguity_score"].mean()),
                "B1_error_rate": float(sub["B1_HARD_DIAG_sample_error"].mean()),
                "P1_error_rate": float(sub["P1_MTL_BALANCED_PROPOSED_sample_error"].mean()),
                "age_mean": float(sub["age"].mean()),
                "sex_1_rate": float(sub["sex"].mean()),
                "multilabel_cardinality_mean": float(y_mat.sum(axis=1).mean()),
            }
            for cls in CLASSES:
                row[f"prev_{cls}"] = float(sub[f"y_{cls}"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def plot_enrichment(summary: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL:
        return plot_enrichment_pil(summary, fig_dir)
    metric_map = [
        ("LowCertCapture@q", "Captura de baja certeza"),
        ("LowCertPrecision@q", "Precision entre derivados"),
        ("LowCertLift@q", "Lift vs azar"),
    ]
    model_order = ["B3", "B2prob", "P0", "P1"]
    colors = {"B3": "#5B6470", "B2prob": "#8770A8", "P0": "#2B8C7E", "P1": "#D17A22"}
    labels = {
        "B3": "B3 umbral",
        "B2prob": "B2 prob.",
        "P0": "P0 multitarea",
        "P1": "P1 balanceado",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), sharex=True)
    for ax, (metric, title) in zip(axes, metric_map):
        for key in model_order:
            s = summary[summary["score"] == key].sort_values("q")
            if s.empty:
                continue
            ax.plot(s["q"], s[f"{metric}_mean"], marker="o", linewidth=2, label=labels[key], color=colors[key])
            ax.fill_between(
                s["q"].to_numpy(dtype=float),
                s[f"{metric}_ci95_low"].to_numpy(dtype=float),
                s[f"{metric}_ci95_high"].to_numpy(dtype=float),
                color=colors[key],
                alpha=0.14,
                linewidth=0,
            )
        if metric == "LowCertCapture@q":
            ax.plot(QS, QS, linestyle="--", color="#8A8A8A", linewidth=1, label="azar")
        if metric == "LowCertLift@q":
            ax.axhline(1.0, linestyle="--", color="#8A8A8A", linewidth=1, label="azar")
        ax.set_title(title)
        ax.set_xlabel("Presupuesto de revision q")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Valor")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Enriquecimiento de ECG de baja certeza humana por presupuesto de revision", y=1.03)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"fig_posthoc_low_cert_enrichment.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(summary: pd.DataFrame, fig_dir: Path, q: float = 0.20) -> None:
    if not HAS_MPL:
        return plot_pareto_pil(summary, fig_dir, q=q)
    s = summary[np.isclose(summary["q"], q)].copy()
    colors = {"B3": "#5B6470", "B2prob": "#8770A8", "P0": "#2B8C7E", "P1": "#D17A22"}
    labels = {
        "B3": "B3",
        "B2prob": "B2-prob",
        "P0": "P0",
        "P1": "P1",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, row in s.iterrows():
        key = row["score"]
        x = row["ErrorCapture@q_mean"]
        y = row["LowCertCapture@q_mean"]
        ax.errorbar(
            x,
            y,
            xerr=[[x - row["ErrorCapture@q_ci95_low"]], [row["ErrorCapture@q_ci95_high"] - x]],
            yerr=[[y - row["LowCertCapture@q_ci95_low"]], [row["LowCertCapture@q_ci95_high"] - y]],
            fmt="o",
            color=colors.get(key, "black"),
            capsize=3,
            markersize=7,
        )
        ax.annotate(labels.get(key, key), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(q, color="#8A8A8A", linestyle="--", linewidth=1)
    ax.axhline(q, color="#8A8A8A", linestyle="--", linewidth=1)
    ax.text(q + 0.004, q + 0.004, "azar", fontsize=8, color="#666666")
    ax.set_xlabel("Captura de errores del modelo")
    ax.set_ylabel("Captura de baja certeza humana")
    ax.set_title(f"Objetivos de derivacion con q={q:.2f}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"fig_posthoc_pareto_error_vs_lowcert_q{int(q*100):02d}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_diffs(diff: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL:
        return plot_calibration_diffs_pil(diff, fig_dir)
    keep = diff[diff["metric"].isin(["Brier_macro", "ECE_macro_10bins", "AUROC_macro", "AUPRC_macro"])].copy()
    labels = {
        "Brier_macro": "Brier macro",
        "ECE_macro_10bins": "ECE macro",
        "AUROC_macro": "AUROC macro",
        "AUPRC_macro": "AUPRC macro",
    }
    keep["plot_label"] = keep["metric"].map(labels)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(keep))
    x = keep["diff_mean"].to_numpy(dtype=float)
    lo = keep["diff_ci95_low"].to_numpy(dtype=float)
    hi = keep["diff_ci95_high"].to_numpy(dtype=float)
    colors = ["#2B8C7E" if v < 0 and m in ["Brier_macro", "ECE_macro_10bins"] else "#8770A8" for v, m in zip(x, keep["metric"])]
    ax.barh(y, x, color=colors, alpha=0.78)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="none", ecolor="#333333", capsize=3)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(keep["plot_label"])
    ax.set_xlabel("Diferencia B2 - B1")
    ax.set_title("Cambio asociado a etiquetas suaves likelihood/100")
    ax.grid(True, axis="x", alpha=0.25)
    for yi, xi in zip(y, x):
        ax.text(xi, yi, f" {xi:+.3f}", va="center", fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"fig_posthoc_b2_vs_b1_calibration.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_quadrants(quad_summary: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL:
        return plot_quadrants_pil(quad_summary, fig_dir)
    order = [
        ("alta_humana_alta_modelo", "Humana alta\nModelo alta", 0, 0),
        ("alta_humana_baja_modelo", "Humana alta\nModelo baja", 0, 1),
        ("baja_humana_alta_modelo", "Humana baja\nModelo alta", 1, 0),
        ("baja_humana_baja_modelo", "Humana baja\nModelo baja", 1, 1),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    vals = quad_summary.set_index("quadrant")
    low_rates = [vals.loc[k, "low_cert_rate_mean"] for k, _, _, _ in order if k in vals.index]
    vmax = max(low_rates) if low_rates else 1
    for key, label, r, c in order:
        if key not in vals.index:
            continue
        row = vals.loc[key]
        color_value = row["low_cert_rate_mean"] / vmax if vmax else 0
        rect = plt.Rectangle((c, 1 - r), 1, 1, facecolor=plt.cm.YlOrBr(0.25 + 0.65 * color_value), edgecolor="#333333", linewidth=1)
        ax.add_patch(rect)
        txt = (
            f"{label}\n"
            f"N={row['N_mean']:.0f} ({100*row['pct_test_mean']:.1f}%)\n"
            f"Baja certeza={100*row['low_cert_rate_mean']:.1f}%\n"
            f"Error B1={100*row['B1_error_rate_mean']:.1f}%"
        )
        ax.text(c + 0.5, 1 - r + 0.5, txt, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Modelo alto", "Modelo bajo"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Humana baja", "Humana alta"])
    ax.set_title("Cuadrantes de ambiguedad humana aprendida e incertidumbre del modelo")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"fig_posthoc_quadrants_human_model.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_capture(class_summary: pd.DataFrame, fig_dir: Path, q: float = 0.20) -> None:
    if not HAS_MPL:
        return plot_class_capture_pil(class_summary, fig_dir, q=q)
    s = class_summary[(class_summary["score"].isin(["B3", "P1"])) & np.isclose(class_summary["q"], q)].copy()
    classes = CLASSES
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for offset, key, color, label in [(-width / 2, "B3", "#5B6470", "B3 umbral"), (width / 2, "P1", "#D17A22", "P1 balanceado")]:
        vals = []
        los = []
        his = []
        for cls in classes:
            row = s[(s["score"] == key) & (s["class"] == cls)].iloc[0]
            vals.append(row["ClassErrorCapture@q_mean"])
            los.append(row["ClassErrorCapture@q_ci95_low"])
            his.append(row["ClassErrorCapture@q_ci95_high"])
        vals = np.asarray(vals)
        los = np.asarray(los)
        his = np.asarray(his)
        ax.bar(x + offset, vals, width, color=color, alpha=0.78, label=label)
        ax.errorbar(x + offset, vals, yerr=[vals - los, his - vals], fmt="none", ecolor="#333333", capsize=3)
    ax.axhline(q, color="#8A8A8A", linestyle="--", linewidth=1, label="azar aprox.")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Captura de errores por clase")
    ax.set_xlabel("Superclase diagnostica")
    ax.set_title(f"Captura de errores por clase con q={q:.2f}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"fig_posthoc_class_error_capture_q{int(q*100):02d}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def to_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, float_format: str = "%.3f") -> None:
    def esc(value) -> str:
        if isinstance(value, float) or isinstance(value, np.floating):
            if not np.isfinite(value):
                return "NA"
            text = float_format % float(value)
        else:
            text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    cols = list(df.columns)
    align = "l" + "c" * max(0, len(cols) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(esc(c) for c in cols) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(row[c]) for c in cols) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float) or isinstance(val, np.floating):
                vals.append(fmt(float(val)))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip)
    out = Path(args.out)
    dirs = ensure_dirs(out)
    rng = np.random.default_rng(args.seed)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zip_names(zf)
        runs = discover_runs(names)
        if not runs:
            raise RuntimeError("No se encontraron runs_hcg_selective full en el ZIP.")

        summaries = {}
        missing = []
        required = [
            "RUN_SUMMARY.json",
            "results/tables/master_predictions_test.csv",
            "results/tables/master_predictions_val.csv",
            "results/tables/diagnostic_metrics_global.csv",
            "results/tables/review_metrics.csv",
            "results/tables/selective_referral_metrics.csv",
            "results/tables/selective_referral_by_class.csv",
            "results/tables/subgroup_metrics.csv",
            "results/tables/bootstrap_summary.csv",
            "results/explainability/lead_ablation.csv",
        ]
        for run in runs:
            for rel in required:
                path = run_path(run, rel)
                if path not in names:
                    missing.append(path)
            summaries[run] = read_json(zf, run_path(run, "RUN_SUMMARY.json"))

        test_dfs: dict[str, pd.DataFrame] = {}
        val_dfs: dict[str, pd.DataFrame] = {}
        diag_global_rows = []
        for run in runs:
            test = add_b2_uncertainty(read_csv(zf, run_path(run, "results/tables/master_predictions_test.csv")))
            val = add_b2_uncertainty(read_csv(zf, run_path(run, "results/tables/master_predictions_val.csv")))
            test["seed"] = str(summaries[run].get("seed", run))
            val["seed"] = str(summaries[run].get("seed", run))
            test["run_id"] = run
            val["run_id"] = run
            test_dfs[run] = test
            val_dfs[run] = val
            diag_global = read_csv(zf, run_path(run, "results/tables/diagnostic_metrics_global.csv"))
            diag_global["seed"] = str(summaries[run].get("seed", run))
            diag_global["run_id"] = run
            diag_global_rows.append(diag_global)

        aggregate_files = [
            "MULTISEED_SUMMARY.md",
            "multiseed_diagnostic_global_test.csv",
            "multiseed_review_metrics.csv",
            "multiseed_selective_referral_q20.csv",
            "multiseed_paired_diffs.csv",
            "multiseed_subgroup_metrics.csv",
        ]
        aggregate_found = [agg_path(f) for f in aggregate_files if agg_path(f) in names]

    diag_global_by_run = pd.concat(diag_global_rows, ignore_index=True) if diag_global_rows else pd.DataFrame()

    # A. Enrichment curves.
    seed_rows = []
    for run, df in test_dfs.items():
        seed = df["seed"].iloc[0]
        for spec in SCORES:
            for q in QS:
                row = referral_metrics(df, spec, q)
                row.update({"run_id": run, "seed": seed, "score": spec.key, "score_label": spec.label, "diag_model": spec.diag_model})
                seed_rows.append(row)
    enrichment_seed = pd.DataFrame(seed_rows)
    enrichment_summary = aggregate_by_seed(
        enrichment_seed,
        ["score", "score_label", "diag_model", "q"],
        ["LowCertCapture@q", "LowCertPrecision@q", "LowCertLift@q", "NNR_low_cert@q", "ErrorCapture@q", "ErrorPrecision@q"],
    )
    enrichment_seed.to_csv(dirs["tables"] / "posthoc_enrichment_by_seed.csv", index=False)
    enrichment_summary.to_csv(dirs["tables"] / "posthoc_enrichment_summary.csv", index=False)

    boot_frames = []
    boot_metrics = ["LowCertCapture@q", "LowCertPrecision@q", "LowCertLift@q", "ErrorCapture@q"]
    for q in [0.10, 0.20]:
        boot_frames.append(
            bootstrap_seed_mean_diffs_multi(test_dfs, SCORES[2], SCORES[0], q, boot_metrics, args.n_bootstrap, rng)
        )
        boot_frames.append(
            bootstrap_seed_mean_diffs_multi(test_dfs, SCORES[1], SCORES[0], q, boot_metrics, args.n_bootstrap, rng)
        )
    boot_df = pd.concat(boot_frames, ignore_index=True)
    boot_df.to_csv(dirs["tables"] / "posthoc_bootstrap_enrichment_diffs.csv", index=False)

    # B. Calibration B1 vs B2.
    cal_seed, cal_class_seed = calibration_table(test_dfs)
    if not diag_global_by_run.empty:
        diag_test = diag_global_by_run[
            (diag_global_by_run["split"] == "test")
            & (diag_global_by_run["model"].isin(["B1_HARD_DIAG", "B2_SOFT_DIRECT"]))
        ][["seed", "model", "AUROC_macro_thrval", "AUPRC_macro_thrval"]].rename(
            columns={"AUROC_macro_thrval": "AUROC_macro_saved", "AUPRC_macro_thrval": "AUPRC_macro_saved"}
        )
        cal_seed = cal_seed.merge(diag_test, on=["seed", "model"], how="left")
        cal_seed["AUROC_macro"] = cal_seed["AUROC_macro_saved"].combine_first(cal_seed["AUROC_macro"])
        cal_seed["AUPRC_macro"] = cal_seed["AUPRC_macro_saved"].combine_first(cal_seed["AUPRC_macro"])
        cal_seed = cal_seed.drop(columns=["AUROC_macro_saved", "AUPRC_macro_saved"])
    cal_summary = aggregate_by_seed(
        cal_seed,
        ["model"],
        ["Brier_macro", "ECE_macro_10bins", "ECE_macro_15bins", "LogLoss_macro", "AUROC_macro", "AUPRC_macro"],
    )
    cal_diff = paired_diff_table(
        cal_seed,
        "B2_SOFT_DIRECT",
        "B1_HARD_DIAG",
        ["Brier_macro", "ECE_macro_10bins", "ECE_macro_15bins", "LogLoss_macro", "AUROC_macro", "AUPRC_macro"],
    )
    cal_class_summary = aggregate_by_seed(cal_class_seed, ["model", "class"], ["Brier", "ECE_10bins"])
    cal_seed.to_csv(dirs["tables"] / "posthoc_calibration_b1_b2_by_seed.csv", index=False)
    cal_summary.to_csv(dirs["tables"] / "posthoc_calibration_b1_b2_summary.csv", index=False)
    cal_diff.to_csv(dirs["tables"] / "posthoc_calibration_b2_minus_b1_diffs.csv", index=False)
    cal_class_summary.to_csv(dirs["tables"] / "posthoc_calibration_b1_b2_by_class_summary.csv", index=False)

    # E. Quadrants.
    quad_seed = quadrant_analysis(val_dfs, test_dfs)
    quad_summary = aggregate_by_seed(
        quad_seed,
        ["quadrant"],
        ["N", "pct_test", "low_cert_rate", "mean_ambiguity", "B1_error_rate", "P1_error_rate", "age_mean", "sex_1_rate", "multilabel_cardinality_mean"] + [f"prev_{c}" for c in CLASSES],
    )
    quad_seed.to_csv(dirs["tables"] / "posthoc_quadrants_by_seed.csv", index=False)
    quad_summary.to_csv(dirs["tables"] / "posthoc_quadrants_summary.csv", index=False)

    # F. Class capture.
    class_rows = []
    for run, df in test_dfs.items():
        seed = df["seed"].iloc[0]
        for spec in [SCORES[0], SCORES[1], SCORES[2], SCORES[3]]:
            for q in [0.10, 0.20]:
                for row in class_capture(df, spec, q):
                    row.update({"run_id": run, "seed": seed})
                    class_rows.append(row)
    class_seed = pd.DataFrame(class_rows)
    class_summary = aggregate_by_seed(
        class_seed,
        ["score", "score_label", "diag_model", "q", "class"],
        ["ClassErrorCapture@q", "FalseNegativeCapture@q", "FalsePositiveCapture@q", "n_class_error", "n_false_negative", "n_false_positive"],
    )
    class_seed.to_csv(dirs["tables"] / "posthoc_class_capture_by_seed.csv", index=False)
    class_summary.to_csv(dirs["tables"] / "posthoc_class_capture_summary.csv", index=False)

    # Compact tables for manuscript.
    q20 = enrichment_summary[np.isclose(enrichment_summary["q"], 0.20)].copy()
    compact = q20[[
        "score_label",
        "LowCertCapture@q_mean",
        "LowCertCapture@q_ci95_low",
        "LowCertCapture@q_ci95_high",
        "LowCertPrecision@q_mean",
        "LowCertLift@q_mean",
        "NNR_low_cert@q_mean",
        "ErrorCapture@q_mean",
    ]].copy()
    compact.columns = [
        "Score",
        "LowCertCapture@20",
        "IC95_inf",
        "IC95_sup",
        "Precision_baja_certeza",
        "Lift",
        "NNR",
        "ErrorCapture@20",
    ]
    compact.to_csv(dirs["tables"] / "table_new_low_cert_enrichment_q20.csv", index=False)
    to_latex_table(
        compact,
        dirs["latex"] / "table_new_low_cert_enrichment_q20.tex",
        "Enriquecimiento de casos de baja certeza humana al derivar el 20\\% de ECG con mayor score.",
        "tab:posthoc_low_cert_enrichment_q20",
    )

    cal_compact = cal_diff[cal_diff["metric"].isin(["Brier_macro", "ECE_macro_10bins", "AUROC_macro", "AUPRC_macro"])].copy()
    cal_compact.to_csv(dirs["tables"] / "table_new_b2_b1_calibration_diffs.csv", index=False)
    to_latex_table(
        cal_compact,
        dirs["latex"] / "table_new_b2_b1_calibration_diffs.tex",
        "Diferencias pareadas entre entrenamiento con etiquetas suaves y duras. En Brier y ECE, valores negativos favorecen B2.",
        "tab:posthoc_b2_b1_calibration",
    )

    # Figures.
    plot_enrichment(enrichment_summary, dirs["figures"])
    plot_pareto(enrichment_summary, dirs["figures"], q=0.20)
    plot_calibration_diffs(cal_diff, dirs["figures"])
    plot_quadrants(quad_summary, dirs["figures"])
    plot_class_capture(class_summary, dirs["figures"], q=0.20)

    # Audit and screening reports.
    first_test = next(iter(test_dfs.values()))
    available_cols = list(first_test.columns)
    audit = f"""
    # AUDIT - exploracion post-hoc HCG-SelectiveECG

    Fecha: {datetime.now().isoformat(timespec='seconds')}

    ZIP revisado: `{zip_path}`

    Runs full encontrados ({len(runs)}):
    {chr(10).join(f"- {run}: mode={summaries[run].get('run_mode')}, device={summaries[run].get('device')}, models={summaries[run].get('models')}" for run in runs)}

    Archivos obligatorios faltantes:
    {chr(10).join(f"- {m}" for m in missing) if missing else "- Ninguno en los artefactos requeridos."}

    Archivos agregados multisemilla encontrados:
    {chr(10).join(f"- {p}" for p in aggregate_found) if aggregate_found else "- No encontrados."}

    Variables disponibles en `master_predictions_test.csv`:
    - Identificadores: ecg_id, patient_id, strat_fold.
    - Metadatos: age, sex, filename_lr.
    - Etiquetas: y_*, soft_*, likelihood_* para NORM, MI, STTC, CD, HYP.
    - Senal humana: ambiguity_score, ambiguity_bin, review_binary.
    - Probabilidades y predicciones: B1, B2, P0, P1 por clase.
    - Scores de revision: B3_threshold_uncertainty_score, P0_review_score, P1_review_score.
    - Errores por muestra: *_sample_error.

    Forma test por semilla: {first_test.shape}

    Analisis factibles sin reentrenamiento:
    - Enriquecimiento de baja certeza por presupuesto q.
    - Calibracion B1 vs B2 desde probabilidades guardadas.
    - Cuadrantes P1-human-score vs B3-model-uncertainty con cortes de validacion.
    - Captura de errores por clase y tipo FP/FN.
    - Bootstrap por paciente usando patient_id.

    Analisis no priorizados ahora:
    - SHAP/UMAP/casos con ECG crudo: requieren mas computo y agregan riesgo metodologico para una entrega cercana.
    - Inferencia desde checkpoints: no es necesaria porque las predicciones ya existen.

    Riesgos metodologicos:
    - Los analisis post-hoc son exploratorios salvo las comparaciones ya predefinidas.
    - Las ventajas por clase deben reportarse con cautela si los intervalos cruzan cero.
    - No debe afirmarse superioridad clinica; los resultados describen priorizacion de revision en PTB-XL.
    """
    write_text(dirs["reports"] / "AUDIT.md", audit)

    screening_rows = [
        {
            "analisis": "A. Enriquecimiento baja certeza",
            "disponibilidad": "completa",
            "costo": "bajo",
            "valor_narrativo": "alto",
            "riesgo": "bajo-medio",
            "decision": "continuar",
            "motivo": "P0/P1 ya muestran ventaja en LowCertCapture; agrega precision/lift/NNR visualizable.",
        },
        {
            "analisis": "B. Calibracion B1 vs B2",
            "disponibilidad": "completa",
            "costo": "bajo",
            "valor_narrativo": "alto",
            "riesgo": "bajo",
            "decision": "continuar como complemento",
            "motivo": "B2 reduce Brier/ECE con perdida discriminativa pequena; no cambia narrativa HCG.",
        },
        {
            "analisis": "C/D. Ensemble e hibrido",
            "disponibilidad": "parcial",
            "costo": "medio",
            "valor_narrativo": "medio",
            "riesgo": "medio-alto",
            "decision": "posponer",
            "motivo": "Requiere seleccionar definicion/alpha en validacion y podria expandir demasiado el manuscrito.",
        },
        {
            "analisis": "E. Cuadrantes humano-modelo",
            "disponibilidad": "completa",
            "costo": "bajo",
            "valor_narrativo": "alto",
            "riesgo": "medio",
            "decision": "continuar como figura secundaria",
            "motivo": "Visualiza que ambiguedad humana e incertidumbre del modelo son constructos distintos.",
        },
        {
            "analisis": "F. NORM vs patologias",
            "disponibilidad": "completa",
            "costo": "bajo",
            "valor_narrativo": "medio-alto",
            "riesgo": "medio",
            "decision": "continuar como tabla/figura exploratoria",
            "motivo": "Muestra escenario favorable por MI/HYP sin afirmar mejora global de errores.",
        },
        {
            "analisis": "G-I. Complejidad, artefactos, edad/sexo",
            "disponibilidad": "parcial",
            "costo": "medio",
            "valor_narrativo": "medio",
            "riesgo": "medio",
            "decision": "posponer salvo uso descriptivo",
            "motivo": "Algunas variables de artefacto/complexidad completa no estan en master_predictions.",
        },
        {
            "analisis": "J-K. Embeddings/explicabilidad",
            "disponibilidad": "lead ablation disponible; embeddings no confirmados",
            "costo": "medio-alto",
            "valor_narrativo": "medio",
            "riesgo": "alto",
            "decision": "no incorporar ahora",
            "motivo": "Puede distraer y no es necesario para demostrar ventaja post-hoc.",
        },
    ]
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(dirs["tables"] / "screening_candidates.csv", index=False)
    screen_md = "# SCREENING de analisis candidatos\n\n" + df_to_markdown(screening)
    write_text(dirs["reports"] / "SCREENING.md", screen_md)

    # Leakage report.
    leak = f"""
    # LEAKAGE_CHECK

    - No se reentrenaron modelos.
    - Los scores usados provienen de predicciones ya guardadas.
    - Los percentiles para cuadrantes se calcularon en validacion y se aplicaron en test.
    - No se seleccionaron hiperparametros usando test.
    - El bootstrap se realizo por `patient_id`.
    - Las curvas de enriquecimiento usan q predefinidos: {QS}.
    - Las comparaciones de B1 vs B2 se calculan con probabilidades de test ya generadas por las corridas full.
    - Los resultados por clase se declaran exploratorios.
    """
    write_text(dirs["reports"] / "LEAKAGE_CHECK.md", leak)

    # Narrative summaries.
    def row_at(summary: pd.DataFrame, score: str, q: float) -> pd.Series:
        return summary[(summary["score"] == score) & np.isclose(summary["q"], q)].iloc[0]

    p1_q20 = row_at(enrichment_summary, "P1", 0.20)
    b3_q20 = row_at(enrichment_summary, "B3", 0.20)
    p0_q20 = row_at(enrichment_summary, "P0", 0.20)
    b2_brier = cal_diff[cal_diff["metric"] == "Brier_macro"].iloc[0]
    b2_ece = cal_diff[cal_diff["metric"] == "ECE_macro_10bins"].iloc[0]
    b2_auroc = cal_diff[cal_diff["metric"] == "AUROC_macro"].iloc[0]
    p1_b3_lc_boot = boot_df[(boot_df["comparison"] == "P1-B3") & np.isclose(boot_df["q"], 0.20) & (boot_df["metric"] == "LowCertCapture@q")].iloc[0]
    p1_b3_err_boot = boot_df[(boot_df["comparison"] == "P1-B3") & np.isclose(boot_df["q"], 0.20) & (boot_df["metric"] == "ErrorCapture@q")].iloc[0]

    results_summary = f"""
    # RESULTS_SUMMARY

    ## Decision principal

    El analisis adicional mas conveniente para el manuscrito es **enriquecimiento de ECG de baja certeza humana**. Cambia muy poco el documento, usa resultados ya existentes y convierte la metrica LowCertCapture en una lectura operacional: cuantos ECG de baja certeza aparecen dentro de una cola de revision de tamano fijo.

    ## Resultado principal post-hoc

    Con q=0.20, P1 alcanza LowCertCapture={fmt(p1_q20['LowCertCapture@q_mean'])} IC95 [{fmt(p1_q20['LowCertCapture@q_ci95_low'])}, {fmt(p1_q20['LowCertCapture@q_ci95_high'])}], frente a B3={fmt(b3_q20['LowCertCapture@q_mean'])} IC95 [{fmt(b3_q20['LowCertCapture@q_ci95_low'])}, {fmt(b3_q20['LowCertCapture@q_ci95_high'])}].

    La diferencia P1-B3 en LowCertCapture@20 fue {fmt(p1_b3_lc_boot['observed_diff_seed_mean'])}, bootstrap por paciente IC95 [{fmt(p1_b3_lc_boot['ci95_low_patient_bootstrap'])}, {fmt(p1_b3_lc_boot['ci95_high_patient_bootstrap'])}].

    En precision de baja certeza dentro de los derivados, P1 obtiene {fmt(p1_q20['LowCertPrecision@q_mean'])}, con lift {fmt(p1_q20['LowCertLift@q_mean'])} y NNR {fmt(p1_q20['NNR_low_cert@q_mean'])}. B3 obtiene precision {fmt(b3_q20['LowCertPrecision@q_mean'])}, lift {fmt(b3_q20['LowCertLift@q_mean'])} y NNR {fmt(b3_q20['NNR_low_cert@q_mean'])}.

    ## Resultado que no debe sobre-venderse

    P1 no mejora la captura agregada de errores. Con q=0.20, ErrorCapture P1={fmt(p1_q20['ErrorCapture@q_mean'])} y B3={fmt(b3_q20['ErrorCapture@q_mean'])}. El bootstrap por paciente estima una diferencia P1-B3={fmt(p1_b3_err_boot['observed_diff_seed_mean'])} IC95 [{fmt(p1_b3_err_boot['ci95_low_patient_bootstrap'])}, {fmt(p1_b3_err_boot['ci95_high_patient_bootstrap'])}], mientras que el resumen entre semillas del manuscrito previo cruzaba cero; por prudencia, conviene reportarlo como ausencia de mejora en captura de errores, no como afirmacion central de inferioridad.

    ## Complemento de calibracion

    B2 frente a B1 muestra una mejora descriptiva de calibracion: diferencia B2-B1 en Brier macro={fmt(b2_brier['diff_mean'])} IC95 semillas [{fmt(b2_brier['diff_ci95_low'])}, {fmt(b2_brier['diff_ci95_high'])}] y ECE macro={fmt(b2_ece['diff_mean'])} IC95 [{fmt(b2_ece['diff_ci95_low'])}, {fmt(b2_ece['diff_ci95_high'])}]. La diferencia en AUROC macro fue {fmt(b2_auroc['diff_mean'])}.

    ## Narrativa recomendada

    Usar Narrativa A + C:
    - A: la senal aprendida en P1 enriquece la cola de revision con casos de baja certeza humana.
    - C: el uso directo de likelihoods como soft labels mejora calibracion en B2 con costo discriminativo pequeno.

    No usar como afirmacion principal que P1 detecta mejor errores del modelo.
    """
    write_text(dirs["reports"] / "RESULTS_SUMMARY.md", results_summary)

    manuscript_insert = f"""
    # MANUSCRIPT_INSERT

    ## Texto breve para Resultados

    Para caracterizar la utilidad operacional del score de revision, se realizo un analisis post-hoc de enriquecimiento de casos de baja certeza humana. Al derivar el 20% de los ECG con mayor score, P1 recupero {fmt(p1_q20['LowCertCapture@q_mean'])} de los casos de baja certeza, frente a {fmt(b3_q20['LowCertCapture@q_mean'])} con B3 y {fmt(p0_q20['LowCertCapture@q_mean'])} con P0. En terminos de rendimiento de la cola de revision, P1 alcanzo una precision de baja certeza de {fmt(p1_q20['LowCertPrecision@q_mean'])}, equivalente a un lift de {fmt(p1_q20['LowCertLift@q_mean'])} respecto de una seleccion aleatoria y a un NNR de {fmt(p1_q20['NNR_low_cert@q_mean'])} ECG revisados por cada caso de baja certeza identificado. La diferencia P1-B3 en LowCertCapture@20 fue {fmt(p1_b3_lc_boot['observed_diff_seed_mean'])}, con IC95 bootstrap por paciente [{fmt(p1_b3_lc_boot['ci95_low_patient_bootstrap'])}, {fmt(p1_b3_lc_boot['ci95_high_patient_bootstrap'])}].

    Este beneficio no se traslado a la captura agregada de errores del clasificador: ErrorCapture@20 fue {fmt(p1_q20['ErrorCapture@q_mean'])} para P1 y {fmt(b3_q20['ErrorCapture@q_mean'])} para B3. Considerando que el analisis entre semillas no apoyo una mejora consistente en ErrorCapture, el score multitarea debe interpretarse como una senal de ambiguedad humana aprendida, no como un detector general de fallos algorítmicos.

    Como complemento, el entrenamiento B2 con etiquetas suaves mostro menor Brier macro que B1 (diferencia B2-B1={fmt(b2_brier['diff_mean'])}) y menor ECE macro (diferencia={fmt(b2_ece['diff_mean'])}), con una diferencia pequena en AUROC macro ({fmt(b2_auroc['diff_mean'])}). Este resultado sugiere que los likelihoods pueden aportar informacion probabilistica util para calibracion, aunque esta observacion debe presentarse separada de la utilidad selectiva de P1.

    ## Texto breve para Discusion

    Los analisis post-hoc refuerzan que la contribucion del enfoque no esta en superar uniformemente a una regla de incertidumbre del modelo en deteccion de errores, sino en capturar una dimension distinta: la baja certeza expresada en las anotaciones humanas de PTB-XL. Esta distincion es metodologicamente relevante, porque permite organizar una cola de revision con mayor concentracion de ECG ambiguos para el anotador, aun cuando dichos casos no coincidan necesariamente con los errores del clasificador.

    ## Figura recomendada

    Insertar `fig_posthoc_low_cert_enrichment.png` como figura nueva o complemento de la figura de captura. Caption sugerido:

    "Analisis post-hoc de enriquecimiento de casos de baja certeza humana. Las curvas muestran, para distintos presupuestos de derivacion q, la captura acumulada de ECG con review_binary=1, la precision de baja certeza dentro de la cola derivada y el lift respecto de seleccion aleatoria. Las bandas representan IC95 entre cinco semillas."

    ## Tabla recomendada

    Insertar `table_new_low_cert_enrichment_q20.tex`. Caption sugerido:

    "Utilidad operacional de los scores de derivacion al revisar el 20% de ECG con mayor score. LowCertCapture mide la fraccion de casos de baja certeza recuperados; Precision indica la proporcion de baja certeza dentro de los derivados; Lift compara esa precision contra seleccion aleatoria; NNR corresponde al numero de ECG revisados por cada caso de baja certeza identificado."

    ## Cambio minimo sugerido al abstract

    Agregar una frase:

    "En analisis post-hoc, el score multitarea aumento el enriquecimiento de ECG de baja certeza humana dentro de la cola de revision, aunque no mejoro la captura agregada de errores del clasificador."
    """
    write_text(dirs["reports"] / "MANUSCRIPT_INSERT.md", manuscript_insert)

    decision = f"""
    # analysis_decision

    ## Metodologia seleccionada

    1. Enriquecimiento de baja certeza humana como analisis post-hoc principal.
    2. Calibracion B1 vs B2 como complemento breve.
    3. Cuadrantes humano-modelo y captura por clase como material visual opcional para presentacion o apendice.

    ## Alternativas descartadas para esta entrega

    - Ensemble/hibrido: requiere una seleccion adicional de definicion de score y podria abrir una linea metodologica nueva.
    - SHAP/UMAP/casos ECG: mas costoso y menos robusto para incorporar en dos dias.
    - Artefactos: no hay variables tecnicas suficientes en `master_predictions`.

    ## Interpretacion correcta

    La afirmacion fuerte y honesta es:

    "P1 concentra mejor que B3 los ECG de baja certeza humana en una cola de revision, pero no debe presentarse como mejor detector global de errores del modelo."

    Esta lectura permite anadir una ventaja concreta del metodo sin cambiar el documento completo.
    """
    write_text(dirs["reports"] / "analysis_decision.md", decision)

    # Final report.
    final_report = f"""
    # POSTHOC_FINAL_REPORT

    ## Que se reviso
    - ZIP: `{zip_path}`
    - Runs full: {len(runs)}
    - Predicciones test/validacion, metricas, bootstrap y tablas por clase.

    ## Analisis ejecutados
    - Enriquecimiento de baja certeza por q.
    - Bootstrap pareado por paciente para diferencias P1-B3 y P0-B3.
    - Calibracion B2 vs B1.
    - Cuadrantes P1/B3 con cortes p80 definidos en validacion.
    - Captura por clase para q=0.10 y q=0.20.

    ## Mejor analisis adicional
    Enriquecimiento de baja certeza humana.

    ## Segundo mejor analisis
    Calibracion B1 vs B2.

    ## Archivos principales
    - `tables/posthoc_enrichment_summary.csv`
    - `tables/posthoc_bootstrap_enrichment_diffs.csv`
    - `tables/table_new_low_cert_enrichment_q20.csv`
    - `figures/fig_posthoc_low_cert_enrichment.png`
    - `figures/fig_posthoc_pareto_error_vs_lowcert_q20.png`
    - `figures/fig_posthoc_b2_vs_b1_calibration.png`

    ## Tareas pendientes
    - Insertar una figura y una tabla en el manuscrito.
    - Decidir si la figura de cuadrantes queda en presentacion o apendice.
    - No reemplazar la conclusion central: P1 aprende ambiguedad humana, no error general.
    """
    write_text(dirs["reports"] / "POSTHOC_FINAL_REPORT.md", final_report)

    print("OK")
    print(f"OUT={out}")
    print(f"runs={len(runs)}")
    print(f"P1 LowCertCapture@20={p1_q20['LowCertCapture@q_mean']:.4f}")
    print(f"B3 LowCertCapture@20={b3_q20['LowCertCapture@q_mean']:.4f}")
    print(f"B2-B1 Brier diff={b2_brier['diff_mean']:.4f}")
    print(f"B2-B1 ECE diff={b2_ece['diff_mean']:.4f}")


if __name__ == "__main__":
    main()
