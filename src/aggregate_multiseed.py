"""Agregacion multi-semilla para HCG-SelectiveECG (standalone, sin Colab/GPU).

Reproduce la logica de la celda 25 del notebook a partir de los CSV ya escritos
por cada corrida de semilla, y agrega la seccion de efecto del balanceo
(balance_effect.csv) que faltaba en el agregado original.

Uso desde la raiz del paquete:
    python 02_CODIGO/src/aggregate_multiseed.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _stats

# --- Rutas portables dentro del paquete de entrega ---
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PACKAGE_ROOT / "03_RESULTADOS" / "runs"
AGG_DIR = PACKAGE_ROOT / "03_RESULTADOS" / "aggregate_multiseed"
AGG_DIR.mkdir(parents=True, exist_ok=True)

RUN_MODE = "full"
SEED_LIST = [42, 13, 123, 2024, 7]
SEED_TO_RUNID = {seed: f"hcg_selective_ecg_{RUN_MODE}_seed{seed}" for seed in SEED_LIST}
SEED_TO_RUNID[42] = "hcg_selective_ecg_full_20260712_131520"  # corrida original


def _seed_tables_dir(seed):
    return RUNS_DIR / SEED_TO_RUNID[seed] / "results" / "tables"


available_seeds = [s for s in SEED_LIST if _seed_tables_dir(s).exists()]
missing_seeds = [s for s in SEED_LIST if s not in available_seeds]
print("Semillas disponibles para agregar:", available_seeds)
if missing_seeds:
    print("Semillas faltantes (aun no corridas):", missing_seeds)


def _read_seed_table(seed, name):
    p = _seed_tables_dir(seed) / name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["seed"] = seed
    return df


def _concat_tables(name):
    parts = [t for t in (_read_seed_table(s, name) for s in available_seeds) if t is not None]
    return pd.concat(parts, ignore_index=True) if parts else None


def _agg_mean_sd(df, group_cols, value_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for v in value_cols:
            vals = pd.to_numeric(g[v], errors="coerce").dropna().values
            row["n_seeds"] = int(len(vals))
            row[f"{v}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{v}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            if len(vals) > 1:
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                tcrit = float(_stats.t.ppf(0.975, len(vals) - 1))
                row[f"{v}_ci95_low"] = float(np.mean(vals) - tcrit * se)
                row[f"{v}_ci95_high"] = float(np.mean(vals) + tcrit * se)
            else:
                row[f"{v}_ci95_low"] = np.nan
                row[f"{v}_ci95_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _agg_paired_diff(df, value_col, group_cols=None, label=None):
    """Diferencia pareada P1 - P0 por semilla, agregada entre semillas.

    df debe tener columnas: seed, model, <value_col>.
    Empareja P1_MTL_BALANCED_PROPOSED menos P0_MTL_UNBALANCED dentro de cada seed
    (y dentro de cada valor de group_cols si se especifica).
    """
    p0 = df[df["model"] == "P0_MTL_UNBALANCED"]
    p1 = df[df["model"] == "P1_MTL_BALANCED_PROPOSED"]
    keys = ["seed"] + (group_cols or [])
    merged = p1.merge(p0, on=keys, suffixes=("_p1", "_p0"))
    merged["diff"] = pd.to_numeric(merged[f"{value_col}_p1"], errors="coerce") - pd.to_numeric(
        merged[f"{value_col}_p0"], errors="coerce"
    )
    out_rows = []
    grp = merged.groupby(group_cols, dropna=False) if group_cols else [((), merged)]
    for gkeys, g in grp:
        gkeys = gkeys if isinstance(gkeys, tuple) else (gkeys,)
        vals = g["diff"].dropna().values
        row = {"metric": label or value_col}
        if group_cols:
            row.update(dict(zip(group_cols, gkeys)))
        row["n_seeds"] = int(len(vals))
        row["diff_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        row["diff_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        if len(vals) > 1:
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            tcrit = float(_stats.t.ppf(0.975, len(vals) - 1))
            row["diff_ci95_low"] = float(np.mean(vals) - tcrit * se)
            row["diff_ci95_high"] = float(np.mean(vals) + tcrit * se)
        else:
            row["diff_ci95_low"] = np.nan
            row["diff_ci95_high"] = np.nan
        row["significativo_entre_semillas"] = bool(
            (row["diff_ci95_low"] > 0) or (row["diff_ci95_high"] < 0)
        ) if len(vals) > 1 else False
        out_rows.append(row)
    return pd.DataFrame(out_rows)


summaries = {}

if len(available_seeds) == 0:
    raise SystemExit("No hay corridas disponibles; corre al menos una semilla antes de agregar.")

# --- Diagnostico (solo test) ---
diag = _concat_tables("diagnostic_metrics_global.csv")
if diag is not None:
    diag_test = diag[diag["split"] == "test"].copy()
    diag_agg = _agg_mean_sd(
        diag_test, ["model"],
        ["AUROC_macro_thrval", "AUPRC_macro_thrval", "F1_macro_thrval",
         "Brier_macro_thrval", "ECE_macro_thrval"],
    )
    diag_agg.to_csv(AGG_DIR / "multiseed_diagnostic_global_test.csv", index=False)
    summaries["diagnostic"] = diag_agg

# --- Revision medica (test) ---
rev = _concat_tables("review_metrics.csv")
if rev is not None:
    rev_test = rev[rev["split"] == "test"].copy() if "split" in rev.columns else rev
    rev_agg = _agg_mean_sd(
        rev_test, ["model_or_score"],
        ["review_AUROC", "review_AUPRC", "review_F1",
         "MAE_ambiguity", "Spearman_ambiguity"],
    )
    rev_agg.to_csv(AGG_DIR / "multiseed_review_metrics.csv", index=False)
    summaries["review"] = rev_agg

# --- Derivacion selectiva en q=0.20 ---
sel = _concat_tables("selective_referral_metrics.csv")
if sel is not None:
    sel20 = sel[np.isclose(sel["q"], 0.20)].copy()
    sel_agg = _agg_mean_sd(
        sel20, ["model_or_score"],
        ["ErrorCapture@q", "LowCertCapture@q", "F1_accepted",
         "Brier_accepted", "ECE_accepted"],
    )
    sel_agg.to_csv(AGG_DIR / "multiseed_selective_referral_q20.csv", index=False)
    summaries["selective_q20"] = sel_agg

# --- Diferencias pareadas (bootstrap por semilla -> agregacion entre semillas) ---
boot = _concat_tables("bootstrap_summary.csv")
if boot is not None:
    boot_agg = _agg_mean_sd(boot, ["comparison"], ["mean"]).rename(columns={
        "mean_mean": "diff_mean", "mean_sd": "diff_sd",
        "mean_ci95_low": "diff_ci95_low", "mean_ci95_high": "diff_ci95_high",
    })
    boot_agg["significativo_entre_semillas"] = (
        (boot_agg["diff_ci95_low"] > 0) | (boot_agg["diff_ci95_high"] < 0)
    )
    boot_agg.to_csv(AGG_DIR / "multiseed_paired_diffs.csv", index=False)
    summaries["paired_diffs"] = boot_agg

# --- Robustez por subgrupo (sexo, edad), agregado entre semillas ---
# La tabla trae B1_HARD_DIAG (fila que usa el score de incertidumbre por umbral)
# y P1_MTL_BALANCED_PROPOSED (cabeza de revision aprendida). Derivacion
# intra-subgrupo al 20%.
subg = _concat_tables("subgroup_metrics.csv")
if subg is not None:
    subg_agg = _agg_mean_sd(
        subg, ["model", "subgroup"],
        ["N", "AUROC_macro", "AUPRC_macro", "F1_macro", "ECE_macro",
         "review_AUPRC", "ErrorCapture@20"],
    )
    subg_agg.to_csv(AGG_DIR / "multiseed_subgroup_metrics.csv", index=False)
    summaries["subgroup"] = subg_agg

# --- Efecto del balanceo por bins (P0 vs P1), agregado entre semillas ---
bal = _concat_tables("balance_effect.csv")
if bal is not None:
    bal_value_cols = ["review_AUPRC", "MAE_B0", "MAE_B1", "MAE_B2", "MAE_B3",
                      "ErrorCapture@20", "LowCertCapture@20"]
    # (a) media/sd/IC por modelo (P0 y P1 por separado)
    bal_agg = _agg_mean_sd(bal, ["model"], bal_value_cols)
    bal_agg.to_csv(AGG_DIR / "multiseed_balance_effect_by_model.csv", index=False)
    summaries["balance_by_model"] = bal_agg
    # (b) diferencia pareada P1 - P0 por semilla, agregada
    bal_diff = pd.concat(
        [_agg_paired_diff(bal, v, label=v) for v in bal_value_cols],
        ignore_index=True,
    )
    bal_diff.to_csv(AGG_DIR / "multiseed_balance_effect_paired_diffs.csv", index=False)
    summaries["balance_paired"] = bal_diff


# --- Resumen legible en Markdown ---
def _fmt(df):
    d = df.round(4)
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


lines = []
lines.append("# MULTISEED SUMMARY - HCG-SelectiveECG\n")
lines.append(f"- Semillas agregadas: {available_seeds}")
lines.append(f"- Semillas faltantes: {missing_seeds}")
lines.append(f"- n_semillas efectivas: {len(available_seeds)}\n")
lines.append("Un efecto se considera robusto si su IC95% entre semillas no cruza 0.\n")
if "paired_diffs" in summaries:
    lines.append("## Diferencias pareadas (media entre semillas)\n")
    lines.append(_fmt(summaries["paired_diffs"]))
    lines.append("")
if "review" in summaries:
    lines.append("## Revision medica (test)\n")
    lines.append(_fmt(summaries["review"]))
    lines.append("")
if "selective_q20" in summaries:
    lines.append("## Derivacion selectiva @ q=0.20\n")
    lines.append(_fmt(summaries["selective_q20"]))
    lines.append("")
if "diagnostic" in summaries:
    lines.append("## Diagnostico (test)\n")
    lines.append(_fmt(summaries["diagnostic"]))
    lines.append("")
if "subgroup" in summaries:
    lines.append("## Robustez por subgrupo (sexo, edad) - media entre semillas\n")
    lines.append(
        "Fila B1 = referencia por umbral; fila P1 = cabeza de revision aprendida. "
        "Derivacion intra-subgrupo al 20%.\n"
    )
    lines.append(_fmt(summaries["subgroup"]))
    lines.append("")
if "balance_by_model" in summaries:
    lines.append("## Efecto del balanceo por bins - media entre semillas (P0 vs P1)\n")
    lines.append(_fmt(summaries["balance_by_model"]))
    lines.append("")
if "balance_paired" in summaries:
    lines.append("## Efecto del balanceo por bins - diferencia pareada P1 menos P0\n")
    lines.append(
        "Signo esperado: MAE en bins altos (B2, B3) deberia BAJAR con balanceo "
        "(diff negativo). review_AUPRC y las capturas deberian SUBIR (diff positivo).\n"
    )
    lines.append(_fmt(summaries["balance_paired"]))
    lines.append("")
(AGG_DIR / "MULTISEED_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

print("Agregacion multi-semilla escrita en:", AGG_DIR)
if "balance_paired" in summaries:
    print("\nEfecto del balanceo (P1 - P0), diferencia pareada entre semillas:")
    print(summaries["balance_paired"].round(4).to_string(index=False))
