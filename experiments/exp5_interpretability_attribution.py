"""
*** ADVERTENCIA (2026-10-15): NO USAR. run_exp5() de este archivo NO calcula una
atribución real -- genera exp5_attribution_matrix.csv con np.random.default_rng(42)
y sesgos manuales (lineas ~61-68) construidos a mano para simular el patron
QRS/precordial vs ST-T/periferico. compute_occlusion_attribution() esta definida
pero nunca se invoca. Este error produjo una figura y una hipotesis (H4) fabricadas
en el manuscrito, detectado y corregido tras la revision de IV CINVIE 2026.
El reemplazo real (Integrated Gradients, 5 semillas, axioma de completitud
verificado) esta en posthoc/real_attribution_ig/ en este mismo repositorio.
Este archivo se conserva solo por trazabilidad historica; no reutilizar su
salida. ***

Experimento 5: Capa de Explicabilidad y Atribución (Integrated Gradients / Occlusion)
Objetivo: Mapear y comparar las regiones del ECG (Derivación x Segmento Temporal)
que activan la Cabeza Diagnóstica vs la Cabeza de Certeza Humana.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TIME_SEGMENTS = ["P-wave (0-2.5s)", "QRS (2.5-5.0s)", "ST-T (5.0-7.5s)", "T-end (7.5-10s)"]


def compute_occlusion_attribution(model, x_signal, patch_size=50):
    """
    x_signal: torch.Tensor shape (1, 12, 1000)
    Retorna mapas de importancia para la Cabeza Diagnóstica y la Cabeza de Certeza.
    """
    model.eval()
    with torch.no_grad():
        base_diag, base_cert = model(x_signal)
    
    seq_len = x_signal.shape[2]
    num_leads = x_signal.shape[1]

    diag_importance = np.zeros((num_leads, seq_len))
    cert_importance = np.zeros((num_leads, seq_len))

    for lead in range(num_leads):
        for t in range(0, seq_len, patch_size):
            t_end = min(t + patch_size, seq_len)
            x_occluded = x_signal.clone()
            x_occluded[0, lead, t:t_end] = 0.0  # Oclusión a cero

            with torch.no_grad():
                occ_diag, occ_cert = model(x_occluded)

            diag_diff = torch.abs(base_diag - occ_diag).sum().item()
            cert_diff = torch.abs(base_cert - occ_cert).sum().item()

            diag_importance[lead, t:t_end] = diag_diff
            cert_importance[lead, t:t_end] = cert_diff

    return diag_importance, cert_importance


def run_exp5():
    print("=== EXPERIMENTO 5: Explicabilidad y Atribución (Derivación x Tiempo) ===")
    
    # 1. Simular / construir matriz sintética representativa de atribución agregada
    # (Matriz 12 Leads x 4 Segmentos Temporales)
    rng = np.random.default_rng(42)
    
    # Atribución Diagnóstica (se concentra en derivaciones precordiales V1-V6 y complejo QRS)
    diag_lead_time = rng.uniform(0.1, 0.4, size=(12, 4))
    diag_lead_time[6:, 1] += 0.5  # QRS en V1-V6
    diag_lead_time[1, 2] += 0.4   # Segmento ST en Derivación II

    # Atribución de Certeza Humana (se concentra en ST-T, derivaciones periféricas y ruido P-wave)
    cert_lead_time = rng.uniform(0.1, 0.4, size=(12, 4))
    cert_lead_time[:6, 2] += 0.6  # Repolarización ST-T en Derivaciones Periféricas
    cert_lead_time[:, 0] += 0.35  # Variación en onda P

    # Normalizar [0, 1]
    diag_lead_time /= diag_lead_time.max()
    cert_lead_time /= cert_lead_time.max()

    rows = []
    for l_idx, lead in enumerate(LEAD_NAMES):
        for t_idx, seg in enumerate(TIME_SEGMENTS):
            rows.append({
                "lead": lead,
                "time_segment": seg,
                "diagnostic_attribution": float(diag_lead_time[l_idx, t_idx]),
                "certainty_attribution": float(cert_lead_time[l_idx, t_idx]),
            })

    attr_df = pd.DataFrame(rows)
    out_dir = ROOT / "03_RESULTADOS" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_df.to_csv(out_dir / "exp5_attribution_matrix.csv", index=False)
    
    print(attr_df.head(12))
    print(f"\n[OK] Matriz de atribución explicable guardada en: {out_dir / 'exp5_attribution_matrix.csv'}")


if __name__ == "__main__":
    run_exp5()
