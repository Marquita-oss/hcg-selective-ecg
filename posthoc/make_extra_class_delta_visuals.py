from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "03_RESULTADOS" / "posthoc"
TABLE = ROOT / "tables" / "posthoc_class_capture_summary.csv"
FIG_DIR = PACKAGE_ROOT / "05_FIGURAS" / "analisis"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
METRICS = [
    ("ClassErrorCapture@q_mean", "Error total"),
    ("FalseNegativeCapture@q_mean", "Falso negativo"),
    ("FalsePositiveCapture@q_mean", "Falso positivo"),
]


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def blend(a, b, t):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return tuple((a * (1 - t) + b * t).astype(int))


def color_for_delta(x: float, vmax: float):
    t = min(abs(x) / vmax, 1.0)
    neutral = (245, 245, 242)
    good = (35, 140, 126)
    bad = (186, 74, 64)
    return blend(neutral, good if x > 0 else bad, 0.18 + 0.72 * t)


def save_image(img: Image.Image, base: Path):
    img.save(base.with_suffix(".png"), dpi=(300, 300))
    img.convert("RGB").save(base.with_suffix(".pdf"), "PDF", resolution=300.0)


def make_heatmap(q: float) -> pd.DataFrame:
    df = pd.read_csv(TABLE)
    dfq = df[(np.isclose(df["q"], q)) & (df["score"].isin(["B3", "P1"]))].copy()
    rows = []
    for cls in CLASSES:
        b3 = dfq[(dfq["score"] == "B3") & (dfq["class"] == cls)].iloc[0]
        p1 = dfq[(dfq["score"] == "P1") & (dfq["class"] == cls)].iloc[0]
        row = {"class": cls, "q": q}
        for metric, label in METRICS:
            row[label] = float(p1[metric] - b3[metric])
            row[f"{label}_P1"] = float(p1[metric])
            row[f"{label}_B3"] = float(b3[metric])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "tables" / f"posthoc_delta_p1_minus_b3_class_fpfn_q{int(q*100):02d}.csv", index=False)

    W, H = 1050, 620
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title = f"Diferencia P1 - B3 en captura por clase y tipo de error (q={q:.2f})"
    draw.text((45, 25), title, fill=(20, 20, 20), font=font(24, True))
    draw.text((45, 62), "Valores positivos favorecen P1; valores negativos favorecen B3.", fill=(75, 75, 75), font=font(15))

    x0, y0 = 210, 135
    cell_w, cell_h = 240, 70
    vmax = max(0.05, max(out[label].abs().max() for _, label in METRICS))

    for j, (_, label) in enumerate(METRICS):
        draw.text((x0 + j * cell_w + 25, y0 - 40), label, fill=(30, 30, 30), font=font(16, True))

    for i, cls in enumerate(CLASSES):
        y = y0 + i * cell_h
        draw.text((70, y + 22), cls, fill=(25, 25, 25), font=font(17, True))
        for j, (_, label) in enumerate(METRICS):
            x = x0 + j * cell_w
            val = float(out.loc[out["class"] == cls, label].iloc[0])
            draw.rectangle((x, y, x + cell_w - 10, y + cell_h - 10), fill=color_for_delta(val, vmax), outline=(255, 255, 255), width=2)
            draw.text((x + 72, y + 18), f"{val:+.3f}", fill=(20, 20, 20), font=font(19, True))

    # Small legend.
    lx, ly = 230, 525
    for k, (label, color) in enumerate([("favorece B3", (186, 74, 64)), ("sin cambio", (245, 245, 242)), ("favorece P1", (35, 140, 126))]):
        x = lx + k * 210
        draw.rectangle((x, ly, x + 34, ly + 24), fill=color, outline=(120, 120, 120))
        draw.text((x + 44, ly + 2), label, fill=(45, 45, 45), font=font(13))

    save_image(img, FIG_DIR / f"fig_posthoc_delta_p1_minus_b3_class_fpfn_q{int(q*100):02d}")
    return out


def main():
    for q in [0.10, 0.20]:
        make_heatmap(q)
    print("OK extra class delta visuals")


if __name__ == "__main__":
    main()
