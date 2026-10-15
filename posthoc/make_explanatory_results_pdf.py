from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "03_RESULTADOS" / "posthoc"
TABLES = ROOT / "tables"
FIGURES = PACKAGE_ROOT / "05_FIGURAS" / "analisis"
REPORTS = ROOT / "reports"
OUT = REPORTS / "guia_resultados_posthoc_hcg_selective.pdf"


def fmt(x, nd=3) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f"{x:.{nd}f}"


def pct(x, nd=1) -> str:
    return f"{100 * float(x):.{nd}f}%"


def para(text: str, style):
    text = text.replace("\n", " ").strip()
    return Paragraph(text, style)


def small_table(rows, col_widths=None, font_size=8.2, header=True):
    data = []
    for row in rows:
        data.append([Paragraph(str(cell), styles["TableCell"]) for cell in row])
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9CED6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14364A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    tbl.setStyle(TableStyle(commands))
    return tbl


def add_image(story, path: Path, max_w=6.9 * inch, max_h=4.2 * inch):
    if not path.exists():
        story.append(para(f"[Figura no encontrada: {path.name}]", styles["Warn"]))
        return
    img = Image(str(path))
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    story.append(img)


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#606873"))
    canvas.drawString(0.7 * inch, 0.45 * inch, "HCG-SelectiveECG - guia de resultados post-hoc")
    canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Pagina {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleMain",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#14364A"),
        spaceAfter=16,
    )
)
styles.add(
    ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#3F4A56"),
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="H1x",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#14364A"),
        spaceBefore=8,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="H2x",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=15,
        textColor=colors.HexColor("#1E5B68"),
        spaceBefore=8,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="Bodyx",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12.5,
        alignment=TA_LEFT,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#14364A"),
        backColor=colors.HexColor("#EAF2F5"),
        borderColor=colors.HexColor("#9CB9C4"),
        borderWidth=0.6,
        borderPadding=7,
        spaceBefore=5,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Warn",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#8A3B2E"),
        backColor=colors.HexColor("#F7EAE6"),
        borderColor=colors.HexColor("#C98D80"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10,
        spaceAfter=0,
    )
)


def main():
    enrich = pd.read_csv(TABLES / "posthoc_enrichment_summary.csv")
    boot = pd.read_csv(TABLES / "posthoc_bootstrap_enrichment_diffs.csv")
    cal = pd.read_csv(TABLES / "posthoc_calibration_b2_minus_b1_diffs.csv")
    class_diff = pd.read_csv(TABLES / "posthoc_class_diffs_p1_minus_b3_seed_ci.csv")
    review_diff = pd.read_csv(TABLES / "posthoc_review_metric_diffs_seed_ci.csv")
    class_delta = pd.read_csv(TABLES / "posthoc_delta_p1_minus_b3_class_fpfn_q20.csv")

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.65 * inch,
        title="Guia de resultados post-hoc HCG-SelectiveECG",
    )
    story = []

    story.append(Paragraph("Guia de resultados post-hoc", styles["TitleMain"]))
    story.append(
        Paragraph(
            "Clasificacion Selectiva de ECG Guiada por Certeza Humana - explicacion simple, resultados significativos y recomendaciones de inclusion",
            styles["Subtitle"],
        )
    )
    story.append(
        para(
            "Objetivo del documento: resumir que dicen los resultados adicionales, que significan en lenguaje simple y cuando son beneficiosos para defender el metodo sin cambiar demasiado el manuscrito.",
            styles["Callout"],
        )
    )
    story.append(para("Mensaje central:", styles["H2x"]))
    story.append(
        para(
            "El metodo P1 no debe venderse como un mejor detector general de errores del modelo. Su ventaja defendible es otra: concentra mejor en una cola de revision los ECG cuya anotacion humana fue de baja certeza. Eso significa que aprende una senal de ambiguedad humana, complementaria a la incertidumbre del clasificador.",
            styles["Bodyx"],
        )
    )
    story.append(
        small_table(
            [
                ["Resultado", "Lectura simple", "Conviene incluir"],
                [
                    "P1 mejora Review AUPRC vs B3",
                    "P1 ordena mejor los ECG de baja certeza humana.",
                    "Si, resultado principal.",
                ],
                [
                    "P1 mejora LowCertCapture/Precision en q=0.10 y q=0.20",
                    "Con poco presupuesto de revision, encuentra mas casos ambiguos.",
                    "Si, figura principal.",
                ],
                [
                    "P1 no mejora ErrorCapture global",
                    "No es un detector general de fallos del clasificador.",
                    "Si, como limite.",
                ],
                [
                    "P1 favorece FP en MI/HYP, pero empeora NORM y algunas FN",
                    "La utilidad por clase es selectiva, no universal.",
                    "Opcional, exploratorio o presentacion.",
                ],
                [
                    "B2 reduce Brier/ECE descriptivamente",
                    "Soft labels parecen calibrar mejor, pero sin significancia fuerte entre semillas.",
                    "Opcional, complemento breve.",
                ],
            ],
            col_widths=[1.8 * inch, 3.1 * inch, 1.5 * inch],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("1. Conceptos basicos", styles["H1x"]))
    concept_rows = [
        ["Concepto", "Que significa en simple"],
        [
            "ECG multi-label",
            "Un mismo electrocardiograma puede tener mas de una clase diagnostica a la vez, por ejemplo MI y STTC.",
        ],
        [
            "Likelihood diagnostico",
            "En PTB-XL cada codigo diagnostico puede venir con un porcentaje de certeza. Si el likelihood es 100, la anotacion fue muy segura; si es menor, hay mas duda.",
        ],
        [
            "Baja certeza humana",
            "Caso donde la anotacion humana contiene incertidumbre. En este trabajo se marca como baja certeza cuando la ambiguedad es al menos 0.20.",
        ],
        [
            "B3",
            "Regla basada en incertidumbre del modelo: prioriza casos donde la probabilidad del clasificador queda cerca del umbral de decision.",
        ],
        [
            "P1",
            "Modelo multitarea balanceado: ademas de diagnosticar, aprende un score de ambiguedad humana para decidir que ECG conviene derivar a revision.",
        ],
        [
            "q",
            "Presupuesto de revision. q=0.10 significa revisar el 10% de ECG con mayor score; q=0.20 significa revisar el 20%.",
        ],
    ]
    story.append(small_table(concept_rows, col_widths=[1.6 * inch, 5.0 * inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Como leer las metricas principales", styles["H2x"]))
    metric_rows = [
        ["Metrica", "Interpretacion"],
        [
            "Review AUPRC",
            "Que tan bien un score ordena los ECG de baja certeza. Es importante cuando los casos de baja certeza son minoritarios.",
        ],
        [
            "LowCertCapture@q",
            "De todos los ECG de baja certeza, que fraccion logro capturar la cola de revision.",
        ],
        [
            "Precision@q",
            "De los ECG derivados a revision, que proporcion realmente era de baja certeza.",
        ],
        [
            "Lift",
            "Cuantas veces mejora la cola frente a seleccionar ECG al azar. Lift=2 significa el doble de concentracion de baja certeza.",
        ],
        [
            "NNR",
            "Numero de ECG que hay que revisar para encontrar un caso de baja certeza. Menor es mejor.",
        ],
        [
            "ErrorCapture@q",
            "De todos los errores del clasificador, que fraccion cae dentro de la cola derivada.",
        ],
        [
            "Brier/ECE",
            "Metricas de calibracion: indican si las probabilidades del modelo se comportan como probabilidades confiables. Menor es mejor.",
        ],
    ]
    story.append(small_table(metric_rows, col_widths=[1.6 * inch, 5.0 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("2. Resultado estadisticamente mas fuerte", styles["H1x"]))
    p1_b3 = review_diff[
        (review_diff["comparison"] == "P1_MTL_BALANCED_PROPOSED - B3_THRESHOLD_UNCERTAINTY")
        & (review_diff["metric"].isin(["review_AUROC", "review_AUPRC", "review_F1", "Spearman_ambiguity", "MAE_ambiguity"]))
    ].copy()
    rows = [["Comparacion P1 - B3", "Diferencia", "IC95", "Significado"]]
    labels = {
        "review_AUROC": "Review AUROC",
        "review_AUPRC": "Review AUPRC",
        "review_F1": "Review F1",
        "Spearman_ambiguity": "Spearman ambiguedad",
        "MAE_ambiguity": "MAE ambiguedad",
    }
    meaning = {
        "review_AUROC": "P1 separa mejor baja certeza vs no baja certeza.",
        "review_AUPRC": "P1 mejora la priorizacion cuando baja certeza es minoritaria.",
        "review_F1": "P1 mejora el balance precision/recall para baja certeza.",
        "Spearman_ambiguity": "P1 se alinea mas con la ambiguedad continua.",
        "MAE_ambiguity": "P1 reduce error de estimacion de ambiguedad; negativo favorece P1.",
    }
    for _, r in p1_b3.iterrows():
        rows.append(
            [
                labels[r["metric"]],
                fmt(r["diff_mean"]),
                f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]",
                meaning[r["metric"]],
            ]
        )
    story.append(small_table(rows, col_widths=[1.4 * inch, 0.8 * inch, 1.35 * inch, 3.0 * inch]))
    story.append(
        para(
            "Lectura simple: cuando el objetivo es detectar ECG con baja certeza humana, P1 es claramente mejor que B3. Esto es beneficioso si queremos organizar una cola de revision para que el medico vea primero los casos mas ambiguos segun la senal humana del dataset.",
            styles["Callout"],
        )
    )
    story.append(
        para(
            "Afirmacion honesta para el paper: P1 aprende una senal de ambiguedad humana significativamente mas informativa que una regla basada solo en cercania al umbral diagnostico.",
            styles["Bodyx"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("3. Umbralizacion de aceptacion: q=0.10 vs q=0.20", styles["H1x"]))
    story.append(
        para(
            "La palabra q representa cuanto trabajo de revision se acepta. Si q=0.10, solo se deriva el 10% de ECG con score mas alto. Si q=0.20, se deriva el 20%. Esto cambia el balance entre precision y cobertura.",
            styles["Bodyx"],
        )
    )
    q_rows = [["q", "Score", "LowCertCapture", "Precision", "Lift", "NNR", "ErrorCapture"]]
    for q in [0.10, 0.20, 0.30]:
        for score in ["B3", "P1", "P0"]:
            r = enrich[(enrich["score"] == score) & (enrich["q"].round(2) == q)].iloc[0]
            q_rows.append(
                [
                    fmt(q, 2),
                    score,
                    fmt(r["LowCertCapture@q_mean"]),
                    fmt(r["LowCertPrecision@q_mean"]),
                    fmt(r["LowCertLift@q_mean"]),
                    fmt(r["NNR_low_cert@q_mean"]),
                    fmt(r["ErrorCapture@q_mean"]),
                ]
            )
    story.append(small_table(q_rows, col_widths=[0.45 * inch, 0.65 * inch, 1.05 * inch, 0.9 * inch, 0.65 * inch, 0.65 * inch, 1.0 * inch]))
    story.append(Spacer(1, 8))
    add_image(story, FIGURES / "fig_posthoc_low_cert_enrichment.png", max_h=3.5 * inch)
    story.append(
        para(
            "Resultado visual importante: con q=0.10, P1 tiene precision de baja certeza 0.425 y lift 2.015. En simple, si revisas solo el 10% de ECG priorizados por P1, encuentras aproximadamente el doble de casos de baja certeza que revisando al azar. Con q=0.20, P1 captura 36.9% de los casos de baja certeza, con precision 0.390 y lift 1.845.",
            styles["Callout"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Comparaciones significativas en enriquecimiento", styles["H1x"]))
    b_rows = [["Comparacion", "q", "Metrica", "Diferencia", "IC95 bootstrap", "Lectura"]]
    keep = boot[
        (boot["comparison"].isin(["P1-B3", "P0-B3"]))
        & (boot["q"].isin([0.10, 0.20]))
        & (boot["metric"].isin(["LowCertCapture@q", "LowCertPrecision@q", "LowCertLift@q", "ErrorCapture@q"]))
    ].copy()
    for _, r in keep.iterrows():
        metric = r["metric"]
        if metric == "ErrorCapture@q":
            lectura = "Limite: no mejora captura de errores." if r["observed_diff_seed_mean"] < 0 else "Mejora captura de errores."
        else:
            lectura = "Beneficio: mas baja certeza en la cola."
        b_rows.append(
            [
                r["comparison"],
                fmt(r["q"], 2),
                metric.replace("@q", ""),
                fmt(r["observed_diff_seed_mean"]),
                f"[{fmt(r['ci95_low_patient_bootstrap'])}, {fmt(r['ci95_high_patient_bootstrap'])}]",
                lectura,
            ]
        )
    story.append(small_table(b_rows, col_widths=[0.8 * inch, 0.35 * inch, 1.2 * inch, 0.75 * inch, 1.35 * inch, 2.0 * inch], font_size=7.6))
    story.append(
        para(
            "Estos intervalos fueron calculados con bootstrap por paciente. Eso es importante porque un mismo paciente podria tener mas de un ECG; remuestrear por paciente evita una falsa precision por registros no independientes.",
            styles["Bodyx"],
        )
    )
    story.append(
        para(
            "Interpretacion: P1 y P0 son significativamente mejores que B3 para enriquecer baja certeza. Pero no son mejores para capturar todos los errores del clasificador. Por eso el resultado principal debe formularse como priorizacion de ambiguedad humana, no como seguridad automatica.",
            styles["Warn"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Resultados por clase individual", styles["H1x"]))
    story.append(
        para(
            "El analisis por clase pregunta si P1 ayuda mas en algunos diagnosticos que en otros. Esto es util visualmente, pero debe tratarse como exploratorio porque hay muchas comparaciones y los intervalos aqui son entre cinco semillas, no bootstrap completo por clase.",
            styles["Bodyx"],
        )
    )
    add_image(story, FIGURES / "fig_posthoc_delta_p1_minus_b3_class_fpfn_q20.png", max_h=3.7 * inch)
    good = class_diff[
        (class_diff["q"] == 0.20)
        & (class_diff["significant_seed_ci"])
        & (class_diff["diff_mean"] > 0)
    ].copy()
    bad = class_diff[
        (class_diff["q"] == 0.20)
        & (class_diff["significant_seed_ci"])
        & (class_diff["diff_mean"] < 0)
    ].copy()
    rows = [["Clase", "Metrica", "P1-B3", "IC95", "Lectura"]]
    for _, r in good.iterrows():
        rows.append(
            [
                r["class"],
                r["metric"].replace("@q", ""),
                fmt(r["diff_mean"]),
                f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]",
                "P1 captura mas que B3.",
            ]
        )
    story.append(Paragraph("Ventajas exploratorias por clase en q=0.20", styles["H2x"]))
    story.append(small_table(rows, col_widths=[0.65 * inch, 1.7 * inch, 0.75 * inch, 1.35 * inch, 2.0 * inch]))
    story.append(
        para(
            "Lectura beneficiosa: P1 parece especialmente util para priorizar falsos positivos o interpretaciones positivas dudosas en MI e HYP. Esto no significa sobrediagnostico clinico; significa falsos positivos del modelo dentro de PTB-XL.",
            styles["Callout"],
        )
    )
    rows = [["Clase", "Metrica", "P1-B3", "IC95", "Lectura"]]
    for _, r in bad.iterrows():
        rows.append(
            [
                r["class"],
                r["metric"].replace("@q", ""),
                fmt(r["diff_mean"]),
                f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]",
                "B3 captura mas que P1.",
            ]
        )
    story.append(Paragraph("Limites por clase en q=0.20", styles["H2x"]))
    story.append(small_table(rows, col_widths=[0.65 * inch, 1.7 * inch, 0.75 * inch, 1.35 * inch, 2.0 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("6. Calibracion: B2 frente a B1", styles["H1x"]))
    cal_rows = [["Metrica", "Diferencia B2-B1", "IC95 entre semillas", "Interpretacion"]]
    cal_labels = {
        "Brier_macro": "Brier macro",
        "ECE_macro_10bins": "ECE macro",
        "AUROC_macro": "AUROC macro",
        "AUPRC_macro": "AUPRC macro",
    }
    for _, r in cal[cal["metric"].isin(cal_labels)].iterrows():
        interp = "Menor es mejor; negativo favorece B2." if r["metric"] in ["Brier_macro", "ECE_macro_10bins"] else "Positivo favorece B2; aqui es pequena perdida."
        cal_rows.append(
            [
                cal_labels[r["metric"]],
                fmt(r["diff_mean"]),
                f"[{fmt(r['diff_ci95_low'])}, {fmt(r['diff_ci95_high'])}]",
                interp,
            ]
        )
    story.append(small_table(cal_rows, col_widths=[1.2 * inch, 1.15 * inch, 1.5 * inch, 2.8 * inch]))
    add_image(story, FIGURES / "fig_posthoc_b2_vs_b1_calibration.png", max_h=3.3 * inch)
    story.append(
        para(
            "Lectura simple: B2, que usa likelihood/100 como etiqueta suave, muestra Brier y ECE menores que B1. Eso apunta a mejor calibracion. Sin embargo, los IC entre semillas cruzan cerca de cero, por lo que conviene presentarlo como resultado descriptivo o complementario, no como afirmacion estadistica fuerte.",
            styles["Warn"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("7. Complementariedad humano-modelo", styles["H1x"]))
    add_image(story, FIGURES / "fig_posthoc_pareto_error_vs_lowcert_q20.png", max_h=3.3 * inch)
    story.append(
        para(
            "El Pareto muestra dos objetivos distintos. En el eje X esta la captura de errores del modelo; en el eje Y, la captura de baja certeza humana. B3 queda relativamente mejor orientado a incertidumbre/error del modelo; P1/P0 quedan mas arriba porque capturan mas baja certeza humana. Esta figura ayuda a explicar que no hay una sola definicion de 'caso dificil'.",
            styles["Bodyx"],
        )
    )
    add_image(story, FIGURES / "fig_posthoc_quadrants_human_model.png", max_h=3.1 * inch)
    story.append(
        para(
            "Los cuadrantes separan casos donde la ambiguedad humana aprendida y la incertidumbre del modelo coinciden o no coinciden. Esto ayuda a defender que P1 no reemplaza a B3: aporta otra senal. Casos con alta ambiguedad humana y baja incertidumbre del modelo pueden ser interpretaciones humanas limitrofes que el modelo resuelve con confianza; casos con baja ambiguedad humana y alta incertidumbre del modelo pueden ser fallos algorítmicos.",
            styles["Bodyx"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("8. Que conviene incluir en el manuscrito", styles["H1x"]))
    include_rows = [
        ["Elemento", "Decision", "Motivo"],
        [
            "Figura de enriquecimiento",
            "Incluir como figura principal post-hoc.",
            "Es clara, significativa y muestra la ventaja real de P1.",
        ],
        [
            "Tabla q=0.10/q=0.20",
            "Incluir o resumir en texto.",
            "Traduce el resultado a carga de trabajo medica: revisar 10% o 20%.",
        ],
        [
            "Heatmap por clase",
            "Usar en presentacion o apendice.",
            "Muy visual, pero exploratorio y mixto: mejora MI/HYP FP, empeora NORM y algunas FN.",
        ],
        [
            "Calibracion B2",
            "Complemento breve.",
            "Interesante para conectar con el tema original de likelihoods, pero no resultado fuerte.",
        ],
        [
            "Quadrants/Pareto",
            "Presentacion o discusion.",
            "Explica por que ambiguedad humana y error del modelo son constructos distintos.",
        ],
    ]
    story.append(small_table(include_rows, col_widths=[1.5 * inch, 1.65 * inch, 3.4 * inch]))
    story.append(Paragraph("Frase fuerte y honesta", styles["H2x"]))
    story.append(
        para(
            "El modelo multitarea P1 mejora significativamente la identificacion y el enriquecimiento de ECG de baja certeza humana en una cola de revision, especialmente cuando el presupuesto de revision es limitado, pero esta senal no debe interpretarse como un detector general de errores del clasificador.",
            styles["Callout"],
        )
    )
    story.append(Paragraph("Texto sugerido para Resultados", styles["H2x"]))
    story.append(
        para(
            "En un analisis post-hoc de utilidad operacional, se evaluo la capacidad de los scores para enriquecer una cola de revision con ECG de baja certeza humana. Al derivar el 10% de los ECG con mayor score, P1 alcanzo precision de baja certeza de 0.425 y lift de 2.015 respecto de seleccion aleatoria, superando a B3. Con q=0.20, P1 capturo 0.369 de los casos de baja certeza frente a 0.272 de B3; la diferencia P1-B3 fue 0.097, con IC95 bootstrap por paciente [0.061, 0.133]. Este beneficio no se traslado a la captura agregada de errores, lo que respalda la interpretacion de P1 como una senal de ambiguedad humana complementaria a la incertidumbre del modelo.",
            styles["Bodyx"],
        )
    )
    story.append(Paragraph("Texto sugerido para Discusion", styles["H2x"]))
    story.append(
        para(
            "Estos hallazgos sugieren que los likelihoods diagnosticos de PTB-XL contienen informacion util para aprender una nocion de dificultad asociada a la certeza humana. La utilidad practica del enfoque estaria en priorizar revision de casos ambiguos para el anotador, no en reemplazar estrategias orientadas a detectar errores del clasificador. La ventaja por clase observada en MI/HYP debe considerarse exploratoria y requiere validacion adicional.",
            styles["Bodyx"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("9. Checklist de interpretacion", styles["H1x"]))
    checklist = [
        ["Pregunta", "Respuesta correcta"],
        [
            "P1 es mejor que B3 en todo?",
            "No. P1 es mejor para baja certeza humana, no para captura global de errores.",
        ],
        [
            "Entonces P1 sirve?",
            "Si, cuando el objetivo es armar una cola de revision enriquecida con casos ambiguos segun la certeza humana.",
        ],
        [
            "Que q conviene mostrar?",
            "q=0.10 si quieres destacar eficiencia; q=0.20 si quieres balance entre cobertura y carga de trabajo.",
        ],
        [
            "Que clases benefician a P1?",
            "Exploratoriamente, HYP error total y falsos positivos, y MI falsos positivos.",
        ],
        [
            "Que clases perjudican a P1?",
            "NORM claramente; tambien falsos negativos en MI/STTC.",
        ],
        [
            "Se puede decir que B2 calibra mejor?",
            "Como resultado descriptivo si; como afirmacion estadistica fuerte, no con estos IC.",
        ],
        [
            "Cual es el mayor riesgo de redaccion?",
            "Confundir baja certeza humana con error clinico real o afirmar superioridad clinica sin validacion externa.",
        ],
    ]
    story.append(small_table(checklist, col_widths=[2.2 * inch, 4.4 * inch]))
    story.append(Spacer(1, 10))
    story.append(
        para(
            "Conclusion final: el resultado visual y defendible es que P1 convierte los likelihoods humanos en una herramienta de priorizacion de revision. El valor metodologico esta en separar 'casos donde el modelo duda' de 'casos donde la anotacion humana fue menos cierta'.",
            styles["Callout"],
        )
    )

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT)


if __name__ == "__main__":
    main()
