from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[0]
MANIFEST_DIR = ROOT / "06_RESPALDO" / "manifests"
EXPECTED_TITLE = (
    "Clasificacion Selectiva de ECG Guiada por Certeza Humana: Uso de los "
    "Likelihoods Diagnosticos de PTB-XL para Derivacion a Revision Medica"
)
RUNS = [
    "hcg_selective_ecg_full_20260712_131520",
    "hcg_selective_ecg_full_seed13",
    "hcg_selective_ecg_full_seed123",
    "hcg_selective_ecg_full_seed2024",
    "hcg_selective_ecg_full_seed7",
]
MODELS = [
    "B1_HARD_DIAG",
    "B2_SOFT_DIRECT",
    "P0_MTL_UNBALANCED",
    "P1_MTL_BALANCED_PROPOSED",
]


def normalized(text: str) -> str:
    table = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "Á": "A",
            "É": "E",
            "Í": "I",
            "Ó": "O",
            "Ú": "U",
            "ñ": "n",
            "Ñ": "N",
        }
    )
    return " ".join(text.translate(table).split())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_summary(path: Path) -> dict:
    notebook = read_json(path)
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    error_outputs = []
    for index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if output.get("output_type") == "error" or "Traceback" in text:
                error_outputs.append(index)
    return {
        "cells": len(notebook.get("cells", [])),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
        "output_blocks": sum(len(cell.get("outputs", [])) for cell in code_cells),
        "error_cells": error_outputs,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> dict:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    tex_path = ROOT / "01_MANUSCRITO" / "OVERLEAF" / "main.tex"
    pdf_path = ROOT / "01_MANUSCRITO" / "PDF" / "manuscrito_clasificacion_selectiva_certeza_humana.pdf"
    tex = tex_path.read_text(encoding="utf-8")
    author_line = next((line for line in tex.splitlines() if line.lstrip().startswith("\\author{")), "")
    check("titulo", EXPECTED_TITLE in normalized(tex), EXPECTED_TITLE)
    check(
        "autoria",
        "Ronald Marca" in author_line
        and "Joaquin Mendoza" in normalized(author_line)
        and "Alexandra" not in author_line
        and "Muñoz" not in author_line,
        author_line,
    )
    check("pdf", pdf_path.exists() and pdf_path.stat().st_size > 100_000, str(pdf_path.relative_to(ROOT)))

    executed = ROOT / "02_CODIGO" / "notebooks" / "01_ENTRENAMIENTO_ANALISIS_FULL_EJECUTADO.ipynb"
    portable = ROOT / "02_CODIGO" / "notebooks" / "01_ENTRENAMIENTO_ANALISIS_FULL_PORTABLE_LIMPIO.ipynb"
    executed_summary = notebook_summary(executed)
    portable_notebook = read_json(portable)
    portable_source = "\n".join(
        "".join(cell.get("source", [])) for cell in portable_notebook.get("cells", [])
    )
    check(
        "notebook_full_ejecutado",
        executed_summary["executed_code_cells"] == executed_summary["code_cells"]
        and not executed_summary["error_cells"],
        json.dumps(executed_summary, ensure_ascii=False),
    )
    check(
        "notebook_portable",
        "C:/Users/" not in portable_source and 'PROJECT_ROOT_OVERRIDE = ""' in portable_source,
        str(portable.relative_to(ROOT)),
    )

    run_details = []
    for run in RUNS:
        run_dir = ROOT / "03_RESULTADOS" / "runs" / run
        summaries = list(run_dir.rglob("RUN_SUMMARY.json"))
        run_details.append({"run": run, "exists": run_dir.exists(), "run_summaries": len(summaries)})
    check(
        "cinco_corridas_full",
        all(item["exists"] and item["run_summaries"] > 0 for item in run_details),
        json.dumps(run_details),
    )

    model_rows = []
    for run in RUNS:
        for model in MODELS:
            model_dir = ROOT / "04_MODELOS" / run / model
            checkpoints = sorted(model_dir.rglob("best.pt"))
            configs = sorted(model_dir.rglob("config.json"))
            model_rows.append(
                {
                    "run": run,
                    "model": model,
                    "checkpoint": checkpoints[0].relative_to(ROOT).as_posix() if checkpoints else "",
                    "config": configs[0].relative_to(ROOT).as_posix() if configs else "",
                    "checkpoint_size_bytes": checkpoints[0].stat().st_size if checkpoints else 0,
                }
            )
    check("veinte_checkpoints", sum(bool(row["checkpoint"]) for row in model_rows) == 20, "20 modelos esperados")

    csv_files = sorted((ROOT / "03_RESULTADOS").rglob("*.csv"))
    bad_csv = []
    for path in csv_files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                next(csv.reader(handle), None)
        except Exception as exc:
            bad_csv.append(f"{path.relative_to(ROOT)}: {exc}")
    check("csv_legibles", not bad_csv, f"{len(csv_files)} CSV; errores={bad_csv}")

    json_files = sorted((ROOT / "03_RESULTADOS").rglob("*.json"))
    bad_json = []
    for path in json_files:
        try:
            read_json(path)
        except Exception as exc:
            bad_json.append(f"{path.relative_to(ROOT)}: {exc}")
    check("json_legibles", not bad_json, f"{len(json_files)} JSON; errores={bad_json}")

    manuscript_figures = sorted((ROOT / "05_FIGURAS" / "manuscrito").glob("*.png"))
    check("seis_figuras_manuscrito", len(manuscript_figures) == 6, f"PNG={len(manuscript_figures)}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with (MANIFEST_DIR / "model_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=model_rows[0].keys())
        writer.writeheader()
        writer.writerows(model_rows)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(ROOT),
        "title": EXPECTED_TITLE,
        "authors": ["Ronald Marca", "Joaquin Mendoza"],
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "counts": {
            "runs": len(RUNS),
            "models": len(model_rows),
            "csv_files": len(csv_files),
            "json_files": len(json_files),
            "manuscript_png_figures": len(manuscript_figures),
        },
    }
    (MANIFEST_DIR / "VALIDATION_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def write_manifest() -> int:
    manifest_path = MANIFEST_DIR / "sha256_manifest.csv"
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and path.suffix.lower() != ".zip"
        and "__pycache__" not in path.parts
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix().lower()):
            writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)])
    return len(files)


if __name__ == "__main__":
    validation = validate()
    manifest_count = write_manifest()
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Archivos incluidos en el manifiesto SHA-256: {manifest_count}")
    raise SystemExit(0 if validation["passed"] else 1)
