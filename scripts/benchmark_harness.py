"""
Benchmark harness for Pro-tier extraction quality.

Usage:
  python scripts/benchmark_harness.py
  python scripts/benchmark_harness.py --docs data/benchmarks/docs --labels data/benchmarks/labels

Label JSON format (one file per document):
{
  "file": "invoice_001.pdf",
  "expected": {
    "classification": "invoice",
    "fields": {
      "invoice_number": "INV-2026-001",
      "invoice_date": "2026-03-10",
      "total_amount": "1299.99",
      "currency": "EUR"
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pdf2image import convert_from_path
from PIL import Image

from app.core.pipeline import _normalise_structured_fields  # reuse production normalization
from app.services.gemini_service import analyse_document, extract_text_from_pdf
from config.settings import GEMINI_IMAGE_DPI, MAX_PAGES, POPPLER_PATH


def _norm_value(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = " ".join(s.split())
    return s


def _load_label(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_pages_and_text(file_path: Path) -> Tuple[List[Image.Image], str]:
    ext = file_path.suffix.lower()
    is_image = ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    if is_image:
        img = Image.open(str(file_path))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return [img], ""

    extracted_text = extract_text_from_pdf(str(file_path))
    convert_kwargs: Dict[str, Any] = {
        "dpi": GEMINI_IMAGE_DPI,
        "last_page": MAX_PAGES,
        "fmt": "jpeg",
    }
    if POPPLER_PATH and os.path.exists(POPPLER_PATH):
        convert_kwargs["poppler_path"] = POPPLER_PATH

    pages = convert_from_path(str(file_path), **convert_kwargs)
    return pages, extracted_text


def _score_doc(result: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    expected_class = _norm_value(expected.get("classification", ""))
    predicted_class = _norm_value(result.get("classification", {}).get("type", ""))
    class_ok = int(expected_class == predicted_class) if expected_class else None

    expected_fields = expected.get("fields", {}) or {}
    predicted_fields = result.get("structured_fields", {}) or {}

    field_rows = []
    matched = 0
    for key, exp_val in expected_fields.items():
        got_val = predicted_fields.get(key)
        ok = _norm_value(exp_val) == _norm_value(got_val)
        matched += int(ok)
        field_rows.append(
            {
                "field": key,
                "expected": exp_val,
                "predicted": got_val,
                "match": ok,
            }
        )

    total_fields = len(expected_fields)
    field_accuracy = (matched / total_fields) if total_fields else None

    return {
        "classification": {
            "expected": expected_class,
            "predicted": predicted_class,
            "match": class_ok,
        },
        "fields": {
            "total": total_fields,
            "matched": matched,
            "accuracy": field_accuracy,
            "details": field_rows,
        },
    }


def run_benchmark(docs_dir: Path, labels_dir: Path, reports_dir: Path) -> Path:
    labels = sorted(labels_dir.glob("*.json"))
    if not labels:
        raise RuntimeError(f"No label files found in {labels_dir}")

    results = []
    class_hits = 0
    class_total = 0
    field_hits = 0
    field_total = 0

    for lbl_path in labels:
        label = _load_label(lbl_path)
        rel_file = label.get("file")
        expected = label.get("expected", {})
        if not rel_file:
            results.append(
                {
                    "label_file": lbl_path.name,
                    "status": "error",
                    "error": "missing 'file' in label",
                }
            )
            continue

        doc_path = docs_dir / rel_file
        if not doc_path.exists():
            results.append(
                {
                    "label_file": lbl_path.name,
                    "file": rel_file,
                    "status": "error",
                    "error": f"document file not found: {doc_path}",
                }
            )
            continue

        try:
            pages, extracted_text = _load_pages_and_text(doc_path)
            model_result = analyse_document(pages, extracted_text)
            _normalise_structured_fields(model_result)

            score = _score_doc(model_result, expected)

            cls_match = score["classification"]["match"]
            if cls_match is not None:
                class_total += 1
                class_hits += int(cls_match)

            field_total += int(score["fields"]["total"])
            field_hits += int(score["fields"]["matched"])

            results.append(
                {
                    "label_file": lbl_path.name,
                    "file": rel_file,
                    "status": "ok",
                    "score": score,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "label_file": lbl_path.name,
                    "file": rel_file,
                    "status": "error",
                    "error": str(exc),
                }
            )

    summary = {
        "docs_total": len(results),
        "docs_ok": sum(1 for r in results if r.get("status") == "ok"),
        "docs_error": sum(1 for r in results if r.get("status") == "error"),
        "classification_accuracy": (class_hits / class_total) if class_total else None,
        "field_accuracy": (field_hits / field_total) if field_total else None,
        "class_hits": class_hits,
        "class_total": class_total,
        "field_hits": field_hits,
        "field_total": field_total,
    }

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "docs_dir": str(docs_dir),
        "labels_dir": str(labels_dir),
        "summary": summary,
        "results": results,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"benchmark-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pro-tier extraction benchmark against labeled docs")
    parser.add_argument("--docs", default="data/benchmarks/docs", help="Directory containing benchmark documents")
    parser.add_argument("--labels", default="data/benchmarks/labels", help="Directory containing label JSON files")
    parser.add_argument("--reports", default="data/benchmarks/reports", help="Output directory for benchmark reports")
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    labels_dir = Path(args.labels)
    reports_dir = Path(args.reports)

    out = run_benchmark(docs_dir, labels_dir, reports_dir)
    print(f"Benchmark complete. Report: {out}")


if __name__ == "__main__":
    main()
