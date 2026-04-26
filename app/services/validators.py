"""
Post-extraction validators.

These helpers operate on the dict produced by Gemini after
``_normalise_structured_fields`` has rewritten keys to canonical
snake_case. They never raise and they never mutate ``gemini_result``
beyond well-defined keys.

Three functions are exposed:

* :func:`validate_invoice_totals` — returns a dict describing whether
  ``subtotal + tax_amount`` equals ``total_amount`` within a small
  tolerance. Useful as a sanity gate before auto-filing.
* :func:`detect_hallucinations` — for every string field of meaningful
  length, checks that the value (or a normalised prefix) appears in the
  OCR text. Surfaces fields that may have been invented by the model.
* :func:`compute_review_required` — combines the per-field self-eval
  output with field counts and validator output to produce a single
  ``low_confidence_review_required`` boolean for downstream consumers
  (e.g. Aura's auto-filing decision gate).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List

logger = logging.getLogger("validators")

_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_TOTALS_TOLERANCE = 0.02  # 2 cents tolerance on currency totals


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip().replace(" ", "")
    if not s:
        return None
    # Accept both EU and US formats — _norm_value should already have
    # converted them, but be defensive.
    if s.count(",") and s.count(".") and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    elif s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = s.replace(",", "")
    if not _NUMERIC_RE.match(s):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def validate_invoice_totals(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check ``subtotal + tax_amount ≈ total_amount`` when all three are
    numeric. Returns a structured result so it can be stored alongside
    the document for audit purposes.
    """
    sub = _to_float(fields.get("subtotal"))
    tax = _to_float(fields.get("tax_amount"))
    total = _to_float(fields.get("total_amount"))

    if sub is None or tax is None or total is None:
        return {
            "checked": False,
            "reason": "insufficient_numeric_fields",
            "subtotal": sub,
            "tax_amount": tax,
            "total_amount": total,
        }

    expected = round(sub + tax, 2)
    delta = round(total - expected, 2)
    return {
        "checked": True,
        "totals_match": abs(delta) <= _TOTALS_TOLERANCE,
        "expected_total": expected,
        "reported_total": total,
        "delta": delta,
        "tolerance": _TOTALS_TOLERANCE,
    }


def _normalise_for_match(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s.lower())


def detect_hallucinations(
    fields: Dict[str, Any],
    ocr_text: str,
    *,
    skip_fields: Iterable[str] = ("currency", "language", "iban", "bic"),
    min_value_length: int = 3,
    max_prefix_length: int = 12,
) -> Dict[str, Any]:
    """
    Flag string fields whose value cannot be found in ``ocr_text``.

    The match is whitespace/punctuation-insensitive. We also try a
    lowercased prefix match so that values which the model cleaned up
    (e.g. trailing commas removed, capitalisation changed) still match.
    Numeric fields are matched in normalised form too.
    """
    if not ocr_text:
        return {"checked": False, "reason": "no_ocr_text", "suspicious": {}}

    haystack = _normalise_for_match(ocr_text)
    skip = {f.lower() for f in skip_fields}
    suspicious: Dict[str, str] = {}

    for key, value in (fields or {}).items():
        if key.lower() in skip:
            continue
        if value is None or isinstance(value, bool):
            continue
        sval = str(value).strip()
        if len(sval) < min_value_length:
            continue
        norm_val = _normalise_for_match(sval)
        if not norm_val:
            continue
        if norm_val in haystack:
            continue
        prefix = norm_val[:max_prefix_length]
        if len(prefix) >= min_value_length and prefix in haystack:
            continue
        suspicious[key] = sval

    return {
        "checked": True,
        "suspicious": suspicious,
        "suspicious_count": len(suspicious),
    }


def compute_review_required(
    *,
    overall_confidence: float | None,
    field_confidence: Dict[str, float] | None,
    structured_fields: Dict[str, Any],
    totals_validation: Dict[str, Any] | None,
    hallucination_check: Dict[str, Any] | None,
    low_confidence_threshold: float = 0.7,
    min_populated_fields: int = 3,
) -> Dict[str, Any]:
    """
    Combine multiple signals into one ``low_confidence_review_required``
    decision plus a list of human-readable reasons.
    """
    reasons: List[str] = []

    populated = sum(1 for v in (structured_fields or {}).values() if v not in (None, "", [], {}))
    if populated < min_populated_fields:
        reasons.append(f"only_{populated}_populated_fields")

    if overall_confidence is not None and overall_confidence < low_confidence_threshold:
        reasons.append(f"overall_confidence_{overall_confidence:.2f}_below_{low_confidence_threshold:.2f}")

    if field_confidence:
        weak = [k for k, v in field_confidence.items() if isinstance(v, (int, float)) and v < low_confidence_threshold]
        if weak:
            reasons.append(f"weak_fields={','.join(weak[:8])}")

    if totals_validation and totals_validation.get("checked") and not totals_validation.get("totals_match", True):
        reasons.append(f"totals_mismatch_delta={totals_validation.get('delta')}")

    if hallucination_check and hallucination_check.get("suspicious_count", 0) > 0:
        susp = list((hallucination_check.get("suspicious") or {}).keys())
        reasons.append(f"hallucinated_fields={','.join(susp[:8])}")

    return {
        "review_required": bool(reasons),
        "reasons": reasons,
        "populated_fields": populated,
    }


__all__ = [
    "validate_invoice_totals",
    "detect_hallucinations",
    "compute_review_required",
]
