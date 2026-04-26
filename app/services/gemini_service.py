"""
Gemini Service
Single entry point for all AI processing in the Pro tier.

Supports:
    - Adaptive retries with token fallback
    - Multi-batch processing for long documents
    - Supplemental native-text context for digital PDFs
"""

import base64
import concurrent.futures
import io
import json
import logging
import random
import re
import threading
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types
from PIL import Image

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_BATCH_PAGE_SIZE,
    GEMINI_RETRY_ATTEMPTS,
    GEMINI_RETRY_BACKOFF_MS,
    GEMINI_PROVIDER_TIMEOUT_SEC,
    GEMINI_CIRCUIT_BREAKER_THRESHOLD,
    GEMINI_CIRCUIT_BREAKER_COOLDOWN_SEC,
    GEMINI_SCHEMA_REPAIR_ATTEMPTS,
    GEMINI_FIELD_SELF_EVAL_ENABLED,
    GEMINI_FIELD_SELF_EVAL_MAX_FIELDS,
    GEMINI_FIELD_SELF_EVAL_MAX_CONTEXT_CHARS,
    GEMINI_SUPPLEMENTAL_TEXT_MAX_CHARS,
    GEMINI_JPEG_QUALITY,
    GEMINI_COST_INPUT_PER_1K,
    GEMINI_COST_OUTPUT_PER_1K,
)

logger = logging.getLogger("gemini_service")

# ── Gemini client (lazy init) ─────────────────────────────
_client: genai.Client | None = None
_circuit_failures = 0
_circuit_open_until = 0.0

# ── Per-process usage accumulator ────────────────────────────
# Tracks the most recent call's token counts so the pipeline can attach a
# per-document cost summary without threading the response through every helper.
_usage_lock = threading.Lock()
_last_usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}


def _record_usage(usage: Any) -> None:
    """Add the token counts from a Gemini ``usage_metadata`` payload to the running totals."""
    if usage is None:
        return
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    tot_tok = int(getattr(usage, "total_token_count", 0) or in_tok + out_tok)
    with _usage_lock:
        _last_usage["input_tokens"] += in_tok
        _last_usage["output_tokens"] += out_tok
        _last_usage["total_tokens"] += tot_tok
        _last_usage["calls"] += 1


def reset_usage() -> None:
    """Clear the accumulator. Call once at the start of every pipeline run."""
    with _usage_lock:
        _last_usage["input_tokens"] = 0
        _last_usage["output_tokens"] = 0
        _last_usage["total_tokens"] = 0
        _last_usage["calls"] = 0


def get_usage_summary() -> Dict[str, Any]:
    """Return the running token totals plus an estimated USD cost."""
    with _usage_lock:
        snap = dict(_last_usage)
    cost = round(
        (snap["input_tokens"] / 1000.0) * GEMINI_COST_INPUT_PER_1K
        + (snap["output_tokens"] / 1000.0) * GEMINI_COST_OUTPUT_PER_1K,
        6,
    )
    snap["estimated_usd"] = cost
    snap["model"] = GEMINI_MODEL
    return snap


def _ensure_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _circuit_is_open() -> bool:
    return time.time() < _circuit_open_until


def _circuit_before_call() -> None:
    if _circuit_is_open():
        remaining = max(0, int(_circuit_open_until - time.time()))
        raise RuntimeError(
            f"Gemini circuit breaker open; retry after ~{remaining}s"
        )


def _circuit_record_success() -> None:
    global _circuit_failures, _circuit_open_until
    _circuit_failures = 0
    _circuit_open_until = 0.0


def _circuit_record_failure() -> None:
    global _circuit_failures, _circuit_open_until
    _circuit_failures += 1
    if _circuit_failures >= max(1, int(GEMINI_CIRCUIT_BREAKER_THRESHOLD)):
        cooldown = max(5, int(GEMINI_CIRCUIT_BREAKER_COOLDOWN_SEC))
        _circuit_open_until = time.time() + cooldown
        logger.error(
            f"Gemini circuit breaker OPEN for {cooldown}s after {_circuit_failures} failures"
        )


def _call_with_timeout(client: genai.Client, parts: List[Any], tokens: int) -> str:
    def _do_call() -> str:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=tokens,
            ),
        )
        # Capture token usage for cost tracking before doing anything else;
        # we still want the metric even if the response body is empty.
        try:
            _record_usage(getattr(resp, "usage_metadata", None))
        except Exception:  # pragma: no cover - never fail the call on telemetry
            pass
        raw_text = (resp.text or "").strip()
        if not raw_text:
            raise ValueError("Empty response text from Gemini")
        return raw_text

    timeout_s = max(5, int(GEMINI_PROVIDER_TIMEOUT_SEC))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_do_call)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise TimeoutError(f"Gemini provider timeout after {timeout_s}s") from exc


# ── Prompt ────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert document analyst with deep knowledge of business, legal, and administrative documents in all languages, especially German.

Analyze the provided document pages and return ONLY a valid JSON object — no markdown fences, no extra text before or after.

The JSON must have exactly these keys:

{
  "ocr_text": "<full extracted text of all pages combined>",
  "language": "<ISO 639-1 code, e.g. de, en, fr, ar>",
  "handwriting_detected": <true|false>,
  "page_quality": <0.0-1.0 estimate of scan quality>,

  "classification": {
    "type": "<invoice|credit_note|debit_note|payment_voucher|remittance_advice|quotation|purchase_order|receipt|delivery_note|packing_slip|bill_of_lading|contract|agreement|non_disclosure_agreement|memorandum_of_understanding|business_letter|letter|memo|proposal|tender_document|bank_statement|financial_statement|tax_document|certificate|license|permit|form|report|statement|resume|job_application|employment_contract|id_document|medical_record|other>",
    "confidence": <0.0-1.0>,
    "reasoning": "<one sentence>"
  },

  "structured_fields": {
    "<field_name>": "<value or null>"
  },

IMPORTANT — structured_fields MUST use these EXACT snake_case keys (never spaces, never camelCase, never uppercase):

    invoice              → invoice_number, invoice_date, due_date, vendor_name, customer_name,
                                                    subtotal, tax_amount, total_amount, currency, payment_terms,
                                                    bank_account, iban, bic
    purchase_order       → po_number, order_date, vendor_name, customer_name,
                                                    delivery_date, total_amount, currency, payment_terms
    receipt              → receipt_number, date, merchant_name, total_amount, currency,
                                                    payment_method
    delivery_note        → delivery_note_number, delivery_date, recipient_name, sender_name,
                                                    po_number
    contract / agreement → contract_number, contract_date, party_a, party_b,
                                                    start_date, end_date, contract_value, currency
    bank_statement       → account_number, statement_period, opening_balance, closing_balance,
                                                    currency, iban, bic
    letter / business_letter → date, sender_name, sender_company, recipient_name, subject
    credit_note          → credit_note_number, date, vendor_name, total_amount, currency
    debit_note           → debit_note_number, date, vendor_name, total_amount, currency
    payment_voucher      → voucher_number, date, payer_name, payee_name, total_amount, currency
    remittance_advice    → remittance_number, date, payer_name, payee_name, total_amount, currency,
                                                    payment_method, invoice_numbers (list of invoices being paid)
    quotation            → quotation_number, date, customer_name, total_amount, currency,
                                                    valid_until
    tax_document         → tax_id, tax_year, tax_period, total_tax, currency

Do NOT invent variant key names like "purchase_order_number", "invoice_no", "InvoiceNumber",
"order number". Always use the canonical snake_case keys above. If a field is not present,
use null.

  "entities": {
    "companies":     ["<name>"],
    "people":        ["<full name>"],
    "dates":         ["<YYYY-MM-DD or original string>"],
    "amounts":       ["<numeric string with currency if available>"],
    "addresses":     ["<full address string>"],
    "emails":        ["<email>"],
    "phone_numbers": ["<phone>"]
  },

  "tables": [
    {
      "title":   "<table title or 'Table N'>",
      "page":    <1-based page number>,
      "headers": ["<col1>", "<col2>"],
      "rows":    [["<cell>", "<cell>"]],
      "bbox":    {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>}
    }
  ],

  "signature": {
    "found":          <true|false>,
    "signer_name":    "<full name or null>",
    "signer_title":   "<job title or null>",
    "signer_company": "<company or null>",
    "page":           <1-based page number or null>,
    "position":       "<bottom-right|bottom-left|center|other or null>",
    "bbox":           {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>}
  },

  "stamps": [
    {
      "issuing_authority": "<authority name or null>",
      "stamp_type":        "<official|company|received|approved|other>",
      "text_on_stamp":     "<visible stamp text or null>",
      "date_on_stamp":     "<YYYY-MM-DD or null>",
      "page":              <1-based page number>,
      "position":          "<bottom-left|bottom-right|top-right|other>",
      "bbox":              {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>}
    }
  ],

  "summary": "<one concise paragraph summarising the document>"
}

Rules:
- structured_fields must contain ALL relevant fields for the document type (invoice_number, dates, amounts, parties, payment terms, bank details, etc.)
- bbox values are fractions of the page dimensions (0.0 to 1.0)
- If a field is not present, use null — do not omit the key
- tables must include ALL tables, including borderless/implicit ones
- stamps and signatures arrays may be empty ([]) if none found
- Return the raw JSON only — no ```json fences

Handwriting rules:
- If the document is fully or partially handwritten, read every handwritten field carefully.
- If no explicit invoice number label is present, look for ORDER NO., ORDER #, REF, REFERENCE, JOB NO., FILE NO. and use that value as invoice_number.
- For handwritten dates, normalize to YYYY-MM-DD.
- For handwritten amounts, strip currency symbols and commas — return digits only (e.g. "281.25" not "$281.25").
- Set handwriting_detected = true whenever any handwritten text is found.

Foreign-language rules:
- The document may be in any language (Arabic, German, French, Spanish, Chinese, Turkish, etc.).
- ALWAYS use the canonical English snake_case keys listed above — NEVER translate or transliterate them.
- Extract field VALUES as they appear in the source language but normalize dates to YYYY-MM-DD and amounts to digit strings.
- For classification.type, ALWAYS use the English type labels (invoice, contract, receipt, etc.) regardless of document language.
- For language, return the ISO 639-1 code (de, ar, fr, es, zh, tr, …).
"""


# ── Image helpers ─────────────────────────────────────────

def _pil_to_part(image: Image.Image) -> types.Part:
    """Convert a PIL image to a Gemini Part, applying optional preprocessing."""
    try:
        from config.settings import (
            IMAGE_PREPROCESS_ENABLED,
            IMAGE_PREPROCESS_AUTO_ROTATE,
            IMAGE_PREPROCESS_DESKEW,
            IMAGE_PREPROCESS_CONTRAST,
        )
        if IMAGE_PREPROCESS_ENABLED:
            from app.services.image_preprocess import preprocess_page
            image = preprocess_page(
                image,
                auto_rotate=IMAGE_PREPROCESS_AUTO_ROTATE,
                deskew=IMAGE_PREPROCESS_DESKEW,
                contrast=IMAGE_PREPROCESS_CONTRAST,
            )
    except Exception as exc:  # pragma: no cover - never fail the call on preprocessing
        logger.debug(f"image preprocessing skipped: {exc}")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(GEMINI_JPEG_QUALITY))
    return types.Part.from_bytes(
        data=buf.getvalue(),
        mime_type="image/jpeg",
    )


def _crop_snippet(image: Image.Image, bbox: Dict[str, float]) -> str | None:
    """
    Crop a region from a PIL image using normalised bbox and return base64 PNG.
    bbox keys: x, y, w, h — all in 0-1 range relative to image size.
    Returns None if bbox is invalid.
    """
    if not bbox or not all(k in bbox for k in ("x", "y", "w", "h")):
        return None
    try:
        iw, ih = image.size
        x1 = int(bbox["x"] * iw)
        y1 = int(bbox["y"] * ih)
        x2 = int((bbox["x"] + bbox["w"]) * iw)
        y2 = int((bbox["y"] + bbox["h"]) * ih)
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"Snippet crop failed: {e}")
        return None


# ── JSON parser ───────────────────────────────────────────

class _TruncatedJSONError(ValueError):
    """Raised when Gemini returned a JSON string that was cut off mid-token."""


def _salvage_truncated_json(text: str) -> Dict[str, Any] | None:
    """
    Best-effort recovery of a JSON object that was cut mid-string.

    Walks the text once tracking string/escape state and the open-container
    stack. At EOF we close any open string then close every open array /
    object. Trailing partial keys or commas are stripped before re-parsing.
    Returns ``None`` if salvage still fails.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in body:
        out.append(ch)
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    if in_string:
        if escape:
            out.pop()
        out.append('"')

    repaired = "".join(out)
    repaired = re.sub(r",\s*$", "", repaired)
    while stack:
        repaired = re.sub(r'[,\s]*(?:"[^"]*"\s*:\s*)?$', "", repaired)
        repaired += stack.pop()
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Extract JSON from a Gemini response, tolerating markdown fences and
    mid-string truncation caused by the model's token budget.

    On clean parse: returns the dict.
    On recoverable truncation: returns the salvaged dict (warning logged).
    On total failure: raises ``_TruncatedJSONError`` so callers can choose
    to retry with a larger token budget.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Trailing junk after the outermost object?
        start = text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            esc = False
            for i, ch in enumerate(text[start:], start):
                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        salvaged = _salvage_truncated_json(text)
        if salvaged is not None:
            logger.warning(
                f"Gemini JSON was truncated; salvaged a partial object "
                f"(len={len(text)} chars, error={e})"
            )
            return salvaged
        raise _TruncatedJSONError(
            f"Could not parse Gemini JSON response: {e}\nRaw (first 500): {raw[:500]}"
        )


def _default_result() -> Dict[str, Any]:
    return {
        "ocr_text": "",
        "language": "unknown",
        "handwriting_detected": False,
        "page_quality": 0.0,
        "classification": {"type": "other", "confidence": 0.0, "reasoning": ""},
        "structured_fields": {},
        "entities": {
            "companies": [],
            "people": [],
            "dates": [],
            "amounts": [],
            "addresses": [],
            "emails": [],
            "phone_numbers": [],
        },
        "tables": [],
        "signature": {
            "found": False,
            "signer_name": None,
            "signer_title": None,
            "signer_company": None,
            "page": None,
            "position": None,
            "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
        },
        "stamps": [],
        "summary": "",
    }


def _ensure_result_shape(result: Dict[str, Any]) -> Dict[str, Any]:
    out = _default_result()
    out.update(result or {})
    if not isinstance(out.get("classification"), dict):
        out["classification"] = _default_result()["classification"]
    if not isinstance(out.get("structured_fields"), dict):
        out["structured_fields"] = {}
    if not isinstance(out.get("entities"), dict):
        out["entities"] = _default_result()["entities"]
    for key in ("companies", "people", "dates", "amounts", "addresses", "emails", "phone_numbers"):
        val = out["entities"].get(key)
        out["entities"][key] = val if isinstance(val, list) else []
    if not isinstance(out.get("tables"), list):
        out["tables"] = []
    if not isinstance(out.get("stamps"), list):
        out["stamps"] = []
    if not isinstance(out.get("signature"), dict):
        out["signature"] = _default_result()["signature"]
    return out


def _validate_result_schema(result: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(result, dict):
        return ["root must be a JSON object"]

    expected = {
        "ocr_text": str,
        "language": str,
        "handwriting_detected": bool,
        "page_quality": (int, float),
        "classification": dict,
        "structured_fields": dict,
        "entities": dict,
        "tables": list,
        "signature": dict,
        "stamps": list,
        "summary": str,
    }

    for key, typ in expected.items():
        if key not in result:
            errors.append(f"missing key: {key}")
            continue
        if not isinstance(result[key], typ):
            errors.append(f"invalid type for {key}: expected {typ}, got {type(result[key]).__name__}")

    cls = result.get("classification")
    if isinstance(cls, dict):
        for k in ("type", "confidence", "reasoning"):
            if k not in cls:
                errors.append(f"classification missing key: {k}")

    sig = result.get("signature")
    if isinstance(sig, dict):
        for k in ("found", "page", "bbox"):
            if k not in sig:
                errors.append(f"signature missing key: {k}")

    ent = result.get("entities")
    if isinstance(ent, dict):
        for k in ("companies", "people", "dates", "amounts", "addresses", "emails", "phone_numbers"):
            if k not in ent:
                errors.append(f"entities missing key: {k}")
            elif not isinstance(ent.get(k), list):
                errors.append(f"entities.{k} must be a list")

    return errors


def _repair_json_with_model(
    client: genai.Client,
    candidate_json: str,
    errors: List[str],
    batch_label: str,
) -> Dict[str, Any]:
    prompt = (
        "You are a strict JSON repair assistant.\n"
        "Fix the provided document-analysis JSON so it matches the required schema.\n"
        "Return ONLY a valid JSON object with all required keys and correct types.\n\n"
        f"Validation errors:\n- " + "\n- ".join(errors[:50]) + "\n\n"
        "JSON to repair:\n"
        f"{candidate_json[:200000]}"
    )
    raw = _call_with_timeout(client, [prompt], max(1024, int(GEMINI_MAX_OUTPUT_TOKENS * 0.35)))
    logger.warning(f"Schema repair attempted for {batch_label}")
    parsed = _parse_json_response(raw)
    return _ensure_result_shape(parsed)


def _parse_validate_repair(
    client: genai.Client,
    raw_text: str,
    batch_label: str,
) -> Dict[str, Any]:
    candidate = _ensure_result_shape(_parse_json_response(raw_text))
    errs = _validate_result_schema(candidate)
    if not errs:
        return candidate

    attempts = max(0, int(GEMINI_SCHEMA_REPAIR_ATTEMPTS))
    candidate_raw = raw_text
    for _ in range(attempts):
        try:
            candidate = _repair_json_with_model(client, candidate_raw, errs, batch_label)
            errs = _validate_result_schema(candidate)
            if not errs:
                return candidate
            candidate_raw = json.dumps(candidate, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Schema repair failed for {batch_label}: {exc}")

    logger.warning(f"Schema still invalid after repair for {batch_label}; applying safe defaults")
    return _ensure_result_shape(candidate)


def _build_parts(pages: List[Image.Image], extracted_text: str = "", page_offset: int = 0) -> List[Any]:
    parts: List[Any] = [_SYSTEM_PROMPT]
    for i, page in enumerate(pages):
        parts.append(f"\n--- Page {page_offset + i + 1} ---\n")
        parts.append(_pil_to_part(page))

    if extracted_text:
        supplemental = extracted_text[:GEMINI_SUPPLEMENTAL_TEXT_MAX_CHARS]
        parts.append("\n--- Supplemental Native PDF Text ---\n")
        parts.append(supplemental)

    return parts


def _token_candidates() -> List[int]:
    base = max(1024, int(GEMINI_MAX_OUTPUT_TOKENS))
    cands = [base]
    for pct in (0.75, 0.5, 0.35, 0.2):
        cands.append(max(1024, int(base * pct)))
    # Keep order, remove duplicates
    ordered = []
    for v in cands:
        if v not in ordered:
            ordered.append(v)
    return ordered


def _generate_with_retries(client: genai.Client, parts: List[Any], batch_label: str) -> str:
    token_candidates = _token_candidates()
    attempts = max(1, int(GEMINI_RETRY_ATTEMPTS))
    backoff_ms = max(100, int(GEMINI_RETRY_BACKOFF_MS))
    last_err: Exception | None = None

    for tokens in token_candidates:
        for attempt in range(1, attempts + 1):
            try:
                _circuit_before_call()
                logger.info(
                    f"Gemini call {batch_label}: attempt={attempt}/{attempts}, max_output_tokens={tokens}"
                )
                raw_text = _call_with_timeout(client, parts, tokens)
                _circuit_record_success()
                return raw_text
            except Exception as exc:
                last_err = exc
                _circuit_record_failure()
                logger.warning(
                    f"Gemini call failed {batch_label}: attempt={attempt}/{attempts}, "
                    f"tokens={tokens}, error={exc}"
                )
                if attempt < attempts:
                    # Exponential backoff with jitter (±50%) to avoid thundering
                    # herd when multiple workers retry the same outage in lockstep.
                    base = (backoff_ms * (2 ** (attempt - 1))) / 1000.0
                    jitter = random.uniform(-0.5, 0.5) * base
                    sleep_s = max(0.1, min(base + jitter, 8.0))
                    time.sleep(sleep_s)

    raise RuntimeError(f"Gemini failed after retries and token fallback: {last_err}")


def _safe_float(v: Any, default: float = 0.5) -> float:
    try:
        out = float(v)
    except Exception:
        out = default
    return max(0.0, min(1.0, out))


def _self_evaluate_field_confidence(
    client: genai.Client,
    structured_fields: Dict[str, Any],
    ocr_text: str,
) -> Dict[str, Any]:
    if not GEMINI_FIELD_SELF_EVAL_ENABLED:
        return {
            "overall_extraction_confidence": None,
            "field_confidence": {},
            "notes": "self_eval_disabled",
        }

    items = list((structured_fields or {}).items())
    items = [(k, v) for k, v in items if v not in (None, "", [], {})]
    items = items[: max(1, int(GEMINI_FIELD_SELF_EVAL_MAX_FIELDS))]
    if not items:
        return {
            "overall_extraction_confidence": 0.0,
            "field_confidence": {},
            "notes": "no_populated_fields",
        }

    prompt = (
        "You are evaluating extraction confidence for document fields.\n"
        "Given OCR/context text and extracted fields, output ONLY valid JSON:\n"
        "{\n"
        "  \"overall_extraction_confidence\": <0..1>,\n"
        "  \"field_confidence\": {\"field_name\": <0..1>},\n"
        "  \"notes\": \"short rationale\"\n"
        "}\n"
        "Be conservative. Lower confidence if OCR evidence is weak or ambiguous.\n\n"
        f"OCR/context (truncated):\n{(ocr_text or '')[:int(GEMINI_FIELD_SELF_EVAL_MAX_CONTEXT_CHARS)]}\n\n"
        f"Extracted fields:\n{json.dumps(dict(items), ensure_ascii=False)}"
    )

    try:
        raw = _generate_with_retries(client, [prompt], "field-self-eval")
        parsed = _parse_json_response(raw)
        field_conf = parsed.get("field_confidence", {}) if isinstance(parsed, dict) else {}
        if not isinstance(field_conf, dict):
            field_conf = {}

        out_map = {}
        for k, _ in items:
            out_map[k] = _safe_float(field_conf.get(k, 0.5), 0.5)

        overall = _safe_float(parsed.get("overall_extraction_confidence", 0.5), 0.5)
        notes = parsed.get("notes", "") if isinstance(parsed, dict) else ""

        return {
            "overall_extraction_confidence": overall,
            "field_confidence": out_map,
            "notes": str(notes)[:500],
        }
    except Exception as exc:
        logger.warning(f"Field self-eval failed: {exc}")
        fallback = {k: 0.5 for k, _ in items}
        return {
            "overall_extraction_confidence": 0.5,
            "field_confidence": fallback,
            "notes": "self_eval_fallback",
        }


def _offset_page_numbers(result: Dict[str, Any], offset: int) -> Dict[str, Any]:
    if offset <= 0:
        return result

    sig = result.get("signature") or {}
    if sig.get("page") is not None:
        try:
            sig["page"] = int(sig["page"]) + offset
        except Exception:
            pass

    for t in result.get("tables", []) or []:
        if isinstance(t, dict) and t.get("page") is not None:
            try:
                t["page"] = int(t["page"]) + offset
            except Exception:
                pass

    for s in result.get("stamps", []) or []:
        if isinstance(s, dict) and s.get("page") is not None:
            try:
                s["page"] = int(s["page"]) + offset
            except Exception:
                pass

    return result


def _merge_unique(seq_a: List[Any], seq_b: List[Any], cap: int = 500) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in (seq_a or []) + (seq_b or []):
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= cap:
            break
    return out


def _merge_results(base: Dict[str, Any], nxt: Dict[str, Any]) -> Dict[str, Any]:
    a = _ensure_result_shape(base)
    b = _ensure_result_shape(nxt)

    # OCR text and summary
    if b.get("ocr_text"):
        a["ocr_text"] = (a.get("ocr_text", "") + "\n\n" + b.get("ocr_text", "")).strip()
    if b.get("summary"):
        if not a.get("summary"):
            a["summary"] = b["summary"]
        else:
            a["summary"] = (a["summary"] + " " + b["summary"]).strip()

    # Classification: keep the higher-confidence one
    ca = a.get("classification", {})
    cb = b.get("classification", {})
    if float(cb.get("confidence", 0.0) or 0.0) > float(ca.get("confidence", 0.0) or 0.0):
        a["classification"] = cb

    # Language and page quality
    if a.get("language") in (None, "", "unknown") and b.get("language"):
        a["language"] = b.get("language")
    a["handwriting_detected"] = bool(a.get("handwriting_detected") or b.get("handwriting_detected"))
    a["page_quality"] = max(float(a.get("page_quality", 0.0) or 0.0), float(b.get("page_quality", 0.0) or 0.0))

    # Structured fields: fill missing values from next batch
    sf_a = a.get("structured_fields", {})
    sf_b = b.get("structured_fields", {})
    for k, v in sf_b.items():
        if k not in sf_a or sf_a[k] in (None, "", []):
            sf_a[k] = v
    a["structured_fields"] = sf_a

    # Entities
    ent_a = a.get("entities", {})
    ent_b = b.get("entities", {})
    for key in ("companies", "people", "dates", "amounts", "addresses", "emails", "phone_numbers"):
        ent_a[key] = _merge_unique(ent_a.get(key, []), ent_b.get(key, []))
    a["entities"] = ent_a

    # Tables / stamps
    a["tables"] = (a.get("tables", []) or []) + (b.get("tables", []) or [])
    a["stamps"] = (a.get("stamps", []) or []) + (b.get("stamps", []) or [])

    # Signature: keep first found, otherwise take next
    sig_a = a.get("signature", {})
    sig_b = b.get("signature", {})
    if not sig_a.get("found") and sig_b.get("found"):
        a["signature"] = sig_b

    return a


def _attach_snippets(result: Dict[str, Any], pages: List[Image.Image]) -> None:
    sig = result.get("signature", {})
    if sig and sig.get("found") and sig.get("bbox"):
        page_idx = (sig.get("page") or 1) - 1
        if 0 <= page_idx < len(pages):
            sig["snippet_base64"] = _crop_snippet(pages[page_idx], sig["bbox"])

    for stamp in result.get("stamps", []) or []:
        if stamp.get("bbox"):
            page_idx = (stamp.get("page") or 1) - 1
            if 0 <= page_idx < len(pages):
                stamp["snippet_base64"] = _crop_snippet(pages[page_idx], stamp["bbox"])


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract native text from a PDF using PyMuPDF (fitz).
    Returns concatenated text from all pages, or "" if extraction fails.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(f"--- Page {page_num + 1} ---\n{text.strip()}")
        doc.close()
        return "\n\n".join(pages_text)
    except ImportError:
        logger.warning("PyMuPDF (fitz) not available — text extraction skipped")
        return ""
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        return ""


# ── Main entry point ──────────────────────────────────────

def analyse_document(pages: List[Image.Image], extracted_text: str = "") -> Dict[str, Any]:
    """
    Send all pages of a logical document to Gemini 2.5 Pro in a single call.

    Args:
        pages: List of PIL Image objects (one per page, already at correct DPI).
        extracted_text: Optional native PDF text appended as supplemental context.

    Returns:
        Parsed result dict with all fields described in _SYSTEM_PROMPT.
        Adds 'snippet_base64' to signature and each stamp if bbox is present.
    """
    if not pages:
        raise ValueError("No pages provided to analyse_document")

    client = _ensure_client()

    total_pages = len(pages)
    batch_size = max(1, int(GEMINI_BATCH_PAGE_SIZE))
    logger.info(
        f"Sending {total_pages} page(s) to Gemini ({GEMINI_MODEL}) with batch_size={batch_size}"
    )

    merged: Dict[str, Any] | None = None
    for i in range(0, total_pages, batch_size):
        batch_pages = pages[i : i + batch_size]
        page_offset = i
        batch_number = (i // batch_size) + 1
        batch_count = (total_pages + batch_size - 1) // batch_size

        parts = _build_parts(
            batch_pages,
            extracted_text=extracted_text if i == 0 else "",
            page_offset=page_offset,
        )
        batch_label = f"batch={batch_number}/{batch_count},pages={page_offset + 1}-{page_offset + len(batch_pages)}"

        # First call — normal token-budget cascade lives in _generate_with_retries.
        raw_text = _generate_with_retries(client, parts, batch_label)
        logger.debug(f"Gemini raw response ({batch_label}, first 300): {raw_text[:300]}")

        try:
            batch_result = _parse_validate_repair(client, raw_text, batch_label)
        except _TruncatedJSONError as exc:
            # Response body was cut mid-string — retry once with the largest
            # available token budget so the model can finish writing the JSON.
            big_budget = max(int(GEMINI_MAX_OUTPUT_TOKENS), 60000)
            logger.warning(
                f"{batch_label}: JSON truncated, retrying with max_output_tokens={big_budget}"
            )
            try:
                raw_text = _call_with_timeout(client, parts, big_budget)
                batch_result = _parse_validate_repair(client, raw_text, batch_label)
            except Exception as retry_exc:
                logger.error(
                    f"{batch_label}: truncation-retry failed ({retry_exc}); "
                    "falling back to default-shape result for this batch."
                )
                batch_result = _ensure_result_shape({})
        batch_result = _offset_page_numbers(batch_result, page_offset)
        merged = batch_result if merged is None else _merge_results(merged, batch_result)

    result = _ensure_result_shape(merged or {})
    conf = _self_evaluate_field_confidence(
        client,
        result.get("structured_fields", {}),
        result.get("ocr_text", "") or extracted_text,
    )
    result["field_confidence"] = conf.get("field_confidence", {})
    result["overall_extraction_confidence"] = conf.get("overall_extraction_confidence")
    result["confidence_notes"] = conf.get("notes", "")
    _attach_snippets(result, pages)

    logger.info(
        f"Gemini analysis complete — type={result.get('classification', {}).get('type')}, "
        f"lang={result.get('language')}, "
        f"tables={len(result.get('tables', []))}, "
        f"sig={result.get('signature', {}).get('found')}, "
        f"stamps={len(result.get('stamps', []))}"
    )

    return result
