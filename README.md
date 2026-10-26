# Adaptive Document Processing - AI Service (Pro Tier)

The Pro tier AI service of the Aura document platform. It sends each uploaded document (PDF or
image) to Google Gemini 2.5 Pro, which reads the pages, classifies the document and returns
structured fields, entities, tables, signatures, stamps and a summary. The service checks the
result, stores it in MongoDB and makes it searchable and answerable in natural language.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Status](#project-status)
3. [How It Works](#how-it-works)
4. [Processing Pipeline](#processing-pipeline)
5. [Components](#components)
6. [AI Models](#ai-models)
7. [Differences from the Basic Tier](#differences-from-the-basic-tier)
8. [Technology Stack](#technology-stack)
9. [Project Structure](#project-structure)
10. [Getting Started](#getting-started)
11. [Configuration](#configuration)
12. [API Reference](#api-reference)
13. [Security](#security)
14. [Testing and Benchmarks](#testing-and-benchmarks)
15. [Known Issues](#known-issues)
16. [License](#license)

---

## Overview

### Purpose

Companies receive many business documents: invoices, contracts, delivery notes, receipts,
bank statements and letters. Typing their key data into business systems by hand is slow and
error-prone. This service is a Python REST API (FastAPI) that automates this work with a large
multimodal language model. For every document it:

- reads all printed and handwritten text (the model works directly on page images),
- detects the language and whether handwriting is present,
- classifies the document into a fixed list of English document types,
- extracts structured fields with fixed snake_case names (for example `invoice_number`,
  `total_amount`, `iban`) and entities (companies, people, dates, amounts, addresses,
  e-mail addresses, phone numbers),
- extracts tables, signatures and stamps, with position and an image snippet,
- writes a short summary,
- checks the result (invoice totals, values not found in the text, confidence per field),
- creates search vectors for semantic search and question answering.

### Place in the Aura platform

The Aura platform offers its AI service in three tiers. Each tier is a separate service with
the same REST contract, so the Aura backend can talk to any of them.

| Tier | Repository | Default port |
|------|------------|--------------|
| Basic (Tier 1) | `AuraAITier1` (local models) | 9000 |
| Plus | separate repository | 9001 |
| **Pro (this repository)** | **`adaptive-Dokumentenverarbeitung-Pro`** | **9002** |

The service shares the MongoDB database `aura` with the Aura backend. `/api/v1/status` has the
shape the Aura Control Panel uses for tier monitoring.

### Intended users

- **The Aura backend**, which uploads documents, polls their status and reads the results.
- **The Aura Control Panel**, which monitors the tier.
- **Developers** who need high-quality document extraction through a REST API.

---

## Project Status

Version 1.0.0 (`API_VERSION` in `config/settings.py`), in active development.

| Feature | Status |
|---------|--------|
| PDF and image upload, background processing | Working |
| Extraction with Gemini 2.5 Pro | Working |
| Batching of long documents (12 pages per call by default) | Working |
| Retries, timeout, circuit breaker, JSON repair | Working |
| Field self-evaluation (confidence per field) | Working (optional) |
| Validators and "review required" flag | Working |
| Result cache by file hash (SHA-256) | Working (optional) |
| Token usage and cost estimate per document | Working |
| Semantic search, Q&A, conversational search | Working |
| Deskew of rotated scans | Only if OpenCV is installed (not in `requirements.txt`) |
| Chunk embeddings | Stored; not used by the search endpoints yet |
| Splitting one PDF into several documents | Not implemented |
| Physical scanners | Not supported on this tier |
| Authentication, automated tests, Docker | Not present |

---

## How It Works

```
   Aura backend / Control Panel
                |
                | REST (JSON, multipart)
                v
   +----------------------------------------------+
   |  Pro AI Service  (FastAPI, port 9002)         |
   |  /api/v1/upload  -->  background task         |
   |                          |                   |
   |                          v                   |
   |   run_pipeline: validate > cache > text >    |
   |   images > Gemini > normalise > embed >      |
   |   validate > store                           |
   +-------+-------------------+------------------+
           |                   |                  |
           v                   v                  v
   +--------------+   +------------------+  +----------------------+
   | MongoDB      |   | Google Gemini    |  | Local embedding model|
   | `aura`       |   | 2.5 Pro (cloud)  |  | multilingual MiniLM  |
   +--------------+   +------------------+  +----------------------+
```

1. The Aura backend creates a record in the `documents` collection and uploads the file to
   `/api/v1/upload` with its `doc_id`, or asks the service to process a file that already
   exists in the upload folder (`/api/v1/process`).
2. The service saves the file as `data/uploads/<doc_id>.<ext>` and returns `queued` at once.
3. The pipeline runs as a FastAPI background task and sets the status to `PROCESSING`.
4. At the end the record holds a `processed_data` array with one entry and the status
   `COMPLETE`, or `FAILED` with an `error` text.
5. The client polls `/api/v1/documents/{doc_id}/status`; when `COMPLETE`, the response
   already contains `processed_data`.
6. The client can then use search, Q&A and conversational search.

---

## Processing Pipeline

Implemented in `run_pipeline()` in `app/core/pipeline.py`; model calls are in
`app/services/gemini_service.py`.

| Step | What it does | Technology |
|------|--------------|------------|
| 1. Validate | File exists and is not larger than `MAX_PDF_SIZE_MB` | Python |
| 2. Result cache | If a byte-identical file (same SHA-256) was already processed, its result is copied and Gemini is not called | MongoDB |
| 3. Native text | Reads the embedded text layer of digital PDFs | PyMuPDF |
| 4. Page images | PDF pages to JPEG (`GEMINI_IMAGE_DPI`, at most `MAX_PAGES_PER_DOCUMENT` pages); image files are loaded directly | pdf2image, Poppler, Pillow |
| 5. Preprocessing | EXIF orientation, deskew (needs OpenCV), auto-contrast | Pillow, OpenCV (optional) |
| 6. Analysis | Sends pages in batches of `GEMINI_BATCH_PAGE_SIZE` with a strict JSON prompt and the native text as context; merges batch results | Gemini 2.5 Pro |
| 7. Repair | Salvages truncated JSON, asks the model to repair invalid JSON, fills missing keys with safe defaults | Python, Gemini |
| 8. Self-evaluation | A second call rates the confidence of each extracted field | Gemini 2.5 Pro |
| 9. Snippets | Crops signature and stamp areas and stores them as base64 PNG | Pillow |
| 10. Normalise | Maps field name variants (English, German, French, Spanish, Italian, Dutch, Turkish) and translated type labels to canonical English keys | Python |
| 11. Embeddings | One 384-dimension vector for the whole text, plus optional chunk vectors | Sentence Transformers |
| 12. Validators | Checks `subtotal + tax_amount = total_amount` (2 cent tolerance), flags values not found in the text, sets `low_confidence_review_required` with `review_reasons` | Python |
| 13. Store | Saves the result, token usage with estimated USD cost, and file hash | MongoDB |

**Reliability of the Gemini call:** timeout per call, retries with backoff, a lower output
token budget step by step after failures (100% down to 20% of `GEMINI_MAX_OUTPUT_TOKENS`),
one extra retry when the JSON answer is cut off, and a circuit breaker that blocks calls for
a cool-down time after several failures in a row.

**Document types:** invoice, credit_note, debit_note, payment_voucher, remittance_advice,
quotation, purchase_order, receipt, delivery_note, packing_slip, bill_of_lading,
shipping_document, contract, agreement, non_disclosure_agreement,
memorandum_of_understanding, employment_contract, letter, business_letter, bank_statement,
financial_statement, tax_document, id_document, medical_record, form, report,
tender_document, resume, job_application, certificate, license, permit, memo, proposal,
statement and other. Fixed field sets exist for 13 of them (for example invoice,
purchase_order, receipt, contract, bank_statement, tax_document).

**Result:** each `processed_data` entry holds `full_text`, `language`,
`handwriting_detected`, `page_quality`, `classification`, `structured_data`,
`field_confidence`, `entities`, `tables`, `signature`, `stamps`, `summary`, `validation`,
`low_confidence_review_required`, `embedding`, `embeddings_chunked`, `cost` and
`file_sha256`.

---

## Components

| File | Responsibility |
|------|----------------|
| `main.py` | Starts Uvicorn with `API_HOST` and `API_PORT` |
| `config/settings.py` | Reads all environment variables; creates `data/uploads` and `data/temp` |
| `app/api/endpoints.py` | Endpoints, MongoDB connection, CORS, path check, search and Q&A |
| `app/core/pipeline.py` | Runs all steps for one document, normalises fields and types, writes to MongoDB |
| `app/services/gemini_service.py` | Prompt, batching, retries, circuit breaker, JSON repair, self-evaluation, snippets, token usage |
| `app/services/embeddings.py` | Full and chunk vectors |
| `app/services/image_preprocess.py` | Orientation, deskew and contrast |
| `app/services/validators.py` | Totals check, hallucination check, review decision |
| `app/utils/logging_setup.py` | Text or JSON logs, trace id per request (`X-Trace-Id` header) |
| `scripts/benchmark_harness.py` | Accuracy benchmark against labelled files |

---

## AI Models

| Model | Where it runs | Used for |
|-------|---------------|----------|
| Google Gemini 2.5 Pro (`GEMINI_MODEL`) | Google cloud API | Reading pages, classification, extraction, tables, signatures, stamps, summary, JSON repair, self-evaluation, answers |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Locally, CPU | 384-dimension search vectors |

The embedding model is downloaded on first use and loaded at start-up. No GPU is needed.

---

## Differences from the Basic Tier

| Topic | Basic tier (`AuraAITier1`) | Pro tier (this repository) |
|-------|----------------------------|----------------------------|
| Port | 9000 | 9002 |
| Main AI | Local models (PaddleOCR, Llama 3.2 3B and others) | Google Gemini 2.5 Pro (cloud) |
| Data leaves the machine | No | Yes, page images and text go to the Gemini API |
| OCR | Separate OCR step | Done by Gemini on page images |
| Tables | Disabled in the pipeline | Extracted |
| Signatures | Text patterns | Visual, with position and image snippet |
| Stamps | Not available | Detected, with text, date, type and snippet |
| PDF splitting | Yes | No, one document per upload |
| Quality checks | OCR correction | Self-evaluation, totals and hallucination checks, review flag |
| Cost tracking | Not applicable | Tokens and estimated USD per document |
| Search index | FAISS | Vectors stored in MongoDB, compared in Python |
| Scanners | Windows WIA | Not supported |
| API key check | `X-API-Key` header | None |

The paths the Aura backend uses (`/api/v1/upload`, `/api/v1/process`,
`/api/v1/documents/{doc_id}/status`, `/api/v1/search`, `/api/v1/conversational-search`,
`/api/v1/scanners`) keep the Basic tier shape.

---

## Technology Stack

| Area | Technology |
|------|------------|
| Language | Python 3 |
| Web framework | FastAPI, Uvicorn |
| AI model | Gemini 2.5 Pro via `google-genai` |
| Embeddings | `sentence-transformers` |
| PDF | PyMuPDF, `pdf2image` with Poppler |
| Images | Pillow (OpenCV optional) |
| Database | MongoDB via `pymongo` |
| Configuration | `python-dotenv` |

Package versions are not pinned in `requirements.txt`.

---

## Project Structure

```
adaptive-Dokumentenverarbeitung-Pro/
|-- main.py
|-- requirements.txt
|-- start_server.bat / start_server.ps1
|-- .env.example
|-- BENCHMARKS.md
|-- config/
|   `-- settings.py
|-- app/
|   |-- api/endpoints.py
|   |-- core/pipeline.py
|   |-- services/
|   |   |-- gemini_service.py
|   |   |-- embeddings.py
|   |   |-- image_preprocess.py
|   |   `-- validators.py
|   `-- utils/logging_setup.py
|-- scripts/
|   `-- benchmark_harness.py
`-- data/benchmarks/
    |-- docs/       2 PDFs, 1 PNG
    |-- labels/     expected results (JSON)
    `-- reports/    generated reports
```

`data/uploads/`, `data/temp/`, `logs/` and `.env` are ignored by Git.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer (the code uses `X | None` type syntax).
- MongoDB (default `mongodb://localhost:27017/`).
- Poppler for `pdf2image`. On Windows, set `POPPLER_PATH` to its `bin` folder.
- A Google Gemini API key and internet access.
- Optional: `opencv-python` for deskew.

### Install, configure, run

```bash
git clone https://github.com/ahmadkobah/adaptive-Dokumentenverarbeitung-Pro.git
cd adaptive-Dokumentenverarbeitung-Pro
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env
python main.py
```

In `.env` set at least `GEMINI_API_KEY`, `MONGODB_URI` and (Windows) `POPPLER_PATH`. The
`POPPLER_PATH` in `.env.example` is a path on the author's machine. On Windows you can also
use `start_server.bat` or `start_server.ps1`; they create `.venv`, install the requirements
and start the server (they always print port 9002).

Check: `curl http://localhost:9002/health`. FastAPI docs: `http://localhost:9002/docs`.

---

## Configuration

Environment variables, read in `config/settings.py`. Names and defaults:

| Variable | Default |
|----------|---------|
| `API_HOST` / `API_PORT` | `0.0.0.0` / `9002` |
| `MONGODB_URI` / `DATABASE_NAME` | `mongodb://localhost:27017/` / `aura` |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `false` |
| `GEMINI_API_KEY` | empty (required) |
| `GEMINI_MODEL` | `gemini-2.5-pro` |
| `GEMINI_MAX_OUTPUT_TOKENS` | `60000` |
| `GEMINI_BATCH_PAGE_SIZE` | `12` |
| `GEMINI_RETRY_ATTEMPTS` / `GEMINI_RETRY_BACKOFF_MS` | `3` / `700` |
| `GEMINI_PROVIDER_TIMEOUT_SEC` | `120` |
| `GEMINI_CIRCUIT_BREAKER_THRESHOLD` / `..._COOLDOWN_SEC` | `5` / `90` |
| `GEMINI_SCHEMA_REPAIR_ATTEMPTS` | `2` |
| `GEMINI_FIELD_SELF_EVAL_ENABLED` | `true` |
| `GEMINI_FIELD_SELF_EVAL_MAX_FIELDS` / `..._MAX_CONTEXT_CHARS` | `60` / `15000` |
| `GEMINI_SUPPLEMENTAL_TEXT_MAX_CHARS` | `120000` |
| `GEMINI_COST_INPUT_PER_1K` / `GEMINI_COST_OUTPUT_PER_1K` | `0.00125` / `0.005` (USD, estimate) |
| `GEMINI_IMAGE_DPI` | `200` (`.env.example` sets `150`) |
| `GEMINI_JPEG_QUALITY` | `92` |
| `MAX_PDF_SIZE_MB` / `MAX_PAGES_PER_DOCUMENT` | `500` / `100` |
| `POPPLER_PATH` | not set |
| `LOW_CONFIDENCE_REVIEW_THRESHOLD` / `MIN_FIELDS_FOR_HIGH_CONFIDENCE` | `0.7` / `3` |
| `IMAGE_PREPROCESS_ENABLED`, `..._AUTO_ROTATE`, `..._DESKEW`, `..._CONTRAST` | `true` |
| `EMBEDDINGS_CHUNK_ENABLED` | `true` |
| `EMBEDDINGS_CHUNK_SIZE_CHARS` / `..._OVERLAP_CHARS` / `EMBEDDINGS_MAX_CHUNKS` | `1500` / `200` / `60` |
| `RESULT_CACHE_ENABLED` | `true` |

---

## API Reference

No endpoint needs authentication.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/`, `/health` | Service, version, tier (`pro`), model |
| GET | `/api/v1/status` | Status for Control Panel monitoring, loaded models, `embedding_model_ready` |
| GET | `/api/v1/scanners` | Always an empty list |
| POST | `/api/v1/upload` | Multipart: `file`, `doc_id`, optional `company_id`; saves and queues the file |
| POST | `/api/v1/upload-and-process` | Same as `/api/v1/upload` |
| POST | `/api/v1/process` | JSON: `doc_id`, `file_path`, optional `config_override`; file must be in `data/uploads` or `data/temp` |
| GET | `/api/v1/status/{doc_id}` | Status, error, completion time |
| GET | `/api/v1/documents/{doc_id}/status` | Same, plus `processed_data` when `COMPLETE` |
| GET | `/api/v1/results/{doc_id}` | `processed_data` and `document_count` |
| POST | `/api/v1/search` | `query`, `top_k` (10), `company_id`; cosine similarity, score 0.15 or higher |
| POST | `/api/v1/qa` | `question`, `doc_id`; Gemini answers about one document |
| POST | `/api/v1/conversational-search` | `query`, `top_k`, `company_id`, `doc_id`; best matches (score 0.10 or higher), then one Gemini answer with sources |

Upload types: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`. `doc_id` must be the
ObjectId of an existing record in `documents`.

```bash
curl -X POST http://localhost:9002/api/v1/upload -F "file=@invoice.pdf" -F "doc_id=<ObjectId>"
curl http://localhost:9002/api/v1/documents/<ObjectId>/status
```

---

## Security

- **No authentication**, and CORS allows all origins with credentials. Run the service only
  on a trusted internal network behind the Aura backend.
- **Data goes to Google.** Page images and document text are sent to the Gemini API. Check
  that this is allowed for your documents (GDPR / DSGVO).
- **Secrets** belong in `.env`, which is ignored by Git. Never commit a real API key or
  database connection string.
- `/api/v1/process` only accepts files inside `data/uploads` or `data/temp`.
- Signature and stamp snippets are stored as images in MongoDB; treat the database as
  sensitive.

---

## Testing and Benchmarks

There are no automated tests. `scripts/benchmark_harness.py` calls Gemini directly (no server
or MongoDB needed) and writes a JSON report with classification accuracy, field accuracy and
per-document details. See `BENCHMARKS.md`.

```bash
python scripts/benchmark_harness.py
```

The three label files only define the expected type (`invoice`) and no fields, so the two
stored reports show classification accuracy 1.0 on 3 documents and no field accuracy.

---

## Known Issues

- Every upload is one logical document; multi-document PDFs are not split.
- `doc_id` from the upload form is used in the file name without validation, and the
  directory check in `/api/v1/process` compares path prefixes only.
- Uploads are read fully into memory and saved before the size limit is checked.
- The upload endpoints do not create the MongoDB record; without an existing record the
  result is not saved.
- The pipeline is an `async` function that calls blocking code, so long jobs can slow other
  requests in the same process.
- Token usage and the circuit breaker are global per process; with parallel jobs, the cost
  summary of one document can include tokens of another.
- Search loads all completed documents and compares vectors in Python; there is no vector
  index, and chunk vectors are not used yet.
- `/api/v1/qa` ignores `company_id` and fails without `doc_id`. `config_override` is ignored.
- Deskew needs OpenCV, which is not in `requirements.txt`.
- Pages after `MAX_PAGES_PER_DOCUMENT` are dropped without a note in the result.
- `.env.example` does not list all variables and uses a different DPI than the code default.
- No Docker files and no pinned package versions.

---

## License

No license file is included. Private. All rights reserved.

Author: Ahmad Kobah
