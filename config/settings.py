"""
Configuration Settings — Pro Tier AI Service
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Server ────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "9002"))
API_TITLE = "Adaptive Document Processing — Pro Tier"
API_VERSION = "1.0.0"

# ── Database ──────────────────────────────────────────────
MONGO_URL = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "aura")

# ── Google Gemini ─────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "60000"))
GEMINI_BATCH_PAGE_SIZE = int(os.getenv("GEMINI_BATCH_PAGE_SIZE", "12"))
GEMINI_RETRY_ATTEMPTS = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "3"))
GEMINI_RETRY_BACKOFF_MS = int(os.getenv("GEMINI_RETRY_BACKOFF_MS", "700"))
GEMINI_SUPPLEMENTAL_TEXT_MAX_CHARS = int(os.getenv("GEMINI_SUPPLEMENTAL_TEXT_MAX_CHARS", "120000"))
GEMINI_PROVIDER_TIMEOUT_SEC = int(os.getenv("GEMINI_PROVIDER_TIMEOUT_SEC", "120"))
GEMINI_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("GEMINI_CIRCUIT_BREAKER_THRESHOLD", "5"))
GEMINI_CIRCUIT_BREAKER_COOLDOWN_SEC = int(os.getenv("GEMINI_CIRCUIT_BREAKER_COOLDOWN_SEC", "90"))
GEMINI_SCHEMA_REPAIR_ATTEMPTS = int(os.getenv("GEMINI_SCHEMA_REPAIR_ATTEMPTS", "2"))
GEMINI_FIELD_SELF_EVAL_ENABLED = os.getenv("GEMINI_FIELD_SELF_EVAL_ENABLED", "true").lower() in ("1", "true", "yes", "on")
GEMINI_FIELD_SELF_EVAL_MAX_FIELDS = int(os.getenv("GEMINI_FIELD_SELF_EVAL_MAX_FIELDS", "60"))
GEMINI_FIELD_SELF_EVAL_MAX_CONTEXT_CHARS = int(os.getenv("GEMINI_FIELD_SELF_EVAL_MAX_CONTEXT_CHARS", "15000"))

# ── Processing ────────────────────────────────────────────
# Default raised to 200 DPI to match the Plus tier and improve handwriting fidelity
# (~18% visual quality improvement on small print and signatures vs 150 DPI).
GEMINI_IMAGE_DPI = int(os.getenv("GEMINI_IMAGE_DPI", "200"))
# JPEG quality used when converting page images for Gemini. 92 is a sweet spot:
# ~30% larger payload than 85, but materially better legibility on handwritten
# text and faint stamps. Override via env when on tight network/budget.
GEMINI_JPEG_QUALITY = int(os.getenv("GEMINI_JPEG_QUALITY", "92"))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "500"))
MAX_PAGES = int(os.getenv("MAX_PAGES_PER_DOCUMENT", "100"))
POPPLER_PATH = os.getenv("POPPLER_PATH") or None

# ── Pipeline guards ────────────────────────────────────────
LOW_CONFIDENCE_REVIEW_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_REVIEW_THRESHOLD", "0.7"))
MIN_FIELDS_FOR_HIGH_CONFIDENCE = int(os.getenv("MIN_FIELDS_FOR_HIGH_CONFIDENCE", "3"))

# ── Image preprocessing ──────────────────────────────────────
IMAGE_PREPROCESS_ENABLED = os.getenv("IMAGE_PREPROCESS_ENABLED", "true").lower() in ("1", "true", "yes", "on")
IMAGE_PREPROCESS_AUTO_ROTATE = os.getenv("IMAGE_PREPROCESS_AUTO_ROTATE", "true").lower() in ("1", "true", "yes", "on")
IMAGE_PREPROCESS_DESKEW = os.getenv("IMAGE_PREPROCESS_DESKEW", "true").lower() in ("1", "true", "yes", "on")
IMAGE_PREPROCESS_CONTRAST = os.getenv("IMAGE_PREPROCESS_CONTRAST", "true").lower() in ("1", "true", "yes", "on")

# ── Embeddings ──────────────────────────────────────────────
EMBEDDINGS_CHUNK_ENABLED = os.getenv("EMBEDDINGS_CHUNK_ENABLED", "true").lower() in ("1", "true", "yes", "on")
EMBEDDINGS_CHUNK_SIZE_CHARS = int(os.getenv("EMBEDDINGS_CHUNK_SIZE_CHARS", "1500"))
EMBEDDINGS_CHUNK_OVERLAP_CHARS = int(os.getenv("EMBEDDINGS_CHUNK_OVERLAP_CHARS", "200"))
EMBEDDINGS_MAX_CHUNKS = int(os.getenv("EMBEDDINGS_MAX_CHUNKS", "60"))

# ── Result cache (file SHA → processed_data) ────────────────────────
RESULT_CACHE_ENABLED = os.getenv("RESULT_CACHE_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# ── Cost tracking (USD per 1k tokens, override per pricing change) ──────────────
GEMINI_COST_INPUT_PER_1K = float(os.getenv("GEMINI_COST_INPUT_PER_1K", "0.00125"))
GEMINI_COST_OUTPUT_PER_1K = float(os.getenv("GEMINI_COST_OUTPUT_PER_1K", "0.005"))

# ── Structured logging ────────────────────────────────────────
LOG_JSON = os.getenv("LOG_JSON", "false").lower() in ("1", "true", "yes", "on")

# ── Data Directories ──────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
TEMP_DIR = DATA_DIR / "temp"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
