"""
FastAPI Endpoints — Pro Tier AI Service
Mirrors the same API contract as the Basic tier so Aura's backend
can route to either service without code changes.
"""

import logging
import logging.handlers
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import hmac
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from config.settings import (
    API_TITLE,
    API_VERSION,
    DATABASE_NAME,
    GEMINI_MODEL,
    MONGO_URL,
    UPLOADS_DIR,
    TEMP_DIR,
    LOG_LEVEL,
    LOG_JSON,
)
from app.core.pipeline import run_pipeline
from app.services.embeddings import embed
from app.utils.logging_setup import configure_logging, trace_id_middleware


# ── Logging ───────────────────────────────────────────────
configure_logging(level=LOG_LEVEL, json_output=bool(LOG_JSON))
logger = logging.getLogger("endpoints")


# ── Security helpers ──────────────────────────────────────
_ALLOWED_DIRS = [
    os.path.realpath(str(UPLOADS_DIR)),
    os.path.realpath(str(TEMP_DIR)),
]


def _safe_path(path: str) -> str:
    resolved = os.path.realpath(path)
    if not any(resolved.startswith(d) for d in _ALLOWED_DIRS):
        raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")
    return resolved


# ── Database ──────────────────────────────────────────────
_db_client: MongoClient | None = None


def get_db():
    global _db_client
    if _db_client is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return _db_client[DATABASE_NAME]


# ── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_client
    logger.info(f"Starting {API_TITLE} v{API_VERSION} — model: {GEMINI_MODEL}")

    # MongoDB
    try:
        _db_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        _db_client.server_info()
        logger.info("MongoDB connected")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")

    # Pre-load embedding model (so first request is fast)
    try:
        from app.services.embeddings import get_model
        get_model()
        logger.info("Embedding model ready")
    except Exception as e:
        logger.warning(f"Embedding model preload failed: {e}")

    yield

    if _db_client:
        _db_client.close()
        logger.info("MongoDB disconnected")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach a trace id to every request so log lines can be correlated
# across the upload → pipeline → Gemini call chain.
app.middleware("http")(trace_id_middleware)


# ── Models ────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    doc_id: str
    file_path: str
    config_override: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    company_id: Optional[str] = None
    doc_id: Optional[str] = None


class QARequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    company_id: Optional[str] = None


# ── Health & Status ──────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tier": "pro",
        "model": GEMINI_MODEL,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/status")
async def status():
    """Status endpoint compatible with AuraControlPanel tier monitoring."""
    from app.services.embeddings import _model  # check if loaded
    return {
        "status": "healthy",
        "tier": "pro",
        "model": GEMINI_MODEL,
        "models_loaded": [
            "Gemini 2.5 Pro",
            "SentenceTransformer (Multilingual)",
        ],
        "embedding_model_ready": _model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/scanners")
async def list_scanners():
    """
    Pro tier uses file upload, not physical scanners.
    Returns empty list in the same shape as the Basic tier so Aura's frontend
    doesn't break — it just shows 'No scanners found'.
    """
    return {"scanners": [], "count": 0, "tier": "pro", "note": "Physical scanner input is not available on the Pro tier. Use file upload instead."}


@app.get("/")
async def root():
    return {"service": API_TITLE, "version": API_VERSION, "tier": "pro"}


# ── Upload & Process ──────────────────────────────────────
@app.post("/api/v1/process")
async def process_document(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger background AI processing for an already-uploaded document.
    Same contract as Basic tier /api/v1/process.
    """
    safe_path = _safe_path(request.file_path)
    db = get_db()
    background_tasks.add_task(run_pipeline, request.doc_id, safe_path, db)
    return {"status": "queued", "doc_id": request.doc_id}


@app.post("/api/v1/upload-and-process")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    company_id: Optional[str] = Form(None),
):
    """
    Upload a PDF and immediately queue processing.
    """
    # Validate extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save file
    safe_filename = f"{doc_id}{ext}"
    dest_path = UPLOADS_DIR / safe_filename
    content = await file.read()
    dest_path.write_bytes(content)

    db = get_db()
    background_tasks.add_task(run_pipeline, doc_id, str(dest_path), db)

    return {"status": "queued", "doc_id": doc_id, "file": safe_filename}


# ── Compat alias: Aura calls /api/v1/upload ──────────────
@app.post("/api/v1/upload")
async def upload_compat(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    company_id: Optional[str] = Form(None),
):
    """Alias of /api/v1/upload-and-process for Aura backend compatibility."""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    safe_filename = f"{doc_id}{ext}"
    dest_path = UPLOADS_DIR / safe_filename
    content = await file.read()
    dest_path.write_bytes(content)

    db = get_db()
    background_tasks.add_task(run_pipeline, doc_id, str(dest_path), db)

    return {"status": "queued", "doc_id": doc_id, "file": safe_filename, "message": "Document queued for processing"}


# ── Status ────────────────────────────────────────────────
@app.get("/api/v1/status/{doc_id}")
async def get_status(doc_id: str):
    from bson import ObjectId
    db = get_db()
    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    doc = db["documents"].find_one({"_id": oid}, {"status": 1, "error": 1, "processing_completed": 1})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "doc_id": doc_id,
        "status": doc.get("status", "UNKNOWN"),
        "error": doc.get("error"),
        "completed_at": doc.get("processing_completed"),
    }


# ── Compat alias: Aura polls /api/v1/documents/{doc_id}/status ──
@app.get("/api/v1/documents/{doc_id}/status")
async def get_document_status_compat(doc_id: str):
    """
    Aura-compatible status endpoint.
    When COMPLETE, includes processed_data so Aura can read results directly.
    """
    from bson import ObjectId
    db = get_db()
    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    doc = db["documents"].find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    status = doc.get("status", "UNKNOWN")
    resp: dict = {
        "doc_id": doc_id,
        "status": status,
        "error": doc.get("error"),
        "completed_at": doc.get("processing_completed"),
    }

    if status == "COMPLETE":
        resp["processed_data"] = doc.get("processed_data", [])
        resp["document_count"] = len(resp["processed_data"])

    return resp


# ── Results ───────────────────────────────────────────────
@app.get("/api/v1/results/{doc_id}")
async def get_results(doc_id: str):
    from bson import ObjectId
    db = get_db()
    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    doc = db["documents"].find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    processed = doc.get("processed_data", [])
    return {
        "doc_id": doc_id,
        "status": doc.get("status"),
        "processed_data": processed,
        "document_count": len(processed),
    }


# ── Semantic Search ───────────────────────────────────────
@app.post("/api/v1/search")
async def semantic_search(request: SearchRequest):
    """
    Vector similarity search over all processed documents.
    Uses cosine similarity on SentenceTransformer embeddings.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    db = get_db()
    query_embedding = embed(request.query)

    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")

    # Load all docs with embeddings (in production this would use a vector index)
    filter_q: Dict[str, Any] = {"status": "COMPLETE"}
    if request.company_id:
        filter_q["company_id"] = request.company_id

    docs = list(db["documents"].find(filter_q, {
        "_id": 1, "originalName": 1, "processed_data": 1
    }))

    import numpy as np

    q_vec = np.array(query_embedding)
    results = []

    for doc in docs:
        for pd in doc.get("processed_data", []):
            emb = pd.get("embedding", [])
            if not emb:
                continue
            d_vec = np.array(emb)
            # Cosine similarity
            norm = np.linalg.norm(q_vec) * np.linalg.norm(d_vec)
            score = float(np.dot(q_vec, d_vec) / norm) if norm > 0 else 0.0
            if score >= 0.15:
                results.append({
                    "doc_id": str(doc["_id"]),
                    "filename": doc.get("originalName", ""),
                    "score": round(score, 4),
                    "summary": pd.get("summary", ""),
                    "classification": pd.get("classification", {}),
                    "language": pd.get("language", ""),
                    "page_range": pd.get("page_range", ""),
                })

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"query": request.query, "results": results[: request.top_k]}


# ── Q&A ───────────────────────────────────────────────────
@app.post("/api/v1/qa")
async def question_answer(request: QARequest):
    """
    Answer a question about a document using Gemini.
    If doc_id is provided, answer about that specific document.
    """
    from google import genai
    from google.genai import types
    from config.settings import GEMINI_API_KEY, GEMINI_MODEL
    from app.services.gemini_service import _ensure_client

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    db = get_db()
    context_text = ""

    if request.doc_id:
        from bson import ObjectId
        try:
            oid = ObjectId(request.doc_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid doc_id")

        doc = db["documents"].find_one({"_id": oid})
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        for pd in doc.get("processed_data", []):
            context_text += pd.get("full_text", "") + "\n\n"

    if not context_text:
        raise HTTPException(status_code=400, detail="No document text available for Q&A")

    client = _ensure_client()

    prompt = (
        "You are an AI assistant helping a user understand a document. "
        "Use the document content below as your primary source of facts about the document. "
        "For questions that require general knowledge, calculations, conversions (e.g. currency, "
        "units), translations, definitions, or reasoning beyond the document, you MAY use your own "
        "knowledge — combine it with the document's data when relevant. "
        "If a fact is not in the document and cannot be reasonably derived or inferred, say so. "
        "When converting currency, use a reasonable recent approximate exchange rate and clearly "
        "label it as approximate.\n\n"
        f"Document content:\n\n{context_text[:50000]}\n\n"
        f"Question: {request.question}\n\n"
        "Answer concisely and helpfully."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1024,
        ),
    )

    return {
        "question": request.question,
        "answer": response.text,
        "doc_id": request.doc_id,
    }


# ── Conversational Search (semantic + Gemini synthesis) ─────────────────────────────────────
@app.post("/api/v1/conversational-search")
async def conversational_search(request: SearchRequest) -> Dict[str, Any]:
    """
    Hybrid: semantic vector search across documents, then synthesize an answer
    with Gemini using the top-matching chunks as context.
    Mirrors the Basic-tier API contract so the Aura backend proxy works.
    """
    from google.genai import types
    from config.settings import GEMINI_MODEL
    from app.services.gemini_service import _ensure_client
    import numpy as np

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    db = get_db()
    query_embedding = embed(request.query)

    # Build doc filter (optional doc_id / company scoping).
    filter_q: Dict[str, Any] = {"status": "COMPLETE"}
    if request.company_id:
        filter_q["company_id"] = request.company_id
    if request.doc_id:
        from bson import ObjectId
        try:
            filter_q["_id"] = ObjectId(request.doc_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid doc_id")

    docs = list(db["documents"].find(filter_q, {
        "_id": 1, "originalName": 1, "processed_data": 1
    }))

    scored: list[Dict[str, Any]] = []
    if query_embedding:
        q_vec = np.array(query_embedding)
        q_norm = float(np.linalg.norm(q_vec))
        for doc in docs:
            for pd in doc.get("processed_data", []):
                emb = pd.get("embedding", [])
                if not emb:
                    continue
                d_vec = np.array(emb)
                d_norm = float(np.linalg.norm(d_vec))
                if q_norm == 0 or d_norm == 0:
                    continue
                score = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
                if score < 0.10:
                    continue
                scored.append({
                    "doc_id": str(doc["_id"]),
                    "filename": doc.get("originalName", ""),
                    "score": round(score, 4),
                    "text": pd.get("full_text", "") or pd.get("summary", ""),
                })
        scored.sort(key=lambda r: r["score"], reverse=True)
    else:
        # Fallback: just take whatever docs we have if embeddings unavailable.
        for doc in docs[:10]:
            for pd in doc.get("processed_data", []):
                txt = pd.get("full_text", "") or pd.get("summary", "")
                if txt:
                    scored.append({
                        "doc_id": str(doc["_id"]),
                        "filename": doc.get("originalName", ""),
                        "score": 0.5,
                        "text": txt,
                    })
                    break

    if not scored:
        return {
            "query": request.query,
            "answer": "No documents matched your query.",
            "source_documents": [],
            "total_documents": 0,
        }

    top = scored[: max(request.top_k, 5)]
    context_parts = [
        f"[Document: {hit['filename']} | score={hit['score']}]\n{hit['text'][:1500]}"
        for hit in top[:5]
    ]
    combined_context = "\n\n".join(context_parts)[:12000]

    prompt = (
        "You are an AI assistant answering questions about a user's document library. "
        "Use the document excerpts below as your primary source. You MAY use general knowledge "
        "for calculations, currency/unit conversions, definitions, and reasoning that build on "
        "the document data. If the answer truly cannot be derived, say so.\n\n"
        f"Documents:\n{combined_context}\n\n"
        f"Question: {request.query}\n\nAnswer concisely."
    )

    answer_text = ""
    try:
        client = _ensure_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        answer_text = (response.text or "").strip()
    except Exception as e:
        logger.error(f"Conversational search Gemini call failed: {e}")
        answer_text = f"(AI synthesis unavailable: {e}) Showing top document matches."

    source_documents = [
        {
            "document_id": hit["doc_id"],
            "filename": hit["filename"],
            "score": hit["score"],
            "text_excerpt": (hit["text"] or "")[:300],
        }
        for hit in top[: request.top_k]
    ]

    return {
        "query": request.query,
        "answer": answer_text,
        "source_documents": source_documents,
        "total_documents": len(source_documents),
    }
