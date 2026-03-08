"""
src/api/main_ml.py  — ML Backend (Port 8000)  v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW in v3:
  • /upload endpoint accepts JPEG, PNG, PDF, DOCX — returns table + summary
  • Async file handling with aiofiles
  • Structured table JSON returned alongside narrative
  • BERTScore added to /evaluate/single
  • Full CORS + error handling
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import time
import json
import os
import sys
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="Clinical ML Narrative Engine — v3.0",
    description=(
        "3-Stage Hybrid Pipeline: Feature Extraction (32 features) → "
        "KNN Retrieval → Slot-Fill Generation + Hallucination Verification. "
        "Accepts PDF, DOCX, JPEG, PNG uploads and returns structured table + narrative."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supported file types ──────────────────────────────────────────────────────

SUPPORTED_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"}

# ── Request / Response Models ─────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    table_text: str
    return_features: Optional[bool] = False
    strict_mode: Optional[bool] = False

class TableResult(BaseModel):
    table_id: str
    table_text: str           # linearized string
    html: Optional[str] = None
    json_data: Optional[List[Dict]] = None
    headers: Optional[List[str]] = None
    summary: str
    verified: bool
    cluster_id: str
    cluster_description: str
    numeric_accuracy: float
    hallucination_free: bool
    inference_time_ms: float
    warnings: List[str]
    slots_filled: Optional[Dict] = None

class UploadResponse(BaseModel):
    filename: str
    file_type: str
    tables_found: int
    results: List[TableResult]
    total_time_ms: float

class SummarizeResponse(BaseModel):
    summary: str
    verified: bool
    cluster_id: str
    cluster_description: str
    numeric_accuracy: float
    hallucination_free: bool
    inference_time_ms: float
    warnings: List[str]
    features: Optional[dict] = None
    slots_filled: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    retrieval_index_loaded: bool
    corpus_size: int
    version: str
    supported_upload_types: List[str]

# ── Component Initialisation ──────────────────────────────────────────────────

from src.feature_engineering.statistical_features import StatisticalFeatureExtractor
from src.generation.slot_fill_generator import SlotFillGenerator, HallucinationError

feature_extractor = StatisticalFeatureExtractor()
generator = SlotFillGenerator(strict_mode=False)

# KNN retrieval index
knn_engine = None
CORPUS_SIZE = 0
INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "retrieval_index.pkl"

try:
    from src.retrieval.knn_retrieval_engine import KNNRetrievalEngine
    if INDEX_PATH.exists():
        knn_engine = KNNRetrievalEngine.load(str(INDEX_PATH))
        CORPUS_SIZE = len(knn_engine.corpus_pairs)
        logger.info(f"✅ KNN index loaded — {CORPUS_SIZE} examples")
    else:
        logger.warning(f"KNN index not found at {INDEX_PATH}")
except Exception as e:
    logger.warning(f"KNN engine unavailable: {e}")


# ── Core summarization helper ─────────────────────────────────────────────────

def _run_pipeline(table_text: str) -> dict:
    """Run the 3-stage ML pipeline on a single linearized table string."""
    start = time.time()

    retrieved_writeup = None
    retrieved_table_text = None
    if knn_engine is not None:
        try:
            best = knn_engine.retrieve_best(table_text)
            retrieved_writeup = best.get("writeup", "")
            retrieved_table_text = best.get("table_text", "")
        except Exception:
            pass

    gen = generator.generate(
        linearized_text=table_text,
        retrieved_writeup=retrieved_writeup,
        retrieved_table_text=retrieved_table_text,
    )

    elapsed = (time.time() - start) * 1000
    return {
        "summary": gen.narrative,
        "verified": gen.verified,
        "cluster_id": gen.cluster_id,
        "cluster_description": gen.cluster_description,
        "numeric_accuracy": gen.numeric_accuracy,
        "hallucination_free": gen.verified,
        "inference_time_ms": round(elapsed, 2),
        "warnings": gen.warnings,
        "slots_filled": gen.slots_filled,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        retrieval_index_loaded=knn_engine is not None,
        corpus_size=CORPUS_SIZE,
        version="3.0.0",
        supported_upload_types=list(SUPPORTED_TYPES.keys()),
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize a linearized table string (text-only input)."""
    try:
        result = _run_pipeline(request.table_text)
        features = feature_extractor.extract(request.table_text) if request.return_features else None
        result["features"] = features
        return SummarizeResponse(**result)
    except HallucinationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("summarize failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_and_summarize(
    file: UploadFile = File(...),
    use_gpu_ocr: bool = Form(False),
):
    """
    ★ PRIMARY ENDPOINT ★

    Upload ANY clinical document (PDF, DOCX, JPEG, PNG) and receive:
      - Structured table JSON + HTML for each detected table
      - Clinical narrative summary per table
      - Hallucination-verified output

    Supports:
      • PDF  — text-based (pdfplumber) or scanned (PyMuPDF + OCR fallback)
      • DOCX — python-docx table parser
      • JPEG / PNG / BMP / TIFF — EasyOCR + img2table
    """
    start_total = time.time()
    content_type = file.content_type or ""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    # Determine file type
    if content_type in IMAGE_TYPES or suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        file_type = "image"
    elif content_type == "application/pdf" or suffix == ".pdf":
        file_type = "pdf"
    elif "word" in content_type or suffix in (".docx", ".doc"):
        file_type = "docx"
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. "
                   f"Supported: PDF, DOCX, JPEG, PNG, BMP, TIFF",
        )

    # Read content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Write to temp file
    tmp_suffix = suffix if suffix else (".jpg" if file_type == "image" else ".pdf")
    with tempfile.NamedTemporaryFile(suffix=tmp_suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    results: List[TableResult] = []

    try:
        if file_type == "image":
            # ── Image path ────────────────────────────────────────────────────
            from src.data_processing.image_extractor import ClinicalImageExtractor
            extractor = ClinicalImageExtractor(use_gpu=use_gpu_ocr)
            img_result = extractor.extract(tmp_path)

            if not img_result.success:
                raise HTTPException(
                    status_code=422,
                    detail=f"Image extraction failed: {img_result.error}"
                )

            for table in img_result.tables:
                if not table.linearized:
                    continue
                pipeline_out = _run_pipeline(table.linearized)
                results.append(TableResult(
                    table_id=table.table_id,
                    table_text=table.linearized,
                    html=table.html,
                    json_data=table.json_data,
                    headers=table.headers,
                    **pipeline_out,
                ))

        else:
            # ── PDF / DOCX path ───────────────────────────────────────────────
            from src.data_processing.pdf_extractor import ClinicalPDFExtractor
            extractor = ClinicalPDFExtractor(tmp_path, use_gpu_ocr=use_gpu_ocr)
            pairs = extractor.extract_all()

            for pair in pairs:
                if not pair.table_text:
                    continue
                pipeline_out = _run_pipeline(pair.table_text)
                results.append(TableResult(
                    table_id=pair.pair_id,
                    table_text=pair.table_text,
                    html=None,   # PDF extraction doesn't produce HTML directly
                    json_data=None,
                    headers=None,
                    **pipeline_out,
                ))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("upload processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    total_ms = round((time.time() - start_total) * 1000, 2)

    return UploadResponse(
        filename=filename,
        file_type=file_type,
        tables_found=len(results),
        results=results,
        total_time_ms=total_ms,
    )


@app.post("/summarize/batch")
async def summarize_batch(tables: List[str]):
    """Batch summarize a list of linearized table strings."""
    if len(tables) > 50:
        raise HTTPException(status_code=400, detail="Batch limit is 50 tables")

    results = []
    for t in tables:
        try:
            out = _run_pipeline(t)
            results.append({"success": True, **out})
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    return {"results": results, "count": len(results)}


@app.get("/clusters")
async def list_clusters():
    """Return all 13 template clusters."""
    from src.generation.template_clusters import TEMPLATE_CLUSTERS
    return {
        cid: {
            "description": meta["description"],
            "required_slots": meta["required_slots"],
        }
        for cid, meta in TEMPLATE_CLUSTERS.items()
    }


@app.get("/features/names")
async def get_feature_names():
    return {"features": feature_extractor.get_feature_names(), "count": 32}


@app.post("/evaluate/single")
async def evaluate_single(
    prediction: str,
    reference: str,
    table_text: str = "",
):
    """Evaluate a single prediction vs reference."""
    from src.evaluation.eval_suite import evaluate_single as _eval
    return _eval(prediction, reference, source_table=table_text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
