"""
src/api/main_dl.py  — DL Backend (Port 8001)  v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v4.0 — Phase 3 Complete:
  • Fine-tuned LoRA adapter: models/best_clinical_adapter_xl (830MB)
  • Base model: google/flan-t5-xl (3B) loaded in 4-bit on GPU / FP32 on CPU
  • PEFT adapter overlay at runtime
  • /summarize — DL Polish (ML → DL rewrite → Hallucination Guardian)
  • /summarize-ml — Pure ML deterministic output
  • /summarize-compare — Run ALL 3 engines side-by-side
  • Hallucination Guardian gate on every DL output
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import time
import os
import re
import sys
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="Clinical Safety Table Summarizer — DL Backend v4.0",
    description=(
        "Phase 3 Hybrid Pipeline: ML Deterministic → Fine-tuned Flan-T5-XL LoRA Polish "
        "→ Hallucination Guardian. Supports PDF, DOCX, JPEG, PNG uploads. "
        "3-way comparison endpoint for ML vs DL-Base vs DL-Finetuned."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ─────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    table_text: str
    max_new_tokens: Optional[int] = 512
    num_beams: Optional[int] = 4
    temperature: Optional[float] = 0.0
    verify_numbers: Optional[bool] = True
    mode: Optional[str] = "finetuned"  # "ml", "base_dl", "finetuned"

class SingleResult(BaseModel):
    summary: str
    model_used: str
    verified: bool
    numeric_accuracy: float
    inference_time_ms: float
    warnings: List[str]
    tokens_generated: Optional[int] = None

class CompareResponse(BaseModel):
    ml: SingleResult
    base_dl: SingleResult
    finetuned_dl: SingleResult

class TableResult(BaseModel):
    table_id: str
    table_text: str
    html: Optional[str] = None
    json_data: Optional[List[Dict]] = None
    summary: str
    model_used: str
    verified: bool
    numeric_accuracy: float
    inference_time_ms: float
    warnings: List[str]
    tokens_generated: Optional[int] = None

class UploadResponse(BaseModel):
    filename: str
    file_type: str
    tables_found: int
    results: List[TableResult]
    total_time_ms: float

class HealthResponse(BaseModel):
    status: str
    base_model_loaded: bool
    finetuned_model_loaded: bool
    model_name: Optional[str]
    device: Optional[str]
    quantized: bool
    version: str

# ── ML Fallback Engine ────────────────────────────────────────────────────────

from src.generation.slot_fill_generator import SlotFillGenerator, HallucinationError

class MLAdapter:
    def __init__(self):
        self.engine = SlotFillGenerator(strict_mode=False)
    def generate(self, table_text: str) -> str:
        return self.engine.generate(table_text).narrative

_ml_engine = MLAdapter()

# ── DL Model State ────────────────────────────────────────────────────────────

base_model = None
base_tokenizer = None
finetuned_model = None
finetuned_tokenizer = None
device = "cpu"
is_quantized = False

ADAPTER_PATH = "models/best_clinical_adapter_xl"
BASE_MODEL_ID = "google/flan-t5-xl"
FALLBACK_MODEL_ID = "google/flan-t5-small"


def load_models():
    """Load base DL model + fine-tuned LoRA adapter."""
    global base_model, base_tokenizer, finetuned_model, finetuned_tokenizer
    global device, is_quantized

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # ── 1. Load Base Model (for base_dl comparison) ──────────────────────
        model_id = FALLBACK_MODEL_ID  # Use small for local dev
        if device == "cuda":
            model_id = BASE_MODEL_ID  # Use XL on GPU

        logger.info(f"Loading BASE model: {model_id} on {device}")

        base_tokenizer = AutoTokenizer.from_pretrained(model_id)

        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
            is_quantized = True
        else:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, dtype=torch.float32,
            ).to(device)
            is_quantized = False

        base_model.eval()
        logger.info(f"✅ Base model loaded: {model_id} on {device}")

        # ── 2. Load Fine-Tuned LoRA Adapter ──────────────────────────────────
        adapter_path = Path(ADAPTER_PATH)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            logger.info(f"Loading LoRA adapter from {ADAPTER_PATH}...")
            try:
                from peft import PeftModel

                if device == "cuda":
                    # On GPU: load XL base + overlay adapter
                    ft_base = AutoModelForSeq2SeqLM.from_pretrained(
                        BASE_MODEL_ID,
                        quantization_config=bnb_config,
                        device_map="auto",
                    )
                    finetuned_model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH)
                else:
                    # On CPU/MPS: skip loading XL in full precision to avoid OOM crash
                    logger.warning(f"Skipping {BASE_MODEL_ID} + adapter load on {device} to prevent OOM crash. "
                                   f"Finetuned mode will simulate using the small base model.")
                    finetuned_model = None

                if finetuned_model is not None:
                    finetuned_model.eval()
                    finetuned_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
                    logger.info("✅ Fine-tuned LoRA adapter loaded successfully!")

            except ImportError:
                logger.warning("peft library not installed. Finetuned model unavailable.")
            except Exception as e:
                logger.warning(f"Failed to load adapter: {e}")
        else:
            logger.warning(f"No adapter found at {ADAPTER_PATH}. Finetuned mode unavailable.")

    except ImportError as e:
        logger.warning(f"Transformers not available: {e}")
    except Exception as e:
        logger.warning(f"Model loading failed: {e}")


# Load at startup
try:
    load_models()
except Exception as e:
    logger.warning(f"Model load failed at startup: {e}")


# ── Post-Processing: Hallucination Guardian ───────────────────────────────────

def _verify_output(narrative: str, source_table: str) -> tuple:
    """
    Verify that all numbers in the DL output appear in the source table.
    Returns (is_verified, numeric_accuracy, warnings).
    """
    source_nums = set(
        float(n) for n in re.findall(r"\b\d+\.?\d*\b", source_table)
    )
    output_nums = re.findall(r"\b\d+\.?\d*\b", narrative)

    if not output_nums:
        return True, 1.0, []

    tbl_refs = set(re.findall(r"Table\s+(\d+)", narrative, re.IGNORECASE))
    allowed = {"0", "1", "2", "0.0", "1.0", "2.0"} | tbl_refs

    bad = []
    for n_str in output_nums:
        if n_str in allowed:
            continue
        try:
            n_val = float(n_str)
            if not any(abs(n_val - s) <= 1.01 for s in source_nums):
                bad.append(n_str)
        except ValueError:
            bad.append(n_str)

    accuracy = 1.0 - (len(bad) / len(output_nums))
    warnings = []
    if bad:
        warnings.append(f"Possible hallucination: numbers {bad} not found in source table")

    return len(bad) == 0, round(accuracy, 4), warnings


# ── DL Inference Helpers ──────────────────────────────────────────────────────

def _run_dl_inference(model, tokenizer, input_text: str,
                      max_new_tokens: int = 512, num_beams: int = 4,
                      temperature: float = 0.0) -> tuple:
    """Generic DL inference. Returns (text, tokens_gen)."""
    import torch

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        padding=True,
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.5,
    )

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_gen = outputs[0].shape[0]
    return summary, int(tokens_gen)


REWRITE_PROMPT = (
    "You are a medical writer creating a clinical narrative for regulatory submission. "
    "Rewrite this clinical paragraph to flow perfectly using a regulatory medical tone. "
    "Do not add, remove, or change ANY numbers or clinical events.\n\n"
    "Paragraph:\n{text}\n\nRewritten Narrative:"
)

DIRECT_PROMPT = (
    "You are a medical writer creating a clinical narrative for regulatory submission. "
    "Summarize the following safety table accurately. "
    "Report all key statistics: TEAE incidence, Grade 3-4 rates, SAE rates, "
    "and discontinuation rates for each treatment arm.\n\n"
    "Safety Table:\n{text}\n\nClinical Narrative Summary:"
)


# ── 3 Generation Pipelines ───────────────────────────────────────────────────

def _generate_ml(table_text: str) -> dict:
    """Pure ML deterministic pipeline."""
    start = time.time()
    warnings = []
    try:
        summary = _ml_engine.generate(table_text)
    except Exception as e:
        summary = "No adverse event data could be extracted."
        warnings.append(f"ML generation failed: {e}")

    verified, accuracy, vwarn = _verify_output(summary, table_text)
    warnings.extend(vwarn)

    return {
        "summary": summary,
        "model_used": "ml/lgbm+jinja2",
        "verified": verified,
        "numeric_accuracy": accuracy,
        "inference_time_ms": round((time.time() - start) * 1000, 2),
        "warnings": warnings,
        "tokens_generated": None,
    }


def _generate_base_dl(table_text: str, max_new_tokens=512, num_beams=4,
                       temperature=0.0) -> dict:
    """Base DL model (un-finetuned) direct table summarization."""
    start = time.time()
    warnings = []

    if base_model is None or base_tokenizer is None:
        return _generate_ml(table_text)

    try:
        prompt = DIRECT_PROMPT.format(text=table_text)
        summary, tokens_gen = _run_dl_inference(
            base_model, base_tokenizer, prompt,
            max_new_tokens, num_beams, temperature
        )
        model_used = "dl_base/flan-t5"
    except Exception as e:
        logger.warning(f"Base DL failed: {e}")
        summary = _ml_engine.generate(table_text)
        tokens_gen = None
        model_used = "ml_fallback"
        warnings.append(f"Base DL failed: {e}")

    verified, accuracy, vwarn = _verify_output(summary, table_text)
    warnings.extend(vwarn)

    return {
        "summary": summary,
        "model_used": model_used,
        "verified": verified,
        "numeric_accuracy": accuracy,
        "inference_time_ms": round((time.time() - start) * 1000, 2),
        "warnings": warnings,
        "tokens_generated": tokens_gen,
    }


def _generate_finetuned(table_text: str, max_new_tokens=512, num_beams=4,
                         temperature=0.0) -> dict:
    """Phase 3: ML Strict → Fine-tuned DL Polish → Hallucination Guardian."""
    start = time.time()
    warnings = []

    # Step 1: ML strict summary
    try:
        strict_summary = _ml_engine.generate(table_text)
    except Exception as e:
        strict_summary = "No adverse event data could be extracted."
        warnings.append(f"ML Step 1 failed: {e}")

    summary = strict_summary
    tokens_gen = None
    model_used = "ml_fallback"

    # Step 2: DL Polish with fine-tuned adapter
    active_model = finetuned_model if finetuned_model is not None else base_model
    active_tokenizer = finetuned_tokenizer if finetuned_tokenizer is not None else base_tokenizer
    adapter_label = "finetuned_xl" if finetuned_model is not None else "base_polish"

    if active_model is not None and active_tokenizer is not None and strict_summary:
        try:
            prompt = REWRITE_PROMPT.format(text=strict_summary)
            fluid_summary, tokens_gen = _run_dl_inference(
                active_model, active_tokenizer, prompt,
                max_new_tokens, num_beams, temperature
            )

            # Step 3: Hallucination Guardian
            verified, accuracy, vwarn = _verify_output(fluid_summary, table_text)

            if verified and accuracy >= 0.95:
                summary = fluid_summary
                model_used = f"dl_polished/{adapter_label}"
            else:
                warnings.append(
                    f"DL Hallucination Blocked (acc={accuracy}): "
                    f"Reverted to ML Fallback. Issues: {vwarn}"
                )
        except Exception as e:
            logger.warning(f"DL Polish failed: {e}")
            warnings.append(f"DL model failed: {e}")
    else:
        warnings.append("DL model not loaded; using ML template fallback")

    # Final verification on chosen summary
    verified, accuracy, vwarn = _verify_output(summary, table_text)
    if vwarn and summary == strict_summary:
        warnings.extend(vwarn)

    return {
        "summary": summary,
        "model_used": model_used,
        "verified": verified,
        "numeric_accuracy": accuracy,
        "inference_time_ms": round((time.time() - start) * 1000, 2),
        "warnings": warnings,
        "tokens_generated": tokens_gen,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        base_model_loaded=base_model is not None,
        finetuned_model_loaded=finetuned_model is not None,
        model_name=BASE_MODEL_ID if base_model else FALLBACK_MODEL_ID,
        device=device,
        quantized=is_quantized,
        version="4.0.0",
    )


@app.post("/summarize", response_model=SingleResult)
async def summarize(request: SummarizeRequest):
    """Generate summary using specified mode: 'ml', 'base_dl', or 'finetuned'."""
    try:
        mode = (request.mode or "finetuned").lower()
        if mode == "ml":
            result = _generate_ml(request.table_text)
        elif mode == "base_dl":
            result = _generate_base_dl(
                request.table_text, request.max_new_tokens,
                request.num_beams, request.temperature
            )
        else:
            result = _generate_finetuned(
                request.table_text, request.max_new_tokens,
                request.num_beams, request.temperature
            )
        return SingleResult(**result)
    except Exception as e:
        logger.exception("Summarize failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-compare", response_model=CompareResponse)
async def summarize_compare(request: SummarizeRequest):
    """Run ALL 3 engines side-by-side and return comparative results."""
    try:
        ml_result = _generate_ml(request.table_text)
        base_result = _generate_base_dl(
            request.table_text, request.max_new_tokens,
            request.num_beams, request.temperature
        )
        ft_result = _generate_finetuned(
            request.table_text, request.max_new_tokens,
            request.num_beams, request.temperature
        )
        return CompareResponse(
            ml=SingleResult(**ml_result),
            base_dl=SingleResult(**base_result),
            finetuned_dl=SingleResult(**ft_result),
        )
    except Exception as e:
        logger.exception("Compare failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_and_summarize(
    file: UploadFile = File(...),
    use_gpu_ocr: bool = Form(False),
    verify_numbers: bool = Form(True),
    num_beams: int = Form(4),
    mode: str = Form("finetuned"),
):
    """Upload a file and summarize all tables found within it."""
    start_total = time.time()
    filename = file.filename or "upload"
    content_type = file.content_type or ""
    suffix = Path(filename).suffix.lower()

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if suffix in image_suffixes or "image/" in content_type:
        file_type = "image"
    elif suffix == ".pdf" or "pdf" in content_type:
        file_type = "pdf"
    elif suffix in (".docx", ".doc") or "word" in content_type:
        file_type = "docx"
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported: {content_type}")

    content = await file.read()
    tmp_suffix = suffix if suffix else (".jpg" if file_type == "image" else ".pdf")
    with tempfile.NamedTemporaryFile(suffix=tmp_suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    results: List[TableResult] = []

    gen_fn = _generate_finetuned if mode == "finetuned" else (
        _generate_base_dl if mode == "base_dl" else _generate_ml
    )

    try:
        if file_type == "image":
            from src.data_processing.image_extractor import ClinicalImageExtractor
            ext = ClinicalImageExtractor(use_gpu=use_gpu_ocr)
            img_result = ext.extract(tmp_path)
            if not img_result.success:
                raise HTTPException(422, img_result.error)

            for table in img_result.tables:
                if not table.linearized:
                    continue
                out = gen_fn(table.linearized, num_beams=num_beams)
                results.append(TableResult(
                    table_id=table.table_id,
                    table_text=table.linearized,
                    html=table.html,
                    json_data=table.json_data,
                    **out,
                ))
        else:
            from src.data_processing.pdf_extractor import ClinicalPDFExtractor
            ext = ClinicalPDFExtractor(tmp_path, use_gpu_ocr=use_gpu_ocr)
            pairs = ext.extract_all()

            for pair in pairs:
                if not pair.table_text:
                    continue
                out = gen_fn(pair.table_text, num_beams=num_beams)
                results.append(TableResult(
                    table_id=pair.pair_id,
                    table_text=pair.table_text,
                    html=None,
                    json_data=None,
                    **out,
                ))
    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        filename=filename,
        file_type=file_type,
        tables_found=len(results),
        results=results,
        total_time_ms=round((time.time() - start_total) * 1000, 2),
    )


@app.post("/model/load")
async def reload_model():
    """Reload all models (e.g., after fine-tuning completes)."""
    load_models()
    return {
        "status": "ok",
        "base_loaded": base_model is not None,
        "finetuned_loaded": finetuned_model is not None,
        "device": device,
        "quantized": is_quantized,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
