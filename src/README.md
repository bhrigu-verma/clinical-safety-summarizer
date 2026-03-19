# Clinical ML Narrative Engine — v3.0
Complete backend for clinical safety table summarization.  
Supports **PDF / DOCX / JPEG / PNG** input → structured table + clinical narrative output.

## Evaluation Docs

- Full evaluation and metrics briefing: [RESEARCH_EVALUATION_CODEBASE_AGENT_BRIEFING.md](../RESEARCH_EVALUATION_CODEBASE_AGENT_BRIEFING.md)
- One-page execution runbook: [EVALUATION_QUICKSTART.md](../EVALUATION_QUICKSTART.md)

---

## Files Changed vs Original src_2.zip

| File | Status | What Changed |
|------|--------|--------------|
| `src/api/main_ml.py` | **REPLACED** | Added `/upload` endpoint (PDF/DOCX/JPEG/PNG), CORS, batch endpoint |
| `src/api/main_dl.py` | **REPLACED** | Added `/upload` endpoint, upgraded to Flan-T5-XL, hallucination verifier |
| `src/data_processing/pdf_extractor.py` | **REPLACED** | Added DOCX support, scanned PDF OCR fallback via PyMuPDF |
| `src/data_processing/image_extractor.py` | **NEW** | Full JPEG/PNG → table extraction (EasyOCR + img2table) |
| `src/training/finetune_dl.py` | **REPLACED** | Upgraded to Flan-T5-XL, proper 4-bit NF4 QLoRA, cosine LR |
| `scripts/generate_synthetic_data.py` | **NEW** | 4-tier data augmentation: number variation, template synthesis, LLM paraphrase, LLM synthesis |
| `scripts/setup_and_run.py` | **NEW** | One-command setup: extract → augment → build index → start servers |
| `requirements.txt` | **NEW** | All dependencies pinned |
| `docker-compose.yml` | **NEW** | Docker deployment for both backends |

**Unchanged** (copied as-is from src_2.zip):
- `src/feature_engineering/statistical_features.py`
- `src/generation/slot_fill_generator.py`
- `src/generation/template_clusters.py`
- `src/retrieval/knn_retrieval_engine.py`
- `src/models/ensemble.py`
- `src/models/generation_model.py`
- `src/evaluation/eval_suite.py`
- `src/data_processing/augmenter.py`
- `src/data_processing/splitter.py`

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
# GPU users: also install CUDA 12.1 + cuDNN 8
```

### Step 2 — One-command setup from your DOCX
```bash
python scripts/setup_and_run.py \
  --docx data/raw/Dataset_of_overall_safety_table_12_Aug_25__1_.docx
```

This will:
1. Extract all table-writeup pairs from the DOCX
2. Build the KNN retrieval index
3. Start both API servers

### Step 3 — Upload a file and get a summary
```bash
# Upload JPEG
curl -X POST http://localhost:8000/upload \
  -F "file=@my_table.jpg"

# Upload PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@clinical_report.pdf"

# Upload DOCX  
curl -X POST http://localhost:8000/upload \
  -F "file=@study_table.docx"
```

### Optional: Generate 1000 synthetic training examples
```bash
# Free (tiers 1+2 only, ~500 examples):
python scripts/generate_synthetic_data.py \
  --docx data/raw/Dataset_of_overall_safety_table_12_Aug_25__1_.docx \
  --output data/augmented/

# With LLM augmentation (~1000 examples, costs ~$5-10 in API credits):
python scripts/generate_synthetic_data.py \
  --docx data/raw/Dataset_of_overall_safety_table_12_Aug_25__1_.docx \
  --output data/augmented/ \
  --api-key YOUR_ANTHROPIC_API_KEY
```

### Optional: Fine-tune the DL model
```bash
# After generating synthetic data:
python src/training/finetune_dl.py
# Model saved to: models/flan_t5_xl_clinical/
```

---

## API Reference

### ML Backend — http://localhost:8000/docs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | **★ Main endpoint** — Upload PDF/DOCX/JPEG/PNG |
| `/summarize` | POST | Summarize a pre-extracted linearized table string |
| `/summarize/batch` | POST | Batch summarize up to 50 tables |
| `/health` | GET | Server health + index status |
| `/clusters` | GET | List all 13 template clusters |

### DL Backend — http://localhost:8001/docs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Same as ML but uses Flan-T5-XL |
| `/summarize` | POST | DL inference on linearized table |
| `/health` | GET | Model load status |
| `/model/load` | POST | Reload model after fine-tuning |

---

## Architecture

```
Upload (PDF/DOCX/JPEG/PNG)
        ↓
[image_extractor.py / pdf_extractor.py]
  EasyOCR + img2table (images)
  pdfplumber → PyMuPDF OCR fallback (PDFs)
  python-docx (DOCX)
        ↓
  Linearized Table String
        ↓
[3-Stage ML Pipeline]
  Stage 1: StatisticalFeatureExtractor (32 features)
  Stage 2: KNNRetrievalEngine (cosine similarity, k=5)
  Stage 3: SlotFillGenerator + HallucinationGuardian
        ↓
  Verified Clinical Narrative
```
