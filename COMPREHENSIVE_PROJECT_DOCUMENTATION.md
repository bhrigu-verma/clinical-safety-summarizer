# Clinical Safety Summarizer: Comprehensive Project Documentation

## What is this project? (Layman's Explanation)
Imagine you are a medical writer working for a major pharmaceutical company. Your job is to look at massive, complex tables full of clinical trial data—specifically regarding adverse events (side effects) that patients experienced—and write clear, accurate paragraphs summarizing that data for regulatory authorities like the FDA. This task is tedious, time-consuming, and highly prone to human error. If you write down the wrong percentage or attribute a severe side effect to the wrong drug arm, the consequences are disastrous.

Standard Artificial Intelligence tools, like ChatGPT, are great at writing fluid text but they have a fatal flaw: they "hallucinate." They will sometimes invent numbers or mix up data points to make the sentence sound better. In the highly regulated medical field, a hallucination rate of even 0.1% is completely unacceptable. Every single number must be mathematically perfectly accurate.

The Clinical Safety Summarizer is a specialized, dual-engine artificial intelligence system built to solve this exact problem. First, it uses a "deterministic" machine learning engine. This engine acts like a strict mathematician: it analyzes the data table, identifies the most important side effects (like severe or fatal events), and slots the numbers into pre-approved, rigid templates. It guarantees a 0.0% hallucination rate because it cannot invent anything; it purely extracts and formats.

However, text generated this way can sound slightly robotic. So, the system has an optional second phase. It passes this perfectly accurate, robotic text to an advanced "deep learning" model that has been trained to write like a human regulatory expert. This model rewrites the paragraph so it flows beautifully. But we don't just blindly trust it. Before the final text is shown to the user, a rigorous "Hallucination Guardian" double-checks every single number in the human-sounding text against the original table. If the Guardian finds even one tiny mistake, it rejects the AI's version and gives you the mathematically perfect, robotic version instead. The result is the holy grail for medical writers: blazing-fast drafts that are beautifully written and mathematically flawless.

---

## 1. Core Objective & Philosophy
This system converts structured **clinical safety tables** (specifically "Overview of Treatment-Emergent Adverse Events" tables from Pfizer Clinical Study Reports) into **natural language summaries** for regulatory submission formatted to **ICH E3 guidelines**.

The architecture revolves around achieving a **0.0% Hallucination Rate** while maintaining high **ROUGE** scores (narrative flow matching human experts) and **100% Numeric Accuracy**.

## 2. System Architecture
The system operates via two parallel backend APIs:
*   **ML Backend (Port 8000):** The deterministic, mathematics-first pipeline that guarantees factual safety.
*   **DL Backend (Port 8001):** The LLM-based rewrite engine (Flan-T5-XXL) that elevates the text to human-like regulatory prose.
*   **Next.js 15 Frontend:** Provides the user interface for PDF/Image uploads and reviewing the draft summaries.

### The Pipeline Process
1.  **Ingestion:** Tables are extracted from PDF, DOCX, or images and flattened into a linearized string format.
2.  **Arm Parsing (Stage 0):** Identifies the experimental vs. control arm using a dynamic keyword parser.
3.  **Content Selection (Stage 1):** Uses a LightGBM classifier to score all table rows and determine which adverse events are clinically significant enough to mention, with forced overrides for Severe or Fatal events.
4.  **Clinical Microplanning (Stage 2):** Unsupervised Agglomerative Clustering groups chosen events into natural sentence subsets.
5.  **Surface Realization (Stage 3):** Routes the grouped data into one of 14 strict grammatical templates (or uses KNN to retrieve a historical example), extracting numbers and embedding correct arm labels perfectly.
6.  **Deep Learning Polish (Phase 3):** Passes the drafted text to a fine-tuned LoRA LLM adapter for fluid rewriting.
7.  **Hallucination Guardian (Final Gate):** A hard-coded gate parses the final string. It validates every single number and arm attribution against the original table matrix. If it fails, the system reverts to the ML template.

## 3. Implemented Components Deep Dive

### 3.1 Data Ingestion & Formatting Modules
*   **Scrapers & Extractors:** `pfizzer_scraper.py` extracted the raw ~148 Pfizer CSR documents. The system uses `pdfplumber`, `PyMuPDF`, `camelot-py`, and `easyocr` recursively in `src/data_processing/pdf_extractor.py` and `image_extractor.py` to strip tables from documents.
*   **Linearized Table Strings:** The inputs are standardized into strings with specific delimiters for processing: `start_table [TABLE_TITLE: ...] [HEADERS: ...] [ROW] ... end_table`.

### 3.2 The ML Pipeline (Deterministic Engine)
*   **`ArmParser` (`src/generation/slot_fill_generator.py`):** Fixes attribution bugs by scanning headers for control keywords (`placebo`, `docetaxel`, `chemotherapy`, etc.) and assigning the correct column index to `experimental_arm_index`.
*   **Content Selector (`src/models/content_selector.py` & `scripts/train_content_selector.py`):**
    *   **Feature Extraction:** Generates 6 mathematical features per adverse event row (`drug_pct`, `placebo_pct`, `risk_diff`, `relative_risk`, `is_sae`, `is_severe`).
    *   **Model:** A LightGBM binary classifier trained with SMOTE to handle class imbalances, supplemented by synthetic data generation (`scripts/generate_synthetic_data.py`).
    *   **Regulatory Rules:** Hard overrides force the inclusion of Severe and Serious Adverse Events (SAEs) regardless of their probability score.
*   **Clinical Microplanning (`scripts/stage2.py`):**
    *   Uses `sklearn.cluster.AgglomerativeClustering` with Ward linkage and a distance threshold of 4.0 to logically group events (e.g., standard high-frequency mild events separately from rare fatal events) into distinct sentences.
*   **Template Clusters & Slot Extractors (`src/generation/template_clusters.py`):**
    *   14 different grammatical cluster templates (e.g., C01 standard, C13 single arm, C14 individual AE).
    *   `OverviewSlotExtractor`: Regular expressions map rows into named slots (e.g., `DR_SAE`, `G34`), ensuring absolute safety by dynamically inserting `experimental` and `control` columns as `{ARM_B}` and `{ARM_A}`.

### 3.3 The LLM Polish Layer (Deep Learning Engine)
*   **Model:** `google/flan-t5-xxl` (11 Billion parameters).
*   **Training Script:** `notebooks/02_train_t5xxl_optimized.py`.
*   **Methodology:** QLoRA Fine-Tuning. The model is 4-bit quantized (NF4, using `bitsandbytes`) to fit inside H100 hardware memory parameters. The LoRA adapter uses rank (`r`)=16 and relies on 8-bit AdamW optimizers.
*   **Prompting Strategy:** "Rewrite this clinical paragraph in regulatory medical tone following ICH E3 guidelines. Preserve ALL numbers exactly. Use passive voice. Report drug arm before placebo."
*   **Anti-Leakage Strategy:** The training applies random sentence shuffling 20% of the time so the LLM actually learns writing styles instead of just memorizing the deterministic template shapes.

### 3.4 Verification & Validation Mechanisms (The Hallucination Guardian)
*   **`HallucinationGuardian` (`src/generation/slot_fill_generator.py`):**
    *   **Numeric Verification:** Examines the final produced text. Validates that every float/integer (with a ±0.11 tolerance for valid rounding behavior) exists identically within the source table array. Exempts contextual text like `Table 3` or `0-5`.
    *   **Attribution Verification:** Identifies phrases like `"3.4% of subjects in the Chemotherapy arm"` and double-checks that `3.4` exists in the `Chemotherapy` column specifically.
    *   **Routing:** Operates as a boolean lock. If `False`, the API serves the Phase 1/2 Draft.

### 3.5 Evaluation Suite (`src/evaluation/eval_suite.py`)
Provides three core metrics evaluated either via Leave-One-Out (LOO) Cross Validation or Multi-Mode API matching:
1.  **ROUGE Scores (1, 2, L):** Determines narrative human-similarity via unigram/bigram string collision mapping. Currently ~0.35 on the deterministic side, with a target of >0.85 via the DL layer.
2.  **Numeric Accuracy:** Target is >0.96. The ML path currently executes this flawlessly at ~1.0.
3.  **Hallucination Rate:** Validates safety. Target is <0.04. ML Path guarantees 0.00.

## 4. Current State & Immediate Focus
The project has successfully resolved significant "v4.0" bugs relating to dynamic slot label misattribution (where percentage mappings would flip if the control arm was arbitrarily placed first/last).

The platform is largely in a pre-production/production-ready posture for ML generation. The immediate roadmap shifts towards scaling and validating the DL (`Flan-T5-XXL`) fine-tuning outputs to shatter the currently capped template ROUGE scores without inducing new hallucinations.
