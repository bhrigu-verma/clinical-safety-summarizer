# Clinical Narrative Generation: Architecture & Process Review

This document outlines the end-to-end architecture and machine learning pipeline we built to generate regulatory-grade, 0% hallucination clinical summaries from tabular safety data. It breaks down the process, the specific ML models used, and the rationale behind our design choices.

---

## 🏗️ At a Glance: The 3-Stage Hybrid Pipeline

We departed from using standard Large Language Models (like GPT-4 or standard T5) as end-to-end generators because LLMs are prone to "hallucinating" numbers or clinical facts. In regulatory medical writing, a 0% hallucination rate is non-negotiable. 

To achieve this, we built a **Deterministic ML Pipeline** divided into three precise stages:

1. **Stage 1: Content Selection** (What to say)
2. **Stage 2: Microplanning** (How to group it)
3. **Stage 3: Surface Realization** (How to say it factually)

*(Note: We also run a parallel 32-feature K-Nearest Neighbors (KNN) retrieval engine designed to find historical reports for stylistic matching and logging, but the factual generation is driven by the 3-stage engine below).*

---

## 🔍 Stage 1: Content Selection (The "Brain")
**Goal:** Decide which adverse events from a massive clinical table are "clinically significant" enough to be mentioned in the summary text, and which are just statistical noise.

**Models & Logic:**
*   **Feature Engineering:** We extract 6 core mathematical and clinical features for every row in the table:
    1. `drug_pct` (Incidence in experimental arm)
    2. `placebo_pct` (Incidence in control arm)
    3. `risk_diff` (Absolute percentage difference)
    4. `relative_risk` (Ratio of drug to placebo incidence)
    5. `is_sae` (Boolean: Is it a Serious Adverse Event or Fatal?)
    6. `is_severe` (Boolean: Is it Grade 3-4 / Severe?)
*   **The Model (LightGBM):** We use **LightGBM (Gradient Boosting)** as a binary classifier. It calculates a probability score (>0.5) to decide if an event should be included.
*   **Regulatory Override (Heuristic Guardrail):** If an event is flagged as an SAE (`is_sae = 1`) or Severe (`is_severe = 1`), it bypasses the ML model and is **forced** into the summary, ensuring we never miss a critical safety signal.
*   **How we trained it:** We generated synthetic data simulating human selection patterns. Since "salient" events are heavily imbalanced against "noise", we used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset before training the LightGBM booster.

---

## 🧩 Stage 2: Clinical Microplanning (The "Organizer")
**Goal:** Take the filtered list of significant events from Stage 1 and group them logically so they don't read like a robotic, comma-separated list.

**Models & Logic:**
*   **The Model (Agglomerative Clustering):** We use **Scikit-Learn's Agglomerative Clustering** (an unsupervised ML algorithm) using 'ward' linkage. 
*   **How it works:** It plots the events in mathematical space based on their 6 features. Events that have similar occurrence rates and risk profiles are grouped into the same cluster. 
*   **The Result:** Instead of writing "Nausea was 10%. Headache was 9%. Stroke was 0.5%", the ML groups the mild high-frequency events into one sentence array and the rare severe events into another.

---

## ✍️ Stage 3: Surface Realization (The "Writer")
**Goal:** Convert the mathematical clusters into fluent, human-readable English without allowing an AI to "invent" prose or hallucinate numbers.

**Models & Logic:**
*   **Deterministic Grammar Engine (Jinja2):** We do not use a generative LLM here. Instead, we use **Jinja2 templating**. 
*   **How it works:** The engine injects the highly structured, verified arrays from Stage 2 into a strict grammatical template (e.g., `"The most notable adverse events included {{ common_events }}. Serious adverse events were reported, specifically {{ serious_events }}."`).
*   **Why we did this:** By keeping the final text generation completely deterministic, we guarantee a **0.0% Hallucination Rate** and an exact 1.0 (100%) Numeric Accuracy match against the source table.

---

## 🔄 The Data Augmentation Process
To ensure our LightGBM model and parallel KNN engine were robust, we had to expand our initial dataset of 191 PDF-extracted pairs. 
We built an augmentation script (`scripts/datasynt.py` & `generate_synthetic_data.py`) that uses:
1.  **Numeric Variation (Tier 1):** Slightly shifting the percentages of real tables to create new mathematical boundaries.
2.  **Template Synthesis (Tier 2):** Generating mock tables with high severity vs. low severity events to teach the LightGBM classifier exactly what "noise" looks like.
This successfully expanded our robust training corpus to ~437 examples.

---

## 💡 Areas for Senior Feedback / Next Steps
*(Questions to ask your senior for further improvement)*

1.  **LightGBM Thresholds:** Currently, our probability cutoff for mentioning a common event is `0.5`. Should we be more aggressive (e.g., `0.3`) to include more events, or more conservative (e.g., `0.7`) for highly concise summaries?
2.  **Clustering Distance:** Our Agglomerative Clustering uses a `distance_threshold=4.0`. Do the resulting sentence groupings look natural to the medical writers, or should we adjust the mathematical distance?
3.  **Ontology Mapping:** Right now, we use Regex strings (e.g., `'fatal'` or `'grade 3'`) to assign the `is_sae` and `is_severe` flags. Should we invest in hooking this step up to a strict MedDRA API/Dictionary for perfect clinical term mapping?
## �� Roadmap to Maximize ROUGE Scores & Accuracy

To push our ROUGE scores to the maximum possible limit while maintaining our 100% Numeric Accuracy guardrails, we will pursue a two-pronged strategy:

### 1. Fine-Tuning the DL Model (Flan-T5-XL) for Stylistic ROUGE
Currently, our `src/api/main_dl.py` relies on a base DL model. To maximize ROUGE scores, we must heavily fine-tune a model specifically on our 437 augmented human-written pairs to mimic the exact vocabulary and phrase structure of regulatory writers.

*   **Model:** We will use `google/flan-t5-xl` (3B parameters) as it provides the best balance of instruction-following and grammar structure.
*   **Technique (QLoRA):** We will use QLoRA (Quantized Low-Rank Adaptation) via `bitsandbytes` (4-bit precision) to fine-tune the model on a single GPU. We will target the `q`, `v`, `k`, and `o` attention heads to maximize the model's ability to learn stylistic nuances. 
*   **Training Script:** Our script `src/training/finetune_dl.py` is already scaffolded for this. We need to execute it on an A100 or H100 instance.
*   **Why for ROUGE?** ROUGE measures word-overlap (n-grams). A fine-tuned Flan-T5 will learn the exact transitional phrases (e.g., "In contrast to...", "A clinically meaningful difference...") that human writers use, boosting our ROUGE-1 and ROUGE-L scores significantly above our current 0.35 baseline.

### 2. Upgrading the ML NLG Engine (The "ROUGE Booster" Mode)
Our current deterministic Jinja2 engine (`src/generation/clinical_nlg_engine.py`) guarantees 100% accuracy but produces slightly rigid text, which limits ROUGE scores when compared against fluid human writing. We can improve this without losing accuracy.

*   **Dynamic Template Expansion:** We will expand our `CSR_TEMPLATE` grammar into a dictionary of 15-20 different grammatical structures.
*   **KNN-Driven Template Selection (Hybrid Approach):** When an input table is received, we will run our 32-feature KNN engine (`src/retrieval/knn_retrieval_engine.py`) to find the most stylistically similar historical table. *Instead* of using that historical table's numbers (which causes hallucinations), we will identify *which grammar template* that historical table used, and instantly map our new 100%-accurate data points into that specific, human-like grammatical structure.
*   **Why for Accuracy?** The ML still controls the numbers and the selection (LightGBM). The KNN simply acts as a "stylistic router" to pick the most human-like sentence structure, bridging the gap between rigid data and fluid narrative. This pushes ROUGE up without ever risking a numeric hallucination.
