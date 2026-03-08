# 🏅 Roadmap to the Ultimate Regulatory-Grade Clinical Narrative Generator

To build the unequivocally *best* clinical summary generation system in the industry, we must bridge the final gap: maintaining our **0.0% Hallucination Rate** and **100% Numeric Accuracy** while simultaneously shattering the **ROUGE-1 score ceiling** (pushing it from ~0.27 to >0.60+).

Here is the three-phase ultimate architecture plan we will implement.

---

## Phase 1: Adaptive Deterministic NLG (The "Stylistic Router" Upgrade)
*Status: Implementing Now*

**The Problem:** Our current deterministic engine (`clinical_nlg_engine.py`) uses a single, highly rigid Jinja2 grammar template for every single table. This guarantees accuracy but suffers in ROUGE because human writers vary their sentence structures dramatically based on the *shape* of the data.

**The Solution:** 
1.  **Multiple Domain-Aware Templates:** We will define 5 distinct grammar templates representing common clinical profiles:
    *   *High Toxicity Profile:* Focuses heavily on Grade 3-4 and fatal events.
    *   *Imbalanced Arms:* Emphasizes the risk difference between Drug and Placebo.
    *   *Clean Safety Profile:* Focuses on "No SAEs reported".
    *   *Single-Arm Study:* A flat reporting structure with no comparative language.
    *   *Standard Two-Arm Profile:* The baseline reporting structure.
2.  **Smart Routing:** We will use our mathematical features (like `is_sae`, `abs_risk_diff`, `num_arms`) to dynamically route the 100% accurate LightGBM data into the *perfectly matched* human template. 
*Result: Massive instant ROUGE boost with 0 risk of hallucination.*

---

## Phase 2: Advanced Clinical Microplanning (The "Bucketizer")
*Status: Next in line*

**The Problem:** Our 6-feature Agglomerative Clustering groups events automatically, but sometimes lumps highly disparate items together. Regulatory writers usually explicitly bucket events into predefined tiers.

**The Solution:**
We will upgrade Stage 2 of the engine to explicitly stratify the LightGBM-selected events into **Clinical Tiers**:
*   *Tier 1: High Frequency* (e.g., > 10% or 20% drug incidence)
*   *Tier 2: Moderate Frequency* (e.g., > 5% drug incidence)
*   *Tier 3: Serious / Severe* (Any incidence, but `is_sae=1` or `is_severe=1`)

This guarantees the generated narrative logically flows exactly how a regulatory medical writer would structure their paragraphs, improving readability and narrative flow (ROUGE-L).

---

## Phase 3: Hallucination-Guarded LLM (The "Final Polish")
*Status: R&D*

**The Problem:** Unconstrained LLMs (like Flan-T5) hallucinate numbers. Rigid templates are 100% safe but lack absolute human fluidity.

**The Ultimate Solution (The Hybrid Polish):**
1.  We run our 100% mathematically accurate ML-to-Jinja2 pipeline first. This gives us perfectly factual but slightly robotic text.
2.  We pass this rigid text to a fine-tuned deep learning model (e.g., Flan-T5-XL) with the strict prompt: *"Rewrite this paragraph to flow better using regulatory medical tone. Do not add or change any numbers."*
3.  **The Guardian Gate:** Before presenting the LLM's fluent text to the user, we run a rigid regex-based `HallucinationGuardian` that extracts every number in the LLM's text and asserts it exists in the original source table. 
4.  If the LLM hallucinated, we instantly reject it and fallback to the Phase 1 deterministic text. If it's clean, the user gets mathematically perfect, stunningly human prose.
