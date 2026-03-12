---
description: "Build a full-page research showcase that presents the ClinicalSafe project — pipeline architecture, data journey, model comparisons, and evaluation metrics — with scroll-triggered Framer Motion animations and glassmorphism cards."
mode: "agent"
agent: "ui-specialist"
---

# Research Showcase Page

Build a complete research showcase page at `frontend/app/research/page.tsx` that tells the story of the ClinicalSafe project through animated sections.

## Required Sections (in scroll order)

### 1. Hero Section
- Large gradient headline: "From Raw Tables to Regulatory Narratives"
- Animated subtitle explaining the system's purpose
- Floating glassmorphism stat cards showing key numbers:
  - **148** Pfizer CSRs processed
  - **1,411** training pairs (augmented)
  - **0%** hallucination target
  - **2** parallel pipelines
- Background: subtle animated gradient orbs (blue + purple)

### 2. Data Journey Timeline
- Vertical animated timeline showing the data pipeline:
  1. PDF Extraction (pdfplumber + custom ClinicalPDFExtractor)
  2. Table Linearization (start_table / end_table format)
  3. Gold Pair Creation (187 human-annotated pairs)
  4. Data Augmentation (→ 1,411 pairs via sentence shuffling, number perturbation, synonym replacement)
  5. Benchmark Creation (Tier 1 Gold + Tier 2 Stress tests)
- Each step animates in on scroll with staggered reveals
- Include small code snippets or format examples in monospace cards

### 3. Dual Pipeline Architecture
- Side-by-side animated comparison of ML vs DL pipelines:

**ML Pipeline (Blue theme)**:
- LightGBM Content Selector (precision 0.89)
- Agglomerative Clustering (Ward linkage)
- Template-based NLG via Jinja2
- Deterministic output — identical input always produces identical output

**DL Pipeline (Purple theme)**:
- Flan-T5-XXL (11B parameters)
- QLoRA Fine-tuning (4-bit quantization, LoRA r=64, alpha=128)
- 830MB adapter producing state-of-the-art summaries
- Beam search decoding with length penalty

- Animated flow arrows between components
- Each component card hover-expands to show detail

### 4. Hallucination Guardian
- Center-aligned section with shield icon animation
- Explain the 3-step verification: Parse → Extract → Compare
- Animated checklist of what gets verified:
  - Every number traces to source table
  - Percentage calculations validated
  - Patient group labels matched
  - Statistical significance preserved
- Show verified vs. failed example (green/red cards side-by-side)

### 5. Evaluation Metrics Dashboard
- Animated metric cards showing benchmark results:
  - ROUGE-1 / ROUGE-2 / ROUGE-L scores
  - Numeric accuracy percentage
  - Hallucination rate (with animated counter reaching 0%)
  - Inference time comparison (bar chart or visual)
- Use Framer Motion number counters that animate from 0 to final value

### 6. Tech Stack Grid
- Animated grid of technology cards, each with:
  - Icon (use Lucide where possible)
  - Name + version
  - One-line role description
- Categories: Frontend, Backend, ML/AI, Data Processing, Infrastructure
- Cards stagger-animate in a wave pattern

## Animation Requirements
- Every section uses `whileInView` with `viewport={{ once: true }}`
- Stagger children by 100ms
- Hero stats use spring animation for bounce effect
- Timeline dots pulse on scroll into view
- Metric counters animate numbers smoothly over 1.5s
- Background gradient orbs drift slowly with infinite animation

## File Structure
Create these files:
- `frontend/app/research/page.tsx` — main page (Server Component wrapper)
- `frontend/app/components/research/HeroSection.tsx`
- `frontend/app/components/research/DataTimeline.tsx`
- `frontend/app/components/research/PipelineComparison.tsx`
- `frontend/app/components/research/HallucinationGuard.tsx`
- `frontend/app/components/research/MetricsDashboard.tsx`
- `frontend/app/components/research/TechStackGrid.tsx`

Add a navigation link to `/research` in the existing layout header.
