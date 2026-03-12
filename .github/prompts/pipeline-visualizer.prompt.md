---
description: "Build an interactive pipeline visualization page that lets users explore the ML and DL architectures with animated flow diagrams, expandable component details, and live inference demos."
mode: "agent"
agent: "ui-specialist"
---

# Pipeline Visualizer Page

Create an interactive pipeline visualization page at `frontend/app/pipeline/page.tsx` that provides a deep technical dive into both processing pipelines.

## Page Structure

### 1. Pipeline Selector Header
- Tab-style selector between "ML Pipeline" and "DL Pipeline" using Radix Tabs
- Animated underline indicator that slides between tabs
- Each tab has its accent color (blue for ML, purple for DL)

### 2. ML Pipeline View (Blue Theme)

Animated vertical flow diagram with expandable nodes:

```
Input: Linearized Table
  ↓ (animated arrow)
ArmParser
  → Extracts treatment arms, headers, data cells
  → Detects table structure (efficacy vs safety)
  ↓
LightGBM Content Selector
  → Binary classifier trained on 1,411 pairs
  → Features: TF-IDF, positional, statistical
  → Precision: 0.89 | Recall: 0.91
  ↓
Agglomerative Clustering
  → Ward linkage method
  → Groups related safety findings
  → Dynamic cluster count based on content
  ↓
Template NLG (Jinja2)
  → 12 curated clinical narrative templates
  → Slot-filling with extracted values
  → Deterministic — same input = same output
  ↓
Hallucination Guardian
  → Parse numbers from output
  → Verify against source table
  → Flag any discrepancies
  ↓
Output: Verified Clinical Summary
```

Each node should be a `GlassCard` that:
- Animates in on scroll
- Shows title + one-line summary by default
- Expands on click to reveal full details, metrics, and example outputs
- Has a pulsing indicator showing data flow direction

### 3. DL Pipeline View (Purple Theme)

Same structure but with DL-specific components:

```
Input: Linearized Table
  ↓
Tokenization
  → SentencePiece tokenizer
  → Max 512 tokens input, 256 tokens output
  ↓
Flan-T5-XXL Backbone
  → 11 billion parameters
  → Encoder-decoder architecture
  → Pretrained on 1,800+ NLP tasks
  ↓
QLoRA Fine-tuning
  → 4-bit NormalFloat quantization
  → LoRA rank: 64, alpha: 128
  → Trainable params: ~830MB adapter
  → 3 epochs, batch size 2, gradient accumulation 8
  ↓
Beam Search Decoding
  → num_beams: 4
  → length_penalty: 1.2
  → no_repeat_ngram_size: 3
  → repetition_penalty: 1.3
  ↓
Hallucination Guardian
  → Same verification as ML pipeline
  ↓
Output: Verified Clinical Summary
```

### 4. Side-by-Side Comparison Panel
At the bottom, an animated comparison table showing:

| Aspect | ML Pipeline | DL Pipeline |
|--------|------------|-------------|
| Approach | Extractive + Template | End-to-end generative |
| Hallucination Risk | Near zero (deterministic) | Low (guardian-verified) |
| Fluency | Template-bound | Natural language |
| Speed | ~50ms | ~2-5s |
| Model Size | ~10MB | ~830MB adapter + 11B base |
| Training Data | 1,411 pairs | 1,411 pairs |

Table rows should animate in with stagger.

### 5. Live Demo Section
A mini version of the main page's input — paste a table, pick ML or DL, see the result with timing and verification status. Reuse components from the main page where possible.

## Animations
- Pipeline nodes animate in sequence (top to bottom) with 200ms stagger
- Connecting arrows draw themselves with SVG path animation
- Node expansion uses `layout` animation from Framer Motion for smooth height change
- Tab switch uses `AnimatePresence mode="wait"` for cross-fade
- Comparison table rows stagger-reveal from left

## Files to Create
- `frontend/app/pipeline/page.tsx`
- `frontend/app/components/pipeline/PipelineNode.tsx`
- `frontend/app/components/pipeline/PipelineArrow.tsx`
- `frontend/app/components/pipeline/ComparisonTable.tsx`

Add navigation link to `/pipeline` in the layout header.
