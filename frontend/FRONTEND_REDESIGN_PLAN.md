# ClinicalSafe Frontend Redesign Plan

## 1. Product Goals
- Communicate the product clearly in under 10 seconds.
- Make the summarizer the primary action from every top-level page.
- Expose all backend capabilities: ML, DL Base, DL Fine-Tuned, and Compare.
- Build trust with visible verification metadata and model behavior notes.

## 2. UX Principles
- Clinical clarity over visual noise.
- High-contrast typography with restrained color coding for model states.
- Progressive disclosure: fast defaults, deeper detail on demand.
- Explicit system status for backend/API dependency.

## 3. Information Architecture
- `/`: Product narrative, outcomes, architecture highlights, CTA.
- `/summarizer`: Main app surface for table input and summary generation.
- `/research`: Methodology, benchmark framing, known limitations.
- `/about`: Project context, mission, roadmap, contributors.

## 4. Visual System
- Keep current "Clinical Ink" palette and typography stack.
- Promote consistent section rhythm using shared spacing tokens.
- Keep animation subtle and meaningful (state transitions, load states).
- Use semantic color coding:
  - ML: serum
  - DL Base: saline
  - DL Fine-Tuned: verified
  - Compare: plasma

## 5. Core User Flows

### Flow A: First-Time Visitor
1. Land on `/` and understand value proposition.
2. Open `/summarizer` from primary CTA.
3. Load example table and generate first result.

### Flow B: Evaluator / Researcher
1. Open `/research` to understand methodology and constraints.
2. Use Compare mode in `/summarizer`.
3. Inspect verification values and warnings.

### Flow C: Stakeholder Review
1. Read project framing in `/about`.
2. Review benchmark framing in `/research`.
3. Validate output behavior in `/summarizer`.

## 6. Summarizer Feature Spec
- Input panel:
  - Freeform table text area
  - Example loader
  - Drag-and-drop/plain file ingestion (text-based)
- Mode selector:
  - ML
  - DL Base
  - DL Fine-Tuned
  - Compare
- Output panel:
  - Single result card for non-compare modes
  - Multi-result layout in compare mode
  - Verification block (accuracy, warning list, status)
  - Processing and error states

## 7. Engineering Plan

### Phase 1 (Current)
- Fix broken route experience (`/research` missing page).
- Standardize shared component API (`SectionHeader` supports legacy and new props).
- Upgrade summarizer to full backend mode coverage.
- Tighten result and verification presentation.

### Phase 2
- Centralize static content in typed config objects.
- Add API health banner and retry UX.
- Add response export (copy/download markdown).

### Phase 3
- Add authenticated evaluation workspace.
- Add dataset upload queue and historical runs.
- Add benchmark visualization dashboards.

## 8. QA Checklist
- Route coverage: `/`, `/summarizer`, `/research`, `/about` all render.
- Summarizer modes map correctly:
  - ML -> `http://localhost:8000/summarize`
  - DL Base -> `http://localhost:8001/summarize` with `mode=base_dl`
  - DL Fine-Tuned -> `http://localhost:8001/summarize` with `mode=finetuned`
  - Compare -> `http://localhost:8001/summarize-compare`
- Error handling for unavailable backend.
- Responsive behavior at mobile, tablet, desktop.
- No TypeScript errors in frontend app files.
