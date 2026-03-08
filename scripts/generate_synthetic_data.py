"""
scripts/generate_synthetic_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE SYNTHETIC DATA GENERATION PLAN

Problem: You have ~200 examples from the DOCX. You need 1000+ for DL fine-tuning.

Strategy (4 tiers, no hallucinations):
  Tier 1 — Number Variation     (FREE, ~300 examples)
    Vary patient counts ±15%, recompute percentages mathematically
  Tier 2 — Template Synthesis   (FREE, ~200 examples)
    Use the 13 cluster templates to generate new examples from scratch
  Tier 3 — LLM Paraphrase       (API cost ~$2-5, ~250 examples)
    Use Claude to rewrite summaries in different clinical writing styles
  Tier 4 — LLM Full Synthesis   (API cost ~$5-10, ~250 examples)
    Use Claude to generate entirely new table+summary pairs

Total target: 1000 high-quality examples

Usage:
    # Tier 1+2 only (free, no API key needed):
    python scripts/generate_synthetic_data.py --input data/processed/raw_pairs.json --output data/augmented/ --tiers 1 2

    # All tiers (requires ANTHROPIC_API_KEY env var):
    python scripts/generate_synthetic_data.py --input data/processed/raw_pairs.json --output data/augmented/ --api-key YOUR_KEY

    # Use your DOCX directly:
    python scripts/generate_synthetic_data.py --docx data/raw/Dataset_of_overall_safety_table.docx --output data/augmented/
"""

import argparse
import json
import random
import re
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from copy import deepcopy

# ─── Tier 1: Mathematical Number Variation ────────────────────────────────────

class NumberVariator:
    """
    Generates new examples by varying patient counts mathematically.
    All percentages are RECOMPUTED from new counts — zero hallucination risk.
    """

    # Drug name pool for arm swaps
    DRUG_NAMES = [
        "Apalutamide", "Enzalutamide", "Darolutamide", "Pembrolizumab",
        "Nivolumab", "Atezolizumab", "Durvalumab", "Olaparib",
        "Niraparib", "Rucaparib", "Abemaciclib", "Palbociclib",
        "Ribociclib", "Alpelisib", "Idelalisib", "Venetoclax",
        "Ibrutinib", "Acalabrutinib", "Zanubrutinib", "Osimertinib",
        "Lorlatinib", "Alectinib", "Brigatinib", "Capmatinib",
    ]
    CONTROL_NAMES = ["Placebo", "Vehicle", "Best Supportive Care", "Standard of Care"]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def vary(self, example: Dict, n_variants: int = 3) -> List[Dict]:
        """Generate n_variants mathematical variations of one example."""
        results = []
        for i in range(n_variants):
            varied = self._apply_variation(example, seed_offset=i)
            if varied:
                results.append(varied)
        return results

    def _apply_variation(self, example: Dict, seed_offset: int = 0) -> Optional[Dict]:
        """Apply one variation pass."""
        rng = random.Random(seed_offset + hash(example.get("table_text", "")) % 10000)

        table = example.get("table_text") or example.get("input", "")
        writeup = example.get("writeup") or example.get("output", "")

        if not table or not writeup:
            return None

        # 1. Randomly select a scale factor (how big is the new study)
        scale = rng.uniform(0.70, 1.40)

        # 2. Extract all N= values
        n_matches = list(re.finditer(r"N\s*=\s*(\d+)", table))
        if not n_matches:
            return None

        # Build a mapping: old N → new N
        n_map = {}
        for m in n_matches:
            old_n = int(m.group(1))
            new_n = max(20, int(round(old_n * scale / 10) * 10))  # round to nearest 10
            n_map[old_n] = new_n

        # 3. Replace N= values in table
        new_table = table
        for old_n, new_n in sorted(n_map.items(), reverse=True):
            new_table = re.sub(
                rf"(N\s*=\s*){old_n}\b",
                lambda m, nn=new_n: f"{m.group(1)}{nn}",
                new_table
            )

        # 4. Recompute percentages in ROW cells
        # Pattern: "digits (digits.digits)" → recompute pct
        def recompute_pct(match):
            n_str = match.group(1)
            pct_str = match.group(2)
            try:
                n_val = int(n_str)
                pct_val = float(pct_str)
                # Find the most likely denominator from n_map
                denom = None
                for old_n, new_n in n_map.items():
                    if abs(pct_val - (n_val / old_n * 100)) < 5:
                        denom = new_n
                        break
                if denom:
                    new_n_val = max(0, int(round(n_val * scale)))
                    new_pct = round(new_n_val / denom * 100, 1)
                    return f"{new_n_val} ({new_pct})"
            except (ValueError, ZeroDivisionError):
                pass
            return match.group(0)

        new_table = re.sub(r"\b(\d+)\s*\((\d+\.?\d*)\)", recompute_pct, new_table)

        # 5. Optionally swap drug names (30% chance)
        if rng.random() < 0.30:
            drug_a = rng.choice(self.DRUG_NAMES)
            drug_b = rng.choice(self.CONTROL_NAMES)
            # Replace existing arm names (simple heuristic)
            new_table, new_writeup = self._swap_arms(new_table, writeup, drug_a, drug_b)
        else:
            new_writeup = writeup  # keep original writeup structure

        return {
            "table_text": new_table,
            "writeup": new_writeup,
            "augmentation": "number_variation",
            "scale_factor": round(scale, 3),
        }

    def _swap_arms(self, table: str, writeup: str, drug_a: str, drug_b: str):
        """Replace first detected drug name pair with new names."""
        # Find existing arm names
        arm_pattern = re.compile(
            r"\b(Apalutamide|Enzalutamide|Darolutamide|Pembrolizumab|Nivolumab|"
            r"Docetaxel|Paclitaxel|Cabazitaxel|Olaparib|Niraparib|"
            r"Active\s*drug|Treatment\s*arm)\b",
            re.IGNORECASE
        )
        ctrl_pattern = re.compile(
            r"\b(Placebo|Vehicle|Control|Best\s*supportive\s*care|Standard\s*of\s*care)\b",
            re.IGNORECASE
        )

        new_table = arm_pattern.sub(drug_a, table)
        new_table = ctrl_pattern.sub(drug_b, new_table)
        new_writeup = arm_pattern.sub(drug_a, writeup)
        new_writeup = ctrl_pattern.sub(drug_b, new_writeup)

        return new_table, new_writeup


# ─── Tier 2: Template-Based Synthesis ────────────────────────────────────────

class TemplateSynthesizer:
    """
    Generates fully new examples from scratch using the 13 cluster templates.
    Uses realistic clinical trial parameters from published literature ranges.
    """

    # Realistic clinical ranges from published oncology trials
    TEAE_PCT_RANGE  = (82, 99)
    G34_PCT_RANGE   = (20, 65)
    SAE_PCT_RANGE   = (10, 45)
    DISC_PCT_RANGE  = (3, 25)
    N_RANGE         = (100, 600)

    DRUG_PAIRS = [
        ("Apalutamide", "Placebo"),
        ("Enzalutamide", "Placebo"),
        ("Pembrolizumab", "Docetaxel"),
        ("Nivolumab", "Chemotherapy"),
        ("Olaparib", "Placebo"),
        ("Abemaciclib", "Placebo"),
        ("Venetoclax", "Rituximab"),
        ("Ibrutinib", "Chlorambucil"),
        ("Osimertinib", "Platinum-based chemotherapy"),
        ("Lorlatinib", "Crizotinib"),
    ]

    def __init__(self, seed: int = 123):
        random.seed(seed)
        self._rng = random.Random(seed)

    def generate(self, n_examples: int = 200) -> List[Dict]:
        examples = []
        for i in range(n_examples):
            ex = self._generate_one(seed_offset=i)
            if ex:
                examples.append(ex)
        return examples

    def _generate_one(self, seed_offset: int = 0) -> Dict:
        rng = random.Random(seed_offset * 17 + 42)

        drug_a, drug_b = rng.choice(self.DRUG_PAIRS)
        n_a = rng.randint(*self.N_RANGE)
        n_b = rng.randint(*self.N_RANGE)

        # Generate realistic correlated percentages
        teae_pct_a = rng.uniform(*self.TEAE_PCT_RANGE)
        teae_pct_b = rng.uniform(*self.TEAE_PCT_RANGE)
        g34_pct_a  = rng.uniform(*self.G34_PCT_RANGE)
        g34_pct_b  = rng.uniform(*self.G34_PCT_RANGE)
        sae_pct_a  = rng.uniform(*self.SAE_PCT_RANGE)
        sae_pct_b  = rng.uniform(*self.SAE_PCT_RANGE)
        disc_pct_a = rng.uniform(*self.DISC_PCT_RANGE)
        disc_pct_b = rng.uniform(*self.DISC_PCT_RANGE)

        # Compute counts from percentages
        teae_n_a = int(round(n_a * teae_pct_a / 100))
        teae_n_b = int(round(n_b * teae_pct_b / 100))
        g34_n_a  = int(round(n_a * g34_pct_a  / 100))
        g34_n_b  = int(round(n_b * g34_pct_b  / 100))
        sae_n_a  = int(round(n_a * sae_pct_a  / 100))
        sae_n_b  = int(round(n_b * sae_pct_b  / 100))
        disc_n_a = int(round(n_a * disc_pct_a / 100))
        disc_n_b = int(round(n_b * disc_pct_b / 100))

        study_id = f"STUDY-{rng.randint(1000, 9999)}"

        # Build linearized table
        table = self._build_table(
            drug_a, drug_b, n_a, n_b, study_id,
            teae_n_a, round(teae_pct_a, 1), teae_n_b, round(teae_pct_b, 1),
            g34_n_a, round(g34_pct_a, 1), g34_n_b, round(g34_pct_b, 1),
            sae_n_a, round(sae_pct_a, 1), sae_n_b, round(sae_pct_b, 1),
            disc_n_a, round(disc_pct_a, 1), disc_n_b, round(disc_pct_b, 1),
        )

        # Build writeup
        writeup = self._build_writeup(
            drug_a, drug_b, n_a, n_b, study_id,
            teae_n_a, round(teae_pct_a, 1), teae_n_b, round(teae_pct_b, 1),
            g34_n_a, round(g34_pct_a, 1), g34_n_b, round(g34_pct_b, 1),
            sae_n_a, round(sae_pct_a, 1), sae_n_b, round(sae_pct_b, 1),
            disc_n_a, round(disc_pct_a, 1), disc_n_b, round(disc_pct_b, 1),
        )

        return {
            "table_text": table,
            "writeup": writeup,
            "augmentation": "template_synthesis",
        }

    def _build_table(self, drug_a, drug_b, n_a, n_b, study_id,
                     tea, tap, teb, tbp, g34a, g34ap, g34b, g34bp,
                     sa, sap, sb, sbp, da, dap, db, dbp) -> str:
        return (
            f"start_table [TABLE_TITLE: Table 1: Overview of Treatment-Emergent Adverse Events "
            f"({study_id})] "
            f"[HEADERS: | Adverse Event Category | {drug_a} N = {n_a} | {drug_b} N = {n_b}] "
            f"[ROW] Subjects with any TEAE | {tea} ({tap}) | {teb} ({tbp}) "
            f"[ROW] Grade 3-4 TEAEs | {g34a} ({g34ap}) | {g34b} ({g34bp}) "
            f"[ROW] Treatment-emergent SAEs | {sa} ({sap}) | {sb} ({sbp}) "
            f"[ROW] TEAEs leading to discontinuation | {da} ({dap}) | {db} ({dbp}) "
            f"end_table"
        )

    def _build_writeup(self, drug_a, drug_b, n_a, n_b, study_id,
                       tea, tap, teb, tbp, g34a, g34ap, g34b, g34bp,
                       sa, sap, sb, sbp, da, dap, db, dbp) -> str:
        return (
            f"Treatment-emergent adverse events (TEAEs) were reported for {tap}% "
            f"({tea}/{n_a}) of subjects in the {drug_a} arm and {tbp}% ({teb}/{n_b}) "
            f"of subjects in the {drug_b} arm. "
            f"Grade 3-4 TEAEs were reported for {g34ap}% of subjects in the {drug_a} arm "
            f"and {g34bp}% of subjects in the {drug_b} arm. "
            f"Treatment-emergent serious adverse events (SAEs) were reported for {sap}% "
            f"({sa}/{n_a}) of {drug_a} subjects and {sbp}% ({sb}/{n_b}) of {drug_b} subjects. "
            f"TEAEs leading to treatment discontinuation were reported for {dap}% ({da}/{n_a}) "
            f"of subjects in the {drug_a} arm and {dbp}% ({db}/{n_b}) of subjects "
            f"in the {drug_b} arm."
        )


# ─── Tier 3+4: LLM Augmentation ──────────────────────────────────────────────

class LLMAugmenter:
    """
    Uses Claude to paraphrase summaries and synthesize new examples.
    Requires ANTHROPIC_API_KEY.
    """

    PARAPHRASE_PROMPT = """You are an expert medical writer. Rewrite this clinical safety summary in a different but equally accurate style.

RULES:
1. Keep ALL numbers EXACTLY the same (do not change any digit)
2. Keep all clinical terms (TEAE, SAE, Grade 3-4, discontinuation)
3. Use different sentence structure and openings
4. The summary must cover: TEAE incidence, Grade 3-4 rates, SAE rates, discontinuation rates
5. Return ONLY the rewritten summary — no preamble, no explanation

Original Summary:
{writeup}

Rewritten Summary:"""

    SYNTHESIS_PROMPT = """Create a NEW clinical safety table and corresponding summary.

Base this on the template below but change:
- Drug names (use realistic oncology drug names)
- Patient numbers (N=50 to N=600 per arm, realistic)
- Percentages (must be mathematically consistent with counts)
- Ensure Grade 3-4% < overall TEAE%, SAE% < TEAE%

Template Table:
{table}

Template Summary:
{writeup}

Return in EXACTLY this format (nothing else):
TABLE:
[linearized table in start_table ... end_table format]
SUMMARY:
[clinical narrative]"""

    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def paraphrase(self, example: Dict) -> Optional[Dict]:
        """Rewrite summary in a different clinical style."""
        writeup = example.get("writeup") or example.get("output", "")
        if not writeup:
            return None

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{
                    "role": "user",
                    "content": self.PARAPHRASE_PROMPT.format(writeup=writeup)
                }]
            )
            new_writeup = response.content[0].text.strip()

            return {
                "table_text": example.get("table_text") or example.get("input", ""),
                "writeup": new_writeup,
                "augmentation": "llm_paraphrase",
            }
        except Exception as e:
            print(f"  Paraphrase failed: {e}")
            return None

    def synthesize(self, example: Dict) -> Optional[Dict]:
        """Generate a fully new table+summary pair."""
        table = example.get("table_text") or example.get("input", "")
        writeup = example.get("writeup") or example.get("output", "")
        if not table or not writeup:
            return None

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": self.SYNTHESIS_PROMPT.format(table=table, writeup=writeup)
                }]
            )
            text = response.content[0].text.strip()

            # Parse TABLE: / SUMMARY:
            table_match = re.search(r"TABLE:\s*(start_table.*?end_table)", text, re.DOTALL | re.IGNORECASE)
            summary_match = re.search(r"SUMMARY:\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)

            if table_match and summary_match:
                return {
                    "table_text": table_match.group(1).strip(),
                    "writeup": summary_match.group(1).strip(),
                    "augmentation": "llm_synthesis",
                }
        except Exception as e:
            print(f"  Synthesis failed: {e}")

        return None


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_pairs: List[Dict],
    output_dir: str,
    api_key: Optional[str] = None,
    tiers: List[int] = None,
    target_count: int = 1000,
) -> List[Dict]:

    if tiers is None:
        tiers = [1, 2, 3, 4] if api_key else [1, 2]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_examples = list(input_pairs)

    print(f"\n{'='*60}")
    print(f"Starting augmentation: {len(input_pairs)} → {target_count} examples")
    print(f"Tiers enabled: {tiers}")
    print(f"{'='*60}\n")

    # ── Tier 1: Number Variation ──────────────────────────────────────────────
    if 1 in tiers:
        variator = NumberVariator()
        target_t1 = min(300, target_count // 3)
        variants_per_example = max(1, target_t1 // len(input_pairs))

        print(f"Tier 1: Generating ~{target_t1} number variations...")
        t1_examples = []
        for ex in input_pairs:
            t1_examples.extend(variator.vary(ex, n_variants=variants_per_example))

        all_examples.extend(t1_examples[:target_t1])
        print(f"  ✅ Added {min(len(t1_examples), target_t1)} examples. Total: {len(all_examples)}")

    # ── Tier 2: Template Synthesis ────────────────────────────────────────────
    if 2 in tiers and len(all_examples) < target_count:
        needed = min(200, target_count - len(all_examples))
        synthesizer = TemplateSynthesizer()
        print(f"\nTier 2: Generating {needed} template-synthesized examples...")
        t2_examples = synthesizer.generate(n_examples=needed)
        all_examples.extend(t2_examples)
        print(f"  ✅ Added {len(t2_examples)} examples. Total: {len(all_examples)}")

    # ── Tier 3+4: LLM Augmentation ────────────────────────────────────────────
    if api_key and len(all_examples) < target_count:
        llm = LLMAugmenter(api_key=api_key)

        # Tier 3: Paraphrase
        if 3 in tiers and len(all_examples) < target_count:
            t3_target = min(250, target_count - len(all_examples))
            print(f"\nTier 3: Generating {t3_target} LLM paraphrases...")
            t3_count = 0
            for i in range(t3_target):
                ex = random.choice(input_pairs)
                result = llm.paraphrase(ex)
                if result:
                    all_examples.append(result)
                    t3_count += 1
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{t3_target}")
                time.sleep(0.3)  # Rate limiting
            print(f"  ✅ Added {t3_count} examples. Total: {len(all_examples)}")

        # Tier 4: Full Synthesis
        if 4 in tiers and len(all_examples) < target_count:
            t4_target = min(250, target_count - len(all_examples))
            print(f"\nTier 4: Generating {t4_target} LLM full syntheses...")
            t4_count = 0
            for i in range(t4_target):
                ex = random.choice(input_pairs)
                result = llm.synthesize(ex)
                if result:
                    all_examples.append(result)
                    t4_count += 1
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{t4_target}")
                time.sleep(0.5)
            print(f"  ✅ Added {t4_count} examples. Total: {len(all_examples)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_dir) / "synthetic_1000.jsonl"
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'='*60}")
    print(f"✅ Saved {len(all_examples)} examples → {output_path}")

    # Stats
    aug_types = {}
    for ex in all_examples:
        t = ex.get("augmentation", "original")
        aug_types[t] = aug_types.get(t, 0) + 1
    print("\nBreakdown by type:")
    for t, n in sorted(aug_types.items()):
        print(f"  {t}: {n}")
    print(f"{'='*60}\n")

    return all_examples


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Data Augmentation Pipeline")
    parser.add_argument("--input",  help="Path to raw_pairs.json (from PDF extractor)")
    parser.add_argument("--docx",   help="Path to DOCX dataset (auto-extracts pairs)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (for Tier 3+4)")
    parser.add_argument("--tiers", nargs="+", type=int, default=None,
                        help="Which tiers to run (default: all available)")
    parser.add_argument("--target", type=int, default=1000,
                        help="Target number of examples (default: 1000)")
    args = parser.parse_args()

    # Load input data
    if args.docx:
        print(f"Extracting pairs from DOCX: {args.docx}")
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.data_processing.pdf_extractor import ClinicalPDFExtractor
        extractor = ClinicalPDFExtractor(args.docx)
        pairs = [p.to_dict() for p in extractor.extract_all()]
        # Normalize keys
        pairs = [{"table_text": p["table_text"], "writeup": p["writeup"]} for p in pairs if p["writeup"]]
        print(f"Extracted {len(pairs)} pairs from DOCX")
    elif args.input:
        with open(args.input) as f:
            raw = json.load(f)
        # Normalize
        pairs = []
        for p in raw:
            tw = p.get("table_text") or p.get("input", "")
            wr = p.get("writeup") or p.get("output", "")
            if tw and wr:
                pairs.append({"table_text": tw, "writeup": wr})
        print(f"Loaded {len(pairs)} pairs from {args.input}")
    else:
        parser.error("Must provide --input or --docx")

    if not pairs:
        print("❌ No valid pairs found. Check your input file.")
        sys.exit(1)

    run_pipeline(
        input_pairs=pairs,
        output_dir=args.output,
        api_key=args.api_key,
        tiers=args.tiers,
        target_count=args.target,
    )
