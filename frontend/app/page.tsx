"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  ShieldCheck,
  BrainCircuit,
  Gauge,
  BarChart3,
  FlaskConical,
  ShieldAlert,
  Database,
  FileText,
} from "lucide-react";
import { SectionHeader, StatBlock, ClinicalCard, Badge } from "@/components/shared/section-header";
import { EcgDivider } from "@/components/shared/ecg-line";

const fade = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};

const capabilities = [
  {
    icon: Gauge,
    title: "Fast ML Pipeline",
    description: "Feature-based retrieval and templated narrative generation for rapid turnaround.",
    accent: "text-serum",
  },
  {
    icon: BrainCircuit,
    title: "DL Generation",
    description: "Transformer-based rewriting and fine-tuned inference for higher narrative quality.",
    accent: "text-saline",
  },
  {
    icon: ShieldCheck,
    title: "Verification Layer",
    description: "Numeric checks and warning flags to reduce hallucination risk before review.",
    accent: "text-verified",
  },
];

const outcomes = [
  "Reduce manual table-to-text authoring time",
  "Generate consistent clinical narrative style",
  "Expose confidence and warning metadata",
  "Compare model paths for review decisions",
];

const limitations = [
  {
    title: "Unbounded Output Space",
    detail:
      "Classical ML models predict labels or numbers. A regulatory summary is variable-length text with syntax, discourse structure, and domain-specific phrasing, so the target is not a fixed label space.",
  },
  {
    title: "Structured Plus Semantic Input",
    detail:
      "The table mixes numeric relationships, clinical row semantics, and variable schemas across studies. Fixed-size feature vectors struggle to preserve both arithmetic structure and language meaning at once.",
  },
  {
    title: "Schema Variability",
    detail:
      "Different studies expose different arm counts, row layouts, and header conventions. A pure classical pipeline assumes a stable feature contract, but these tables do not stay stable enough for direct end-to-end generation.",
  },
  {
    title: "Generation Requires Planning",
    detail:
      "The system must decide what to mention, in what order, and how to express comparisons in ICH E3 style. That is closer to constrained language generation than standard classification or regression.",
  },
];

const contributions = [
  {
    title: "Deterministic Template Generation",
    detail:
      "A rule-based linearisation and template system converts structured adverse event tables into guaranteed-correct regulatory scaffolds before any neural rewriting occurs.",
    accent: "text-serum",
  },
  {
    title: "Fine-Tuned Seq2Seq Rewriting",
    detail:
      "A QLoRA-adapted Flan-T5-XXL model is trained for the narrower rewrite task: transforming rigid source-grounded templates into fluent ICH E3 prose rather than solving end-to-end table-to-text generation.",
    accent: "text-saline",
  },
  {
    title: "Hallucination-Safe Fallback",
    detail:
      "Unsafe rewrites are rejected by the guard layer. When source grounding is insufficient, the system returns the deterministic template instead of emitting a fluent but unreliable summary.",
    accent: "text-verified",
  },
];

const blueprint = [
  {
    icon: Database,
    title: "Stage 1 · Deterministic Template Generation",
    detail: "Structured adverse event tables are linearized, parsed, and converted into a formulaic regulatory template. Every number is copied from the source table, giving a provably hallucination-free baseline.",
    accent: "text-serum",
  },
  {
    icon: BrainCircuit,
    title: "Stage 2 · Fine-Tuned Seq2Seq Rewriting",
    detail: "A fine-tuned Flan-T5-XXL rewrite model converts the rigid template into fluent ICH E3 prose. Training uses QLoRA with 4-bit NF4 quantization, LoRA r=16, 10K+ combined narrative pairs, and 5x upweighting for 191 gold examples.",
    accent: "text-saline",
  },
  {
    icon: ShieldAlert,
    title: "Stage 3 · Hallucination Control",
    detail: "The system originally used constrained beam search with forced number constraints, but T5-family models produced number-dump artefacts. The production design now accepts model output only when hallucinations are at most 2 and source-number coverage is at least 50%; otherwise the guaranteed-correct template is returned.",
    accent: "text-verified",
  },
  {
    icon: FileText,
    title: "Outputs",
    detail: "Final output is either verified fluent prose or deterministic fallback prose, always paired with traceable metadata for reviewer inspection and model auditing.",
    accent: "text-plasma",
  },
];

const flowNodes = [
  { id: "input", label: "Input Tables", x: 120, y: 74, color: "#C9A84C" },
  { id: "template", label: "Stage 1 Template", x: 330, y: 40, color: "#C9A84C" },
  { id: "rewrite", label: "Stage 2 Rewrite", x: 330, y: 108, color: "#6BA3BE" },
  { id: "guard", label: "Stage 3 Guard", x: 520, y: 74, color: "#3B8A5A" },
  { id: "output", label: "Verified Output", x: 710, y: 74, color: "#D4463B" },
];

export default function HomePage() {
  return (
    <div className="space-y-24 pb-20">
      <section className="relative overflow-hidden pt-10 md:pt-16">
        <div className="ambient-layer ambient-layer-a" />
        <div className="ambient-layer ambient-layer-b" />
        <div className="ambient-layer ambient-layer-c" />
        <div className="absolute inset-0 opacity-[0.08]">
          <div
            className="h-full w-full"
            style={{
              backgroundImage:
                "linear-gradient(to right, rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.06) 1px, transparent 1px)",
              backgroundSize: "44px 44px",
            }}
          />
        </div>

        <motion.div
          className="relative max-w-4xl"
          initial="hidden"
          animate="show"
          variants={{ show: { transition: { staggerChildren: 0.1 } } }}
        >
          <motion.div variants={fade}>
            <Badge variant="plasma" size="md">Clinical NLP Platform</Badge>
          </motion.div>

          <motion.h1 variants={fade} className="mt-6 font-display text-5xl leading-[1.05] text-parchment md:text-7xl">
            Clinical Safety Tables
            <span className="block text-plasma">Converted Into Narrative Evidence</span>
          </motion.h1>

          <motion.p variants={fade} className="mt-6 max-w-3xl text-lg leading-relaxed text-cortex md:text-2xl">
            ClinicalSafe transforms adverse event tables into review-ready summaries using hybrid ML and deep learning workflows with explicit verification signals.
          </motion.p>

          <motion.div variants={fade} className="mt-8 flex flex-wrap gap-3">
            <div className="chip-float border border-serum/30 bg-serum/10 px-3 py-1.5 text-xs uppercase tracking-wide text-serum">
              0.0 Hallucination Target
            </div>
            <div className="chip-float border border-saline/30 bg-saline/10 px-3 py-1.5 text-xs uppercase tracking-wide text-saline">
              ML + DL Compare Mode
            </div>
            <div className="chip-float border border-verified/30 bg-verified/10 px-3 py-1.5 text-xs uppercase tracking-wide text-verified">
              Numeric Guardrails
            </div>
          </motion.div>

          <motion.div variants={fade} className="mt-10 flex flex-wrap gap-4">
            <Link href="/summarizer" className="btn-primary">
              Open Summarizer
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link href="/research" className="btn-secondary">Explore Methodology</Link>
          </motion.div>

          <motion.div variants={fade} className="mt-14 grid grid-cols-2 gap-5 md:grid-cols-4">
            <StatBlock value="10K" label="Synthetic Pairs" />
            <StatBlock value="191" label="Indexed Examples" />
            <StatBlock value="98%" label="Numeric Checks" />
            <StatBlock value="<2s" label="DL Runtime" />
          </motion.div>

          <motion.div
            variants={fade}
            className="mt-8 inline-flex items-center gap-2 text-xs uppercase tracking-wider text-cortex"
            animate={{ y: [0, 6, 0] }}
            transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
          >
            Scroll to see architecture
            <span className="h-2 w-2 rounded-full bg-plasma" />
          </motion.div>
        </motion.div>
      </section>

      <EcgDivider />

      <section>
        <SectionHeader
          label="Capabilities"
          title="Designed for Clinical Review Workloads"
          description="Three coordinated layers balance speed, generation quality, and trust for safety reporting workflows."
        />
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {capabilities.map((item) => (
            <ClinicalCard key={item.title} className="p-6" hover>
              <item.icon className={`h-6 w-6 ${item.accent}`} />
              <h3 className="mt-4 font-heading text-xl text-parchment">{item.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-cortex">{item.description}</p>
            </ClinicalCard>
          ))}
        </div>
      </section>

      <EcgDivider />

      <section>
        <SectionHeader
          label="System Blueprint"
          title="Three-Stage Clinical Generation Architecture"
          description="The front-end architecture view now reflects the actual research pipeline: deterministic template generation, fine-tuned seq2seq rewriting, and a hard hallucination-control gate before release."
        />

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.55 }}
          className="mt-10 overflow-hidden rounded-xl border border-white/10 bg-marrow/70 p-4 md:p-6"
        >
          <div className="relative">
            <svg viewBox="0 0 820 160" className="w-full">
              <defs>
                <linearGradient id="flowWarm" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#C9A84C" />
                  <stop offset="100%" stopColor="#D4463B" />
                </linearGradient>
                <linearGradient id="flowCool" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#6BA3BE" />
                  <stop offset="100%" stopColor="#3B8A5A" />
                </linearGradient>
              </defs>

              <motion.path
                d="M165 74 L285 40"
                stroke="url(#flowWarm)"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0.4 }}
                whileInView={{ pathLength: 1, opacity: 0.9 }}
                viewport={{ once: true }}
                transition={{ duration: 0.9, ease: "easeOut" }}
              />
              <motion.path
                d="M165 74 L285 108"
                stroke="url(#flowCool)"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0.4 }}
                whileInView={{ pathLength: 1, opacity: 0.9 }}
                viewport={{ once: true }}
                transition={{ duration: 0.9, delay: 0.08, ease: "easeOut" }}
              />
              <motion.path
                d="M375 40 L474 74"
                stroke="url(#flowWarm)"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0.4 }}
                whileInView={{ pathLength: 1, opacity: 0.9 }}
                viewport={{ once: true }}
                transition={{ duration: 0.9, delay: 0.2, ease: "easeOut" }}
              />
              <motion.path
                d="M375 108 L474 74"
                stroke="url(#flowCool)"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0.4 }}
                whileInView={{ pathLength: 1, opacity: 0.9 }}
                viewport={{ once: true }}
                transition={{ duration: 0.9, delay: 0.28, ease: "easeOut" }}
              />
              <motion.path
                d="M565 74 L664 74"
                stroke="url(#flowWarm)"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0.4 }}
                whileInView={{ pathLength: 1, opacity: 0.9 }}
                viewport={{ once: true }}
                transition={{ duration: 0.9, delay: 0.36, ease: "easeOut" }}
              />
            </svg>

            <div className="pointer-events-none absolute inset-0 hidden md:block">
              <div className="flow-pulse flow-pulse-a" />
              <div className="flow-pulse flow-pulse-b" />
              <div className="flow-pulse flow-pulse-c" />
            </div>

            <div className="mt-3 grid grid-cols-1 gap-3 md:absolute md:inset-0 md:mt-0 md:grid-cols-5 md:items-center md:gap-0">
              {flowNodes.map((node, index) => (
                <motion.div
                  key={node.id}
                  initial={{ opacity: 0, y: 18 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.35 }}
                  transition={{ delay: 0.1 + index * 0.08, duration: 0.45 }}
                  className="mx-auto w-full max-w-[170px] rounded-lg border border-white/12 bg-ink/85 p-3 text-center shadow-[0_14px_24px_rgba(0,0,0,0.28)]"
                  whileHover={{ y: -4, borderColor: "rgba(255,255,255,0.28)" }}
                >
                  <div className="text-xs uppercase tracking-wider text-cortex">Stage</div>
                  <div className="mt-1 font-heading text-sm text-parchment">{node.label}</div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        <div className="mt-8 grid gap-4 md:grid-cols-2">
          {blueprint.map((item, index) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ delay: 0.06 * index, duration: 0.45 }}
            >
              <ClinicalCard className="p-5" hover>
                <item.icon className={`h-5 w-5 ${item.accent}`} />
                <h3 className="mt-3 font-heading text-lg text-parchment">{item.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-cortex">{item.detail}</p>
              </ClinicalCard>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.5 }}
          className="mt-8 rounded-xl border border-white/10 bg-ink/60 p-6"
        >
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cortex">
            <ShieldCheck className="h-4 w-4 text-verified" />
            Safety Rationale
          </div>
          <p className="mt-3 text-sm leading-relaxed text-cortex">
            The key contribution is that the model is not asked to solve the full table-to-narrative problem in one jump. It first receives a guaranteed-correct regulatory template, then performs a narrower rewrite task. If the rewrite does not preserve enough source grounding, the system drops back to the deterministic template instead of emitting an unsafe summary.
          </p>
        </motion.div>
      </section>

      <EcgDivider />

      <section>
        <SectionHeader
          label="Motivation"
          title="Why Classical ML Alone Is Not Enough"
          description="This problem cannot be solved as a simple classifier or regressor. The input is structured and semantic at the same time, and the output is an unconstrained clinical paragraph rather than a fixed prediction target."
        />

        <div className="mt-10 grid gap-4 md:grid-cols-2">
          {limitations.map((item, index) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ delay: 0.06 * index, duration: 0.45 }}
            >
              <ClinicalCard className="p-5" hover>
                <h3 className="font-heading text-lg text-parchment">{item.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-cortex">{item.detail}</p>
              </ClinicalCard>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
          transition={{ duration: 0.5 }}
          className="mt-8 rounded-xl border border-white/10 bg-marrow/60 p-6"
        >
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cortex">
            <BrainCircuit className="h-4 w-4 text-saline" />
            Design Consequence
          </div>
          <p className="mt-3 text-sm leading-relaxed text-cortex">
            That is why ClinicalSafe splits the task. The deterministic stage solves factual extraction and ordering, while the fine-tuned seq2seq model handles the narrower rewrite problem of turning a rigid but correct template into fluent regulatory prose.
          </p>
        </motion.div>
      </section>

      <EcgDivider />

      <section>
        <SectionHeader
          label="Research Contribution"
          title="Three Core Technical Contributions"
          description="The system contribution is not just a model. It is the combination of a guaranteed-correct deterministic stage, a narrower rewrite objective for the seq2seq model, and a production fallback policy that privileges correctness over stylistic fluency."
        />

        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {contributions.map((item, index) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ delay: 0.06 * index, duration: 0.45 }}
            >
              <ClinicalCard className="h-full p-6" hover>
                <div className={`text-xs uppercase tracking-wider ${item.accent}`}>Contribution {index + 1}</div>
                <h3 className="mt-3 font-heading text-xl text-parchment">{item.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-cortex">{item.detail}</p>
              </ClinicalCard>
            </motion.div>
          ))}
        </div>
      </section>

      <EcgDivider />

      <section className="grid gap-8 lg:grid-cols-2">
        <div>
          <SectionHeader
            label="Value"
            title="What Teams Get"
            description="ClinicalSafe is structured to support both exploratory prototyping and high-accountability medical writing workflows."
          />
          <ul className="mt-8 space-y-4">
            {outcomes.map((item) => (
              <li key={item} className="flex items-start gap-3 text-parchment/85">
                <span className="mt-2 h-1.5 w-1.5 rounded-full bg-plasma" />
                {item}
              </li>
            ))}
          </ul>
        </div>

        <ClinicalCard className="p-7" hover={false}>
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cortex">
            <FlaskConical className="h-3.5 w-3.5" />
            Workflow
          </div>
          <h3 className="mt-3 font-heading text-2xl text-parchment">From Raw Table to Decision-Ready Output</h3>
          <div className="mt-6 space-y-4 text-sm text-cortex">
            <p>1. Ingest a structured safety table from text, PDF, DOCX, or image extraction.</p>
            <p>2. Build a deterministic regulatory template using source-grounded values only.</p>
            <p>3. Rewrite that template with the fine-tuned seq2seq model when fluent generation is requested.</p>
            <p>4. Accept the rewrite only if hallucinations are at most 2 and coverage stays at or above 50%; otherwise return the deterministic template.</p>
          </div>
          <Link href="/summarizer" className="btn-primary mt-7 inline-flex">
            Start a Run
            <BarChart3 className="h-4 w-4" />
          </Link>
        </ClinicalCard>
      </section>
    </div>
  );
}
