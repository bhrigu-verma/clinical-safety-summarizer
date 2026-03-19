"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { SectionHeader, ClinicalCard, StatBlock } from "@/components/shared/section-header";
import { EcgDivider } from "@/components/shared/ecg-line";
import { ArrowRight, TestTubeDiagonal, Scale, ShieldAlert } from "lucide-react";

const pipelineRows = [
  {
    name: "ML",
    strengths: "Fast, interpretable, stable for repeated drafts",
    tradeoffs: "Less fluent phrasing in complex narrative contexts",
    color: "text-serum",
  },
  {
    name: "DL Base",
    strengths: "More fluent language generation than pure templates",
    tradeoffs: "Higher variability and latency",
    color: "text-saline",
  },
  {
    name: "DL Fine-Tuned",
    strengths: "Best quality profile with domain-tuned behavior",
    tradeoffs: "Heavier runtime requirements and model-loading constraints",
    color: "text-verified",
  },
];

const evaluationNotes = [
  "ROUGE metrics are used for overlap quality checks, not final clinical validity.",
  "Numeric accuracy and warning flags are first-pass heuristics for reviewer attention.",
  "Outputs should remain under human review for regulatory submission contexts.",
];

export default function ResearchPage() {
  return (
    <div className="space-y-20 pb-20">
      <section className="pt-10 md:pt-16">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="max-w-5xl"
        >
          <p className="text-xs uppercase tracking-wider text-plasma">Research View</p>
          <h1 className="mt-4 font-display text-5xl leading-[1.08] text-parchment md:text-6xl">
            Methodology and Evaluation Framing
          </h1>
          <p className="mt-6 max-w-3xl text-lg leading-relaxed text-cortex">
            ClinicalSafe combines retrieval-based and transformer-based generation pathways. This page summarizes the frontend-facing interpretation layer used for model comparison and review workflows.
          </p>
        </motion.div>
      </section>

      <section className="grid gap-5 sm:grid-cols-3">
        <ClinicalCard className="p-6" hover={false}>
          <StatBlock value="3" label="Generation Modes" variant="plasma" />
        </ClinicalCard>
        <ClinicalCard className="p-6" hover={false}>
          <StatBlock value="10K" label="Training Pair Scale" variant="serum" />
        </ClinicalCard>
        <ClinicalCard className="p-6" hover={false}>
          <StatBlock value="191" label="Indexed Retrieval Rows" variant="verified" />
        </ClinicalCard>
      </section>

      <EcgDivider />

      <section>
        <SectionHeader
          label="Model Comparison"
          title="Pipeline Tradeoff Matrix"
          description="Each mode is exposed in the summarizer because reviewers need speed-quality tradeoffs at run time, not only in offline benchmarks."
        />

        <div className="mt-10 space-y-4">
          {pipelineRows.map((row) => (
            <ClinicalCard key={row.name} className="p-5" hover={false}>
              <div className="grid gap-4 md:grid-cols-[120px_1fr_1fr] md:items-start">
                <p className={`font-heading text-lg ${row.color}`}>{row.name}</p>
                <div>
                  <p className="text-xs uppercase tracking-wider text-cortex">Strengths</p>
                  <p className="mt-1 text-sm text-parchment/85">{row.strengths}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wider text-cortex">Tradeoffs</p>
                  <p className="mt-1 text-sm text-parchment/85">{row.tradeoffs}</p>
                </div>
              </div>
            </ClinicalCard>
          ))}
        </div>
      </section>

      <EcgDivider />

      <section className="grid gap-6 lg:grid-cols-2">
        <ClinicalCard className="p-6" hover={false}>
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cortex">
            <Scale className="h-4 w-4" />
            Evaluation Interpretation
          </div>
          <ul className="mt-5 space-y-3 text-sm text-parchment/85">
            {evaluationNotes.map((note) => (
              <li key={note} className="flex items-start gap-3">
                <span className="mt-2 h-1.5 w-1.5 rounded-full bg-saline" />
                {note}
              </li>
            ))}
          </ul>
        </ClinicalCard>

        <ClinicalCard className="p-6" hover={false}>
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cortex">
            <ShieldAlert className="h-4 w-4" />
            Safety Position
          </div>
          <p className="mt-3 text-sm leading-relaxed text-cortex">
            Outputs are drafting aids. Verification metadata helps prioritize reviewer focus but does not replace domain expertise or regulatory checks.
          </p>
          <Link href="/summarizer" className="btn-primary mt-6 inline-flex">
            <TestTubeDiagonal className="h-4 w-4" />
            Run Comparative Test
            <ArrowRight className="h-4 w-4" />
          </Link>
        </ClinicalCard>
      </section>
    </div>
  );
}
