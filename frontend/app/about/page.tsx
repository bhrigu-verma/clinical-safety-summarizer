"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { SectionHeader, ClinicalCard } from "@/components/shared/section-header";
import { EcgDivider } from "@/components/shared/ecg-line";
import { ArrowRight, Shield, Microscope, Target, GitBranch, Users } from "lucide-react";

const principles = [
  {
    icon: Shield,
    title: "Safety-Critical Mindset",
    description: "Generated text is treated as assistive output, never as an unreviewed final source of truth.",
  },
  {
    icon: Microscope,
    title: "Evidence Traceability",
    description: "Model outputs are paired with confidence and warning cues to support validation workflows.",
  },
  {
    icon: Target,
    title: "Clinical Utility",
    description: "The product is optimized for practical safety narrative drafting, not generic text generation demos.",
  },
];

const roadmap = [
  "Strengthen tabular parsing robustness for multi-line and sparse formats.",
  "Add export templates for CSR-friendly narrative sections.",
  "Introduce run history and reproducibility metadata in the frontend.",
  "Extend benchmark pages with mode-by-mode score visualizations.",
];

export default function AboutPage() {
  return (
    <div className="space-y-20 pb-20">
      <section className="pt-10 md:pt-16">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="max-w-4xl"
        >
          <p className="text-xs uppercase tracking-wider text-plasma">About ClinicalSafe</p>
          <h1 className="mt-4 font-display text-5xl leading-[1.08] text-parchment md:text-6xl">
            Building Reliable Clinical Narrative Assistants
          </h1>
          <p className="mt-6 max-w-3xl text-lg leading-relaxed text-cortex">
            ClinicalSafe is a research-to-product initiative focused on one narrow, high-impact task:
            converting clinical safety tables into clear draft narratives that are easier and faster for expert teams to review.
          </p>
        </motion.div>
      </section>

      <EcgDivider />

      <section>
        <SectionHeader
          label="Principles"
          title="How We Design"
          description="The frontend is intentionally explicit about model behavior so reviewers can quickly decide what to trust, what to check, and what to refine."
        />
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {principles.map((item) => (
            <ClinicalCard key={item.title} className="p-6">
              <item.icon className="h-6 w-6 text-saline" />
              <h3 className="mt-4 font-heading text-xl text-parchment">{item.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-cortex">{item.description}</p>
            </ClinicalCard>
          ))}
        </div>
      </section>

      <EcgDivider />

      <section className="grid gap-7 lg:grid-cols-2">
        <ClinicalCard className="p-6" hover={false}>
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cortex">
            <GitBranch className="h-4 w-4" />
            Near-Term Roadmap
          </div>
          <ul className="mt-5 space-y-3 text-sm text-parchment/85">
            {roadmap.map((item) => (
              <li key={item} className="flex items-start gap-3">
                <span className="mt-2 h-1.5 w-1.5 rounded-full bg-verified" />
                {item}
              </li>
            ))}
          </ul>
        </ClinicalCard>

        <ClinicalCard className="p-6" hover={false}>
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-cortex">
            <Users className="h-4 w-4" />
            Collaborate
          </div>
          <h2 className="mt-3 font-heading text-2xl text-parchment">Open, Reviewable, and Extensible</h2>
          <p className="mt-3 text-sm leading-relaxed text-cortex">
            The stack is intentionally transparent so teams can adapt prompts, templates, verification logic, and model selection criteria to their own medical review policies.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link href="/research" className="btn-secondary">Read Research</Link>
            <Link href="/summarizer" className="btn-primary">
              Try Summarizer
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </ClinicalCard>
      </section>
    </div>
  );
}
