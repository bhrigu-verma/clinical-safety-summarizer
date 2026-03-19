"use client";

import { motion } from "framer-motion";
import { SectionHeader, StatBlock } from "@/components/shared/section-header";

/* ============================================
   METRICS GRID
   ============================================
   Displays model performance metrics with
   visual emphasis on key differentiators.
*/

interface Metric {
  label: string;
  value: string;
  sublabel?: string;
}

interface MetricGroup {
  title: string;
  metrics: Metric[];
  accent: "serum" | "saline";
}

const metricGroups: MetricGroup[] = [
  {
    title: "ML Pipeline (LightGBM + KNN)",
    accent: "serum",
    metrics: [
      { label: "ROUGE-1", value: "0.71", sublabel: "F1 Score" },
      { label: "ROUGE-L", value: "0.68", sublabel: "F1 Score" },
      { label: "Latency", value: "<100ms", sublabel: "Per table" },
      { label: "Interpretability", value: "High", sublabel: "Feature importance" },
    ],
  },
  {
    title: "DL Pipeline (Flan-T5-XL LoRA)",
    accent: "saline",
    metrics: [
      { label: "ROUGE-1", value: "0.76", sublabel: "F1 Score" },
      { label: "ROUGE-L", value: "0.73", sublabel: "F1 Score" },
      { label: "Latency", value: "~2s", sublabel: "GPU inference" },
      { label: "Fluency", value: "High", sublabel: "Natural language" },
    ],
  },
];

export function MetricsGrid() {
  return (
    <section className="py-24 px-6 bg-ink/50">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="Benchmarks"
          title="Performance Metrics"
          description="Validated on clinical safety tables from FDA submissions"
        />

        <div className="grid md:grid-cols-2 gap-8">
          {metricGroups.map((group, groupIndex) => (
            <MetricCard key={group.title} group={group} delay={groupIndex * 0.15} />
          ))}
        </div>

        {/* Additional context */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <p className="text-cortex text-sm max-w-2xl mx-auto">
            Metrics computed on held-out test set of 50 clinical safety tables. 
            ML pipeline optimized for speed and interpretability; DL pipeline 
            for maximum quality and nuance.
          </p>
        </motion.div>
      </div>
    </section>
  );
}

interface MetricCardProps {
  group: MetricGroup;
  delay: number;
}

function MetricCard({ group, delay }: MetricCardProps) {
  const accentColors = {
    serum: {
      border: "border-serum/30",
      text: "text-serum",
      bg: "bg-serum/5",
    },
    saline: {
      border: "border-saline/30",
      text: "text-saline",
      bg: "bg-saline/5",
    },
  };

  const colors = accentColors[group.accent];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className={`clinical-card border ${colors.border} ${colors.bg}`}
    >
      <h3 className={`font-heading text-lg ${colors.text} mb-6`}>{group.title}</h3>
      
      <div className="grid grid-cols-2 gap-4">
        {group.metrics.map((metric, index) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: delay + 0.1 * index }}
          >
            <StatBlock
              label={metric.label}
              value={metric.value}
              sublabel={metric.sublabel}
            />
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

/* ============================================
   COMPARISON TABLE
   ============================================
   Side-by-side feature comparison
*/

interface ComparisonRow {
  feature: string;
  ml: string;
  dl: string;
}

const comparisonData: ComparisonRow[] = [
  { feature: "Training Data", ml: "187 pairs", dl: "10K+ augmented" },
  { feature: "Model Size", ml: "~50MB", dl: "~3GB (quantized)" },
  { feature: "GPU Required", ml: "No", dl: "Yes (inference)" },
  { feature: "Explainability", ml: "Full feature importance", dl: "Attention maps" },
  { feature: "Best For", ml: "Production speed", dl: "Research quality" },
];

export function ComparisonTable() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="overflow-hidden rounded-lg border border-parchment/10"
    >
      {/* Header */}
      <div className="grid grid-cols-3 bg-marrow/50">
        <div className="px-4 py-3 text-xs uppercase tracking-wider text-cortex">
          Feature
        </div>
        <div className="px-4 py-3 text-xs uppercase tracking-wider text-serum text-center">
          ML Pipeline
        </div>
        <div className="px-4 py-3 text-xs uppercase tracking-wider text-saline text-center">
          DL Pipeline
        </div>
      </div>

      {/* Rows */}
      {comparisonData.map((row, index) => (
        <motion.div
          key={row.feature}
          initial={{ opacity: 0, x: -10 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.05 * index }}
          className={`grid grid-cols-3 ${index % 2 === 0 ? "bg-ink/30" : "bg-ink/10"}`}
        >
          <div className="px-4 py-3 text-sm text-parchment">{row.feature}</div>
          <div className="px-4 py-3 text-sm text-cortex text-center font-mono">
            {row.ml}
          </div>
          <div className="px-4 py-3 text-sm text-cortex text-center font-mono">
            {row.dl}
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}
