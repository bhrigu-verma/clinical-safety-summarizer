"use client";

import { motion } from "framer-motion";
import { SectionHeader } from "@/components/shared/section-header";
import { EcgDivider } from "@/components/shared/ecg-line";

/* ============================================
   PIPELINE VISUALIZATION
   ============================================
   Shows the dual ML/DL processing pipelines
   with animated flow indicators.
*/

interface PipelineStep {
  name: string;
  description: string;
}

const mlPipeline: PipelineStep[] = [
  { name: "Feature Extraction", description: "Row statistics & patterns" },
  { name: "Content Selection", description: "LightGBM classifier" },
  { name: "KNN Retrieval", description: "Similar table matching" },
  { name: "Template Generation", description: "Slot-filling via templates" },
];

const dlPipeline: PipelineStep[] = [
  { name: "Tokenization", description: "Table structure encoding" },
  { name: "Encoder", description: "Flan-T5-XL attention" },
  { name: "Fine-tuned Adapter", description: "LoRA clinical specialization" },
  { name: "Decoder", description: "Autoregressive generation" },
];

export function PipelineVisual() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="Architecture"
          title="Dual-Pathway Processing"
          description="Two complementary approaches for maximum accuracy and interpretability"
        />

        <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
          {/* ML Pipeline */}
          <PipelineColumn
            title="ML Pipeline"
            subtitle="Interpretable & Fast"
            steps={mlPipeline}
            accentColor="serum"
            delay={0}
          />

          {/* DL Pipeline */}
          <PipelineColumn
            title="DL Pipeline"
            subtitle="Neural & Nuanced"
            steps={dlPipeline}
            accentColor="saline"
            delay={0.2}
          />
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6 }}
          className="mt-12"
        >
          <EcgDivider />
        </motion.div>
      </div>
    </section>
  );
}

interface PipelineColumnProps {
  title: string;
  subtitle: string;
  steps: PipelineStep[];
  accentColor: "serum" | "saline" | "plasma";
  delay: number;
}

function PipelineColumn({ title, subtitle, steps, accentColor, delay }: PipelineColumnProps) {
  const colorClasses = {
    serum: {
      border: "border-serum/30",
      bg: "bg-serum/10",
      text: "text-serum",
      dot: "bg-serum",
    },
    saline: {
      border: "border-saline/30",
      bg: "bg-saline/10",
      text: "text-saline",
      dot: "bg-saline",
    },
    plasma: {
      border: "border-plasma/30",
      bg: "bg-plasma/10",
      text: "text-plasma",
      dot: "bg-plasma",
    },
  };

  const colors = colorClasses[accentColor];

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className={`clinical-card border ${colors.border}`}
    >
      {/* Header */}
      <div className="mb-6">
        <h3 className={`font-heading text-xl ${colors.text} mb-1`}>{title}</h3>
        <p className="text-cortex text-sm">{subtitle}</p>
      </div>

      {/* Steps */}
      <div className="space-y-4">
        {steps.map((step, index) => (
          <motion.div
            key={step.name}
            initial={{ opacity: 0, x: -10 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: delay + 0.1 * index }}
            className="flex items-start gap-4"
          >
            {/* Step number */}
            <div className="flex flex-col items-center">
              <div className={`w-8 h-8 rounded-full ${colors.bg} flex items-center justify-center`}>
                <span className={`font-mono text-sm ${colors.text}`}>{index + 1}</span>
              </div>
              {index < steps.length - 1 && (
                <div className={`w-px h-8 ${colors.bg} mt-1`} />
              )}
            </div>

            {/* Step content */}
            <div className="pt-1">
              <h4 className="text-parchment font-medium text-sm">{step.name}</h4>
              <p className="text-cortex text-xs mt-0.5">{step.description}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

/* ============================================
   ARCHITECTURE DIAGRAM
   ============================================
   Higher-level system overview
*/

export function ArchitectureDiagram() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { opacity: 1, scale: 1 },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      className="relative py-8"
    >
      {/* Input */}
      <motion.div 
        variants={itemVariants}
        className="flex justify-center mb-8"
      >
        <div className="clinical-card border border-parchment/20 text-center max-w-xs">
          <div className="text-xs text-cortex uppercase tracking-wider mb-1">Input</div>
          <div className="font-heading text-parchment">Clinical Safety Table</div>
        </div>
      </motion.div>

      {/* Arrow down */}
      <motion.div variants={itemVariants} className="flex justify-center mb-8">
        <svg className="w-6 h-8 text-cortex" fill="none" viewBox="0 0 24 32" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v24m0 0l-6-6m6 6l6-6" />
        </svg>
      </motion.div>

      {/* Processing branches */}
      <motion.div variants={itemVariants} className="flex justify-center gap-8 mb-8">
        <div className="clinical-card border border-serum/30 text-center">
          <div className="text-xs text-serum uppercase tracking-wider mb-1">ML Path</div>
          <div className="font-mono text-sm text-parchment">Port 8000</div>
        </div>
        <div className="clinical-card border border-saline/30 text-center">
          <div className="text-xs text-saline uppercase tracking-wider mb-1">DL Path</div>
          <div className="font-mono text-sm text-parchment">Port 8001</div>
        </div>
      </motion.div>

      {/* Arrow down */}
      <motion.div variants={itemVariants} className="flex justify-center mb-8">
        <svg className="w-6 h-8 text-cortex" fill="none" viewBox="0 0 24 32" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v24m0 0l-6-6m6 6l6-6" />
        </svg>
      </motion.div>

      {/* Output */}
      <motion.div 
        variants={itemVariants}
        className="flex justify-center"
      >
        <div className="clinical-card border border-verified/30 text-center max-w-xs">
          <div className="text-xs text-verified uppercase tracking-wider mb-1">Output</div>
          <div className="font-heading text-parchment">Clinical Summary</div>
        </div>
      </motion.div>
    </motion.div>
  );
}
