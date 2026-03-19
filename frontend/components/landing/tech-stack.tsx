"use client";

import { motion } from "framer-motion";
import { SectionHeader } from "@/components/shared/section-header";
import { EcgDivider } from "@/components/shared/ecg-line";

/* ============================================
   TECH STACK
   ============================================
   Showcases the technologies powering the system
   with clinical-themed styling.
*/

interface TechItem {
  name: string;
  category: string;
  description: string;
}

const techStack: TechItem[] = [
  { name: "Flan-T5-XL", category: "Generation", description: "3B parameter encoder-decoder" },
  { name: "LoRA Adapters", category: "Fine-tuning", description: "Parameter-efficient training" },
  { name: "LightGBM", category: "Classification", description: "Gradient boosting for content selection" },
  { name: "Sentence Transformers", category: "Embeddings", description: "Clinical text representations" },
  { name: "FastAPI", category: "API Layer", description: "Async Python web framework" },
  { name: "Next.js 16", category: "Frontend", description: "React meta-framework" },
];

const infrastructureStack: TechItem[] = [
  { name: "NVIDIA H100", category: "Training", description: "80GB HBM3 GPU" },
  { name: "bitsandbytes", category: "Quantization", description: "8-bit inference" },
  { name: "Hugging Face", category: "Model Hub", description: "Model hosting & serving" },
  { name: "PEFT", category: "Training", description: "Parameter-efficient fine-tuning" },
];

export function TechStack() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="Technology"
          title="Built on Proven Foundations"
          description="Modern ML stack optimized for clinical text processing"
        />

        <div className="space-y-12">
          {/* Core ML Stack */}
          <div>
            <h3 className="font-heading text-lg text-serum mb-6">Core Stack</h3>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {techStack.map((tech, index) => (
                <TechCard key={tech.name} tech={tech} delay={index * 0.05} />
              ))}
            </div>
          </div>

          <EcgDivider color="plasma" />

          {/* Infrastructure */}
          <div>
            <h3 className="font-heading text-lg text-saline mb-6">Infrastructure</h3>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {infrastructureStack.map((tech, index) => (
                <TechCard key={tech.name} tech={tech} delay={0.3 + index * 0.05} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

interface TechCardProps {
  tech: TechItem;
  delay: number;
}

function TechCard({ tech, delay }: TechCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay }}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
      className="clinical-card border border-parchment/10 hover:border-parchment/20 transition-colors"
    >
      <div className="text-xs text-cortex uppercase tracking-wider mb-1">
        {tech.category}
      </div>
      <div className="font-heading text-parchment mb-1">{tech.name}</div>
      <div className="text-sm text-cortex">{tech.description}</div>
    </motion.div>
  );
}

/* ============================================
   FEATURES LIST
   ============================================
   Key capabilities of the system
*/

interface Feature {
  title: string;
  description: string;
  icon: "speed" | "accuracy" | "explain" | "scale";
}

const features: Feature[] = [
  {
    title: "Sub-second ML Inference",
    description: "LightGBM content selection and KNN retrieval enable real-time processing for production workflows.",
    icon: "speed",
  },
  {
    title: "State-of-the-art Quality",
    description: "Fine-tuned Flan-T5-XL achieves 0.76 ROUGE-1 on clinical safety table summarization.",
    icon: "accuracy",
  },
  {
    title: "Full Interpretability",
    description: "ML pipeline provides feature importance scores; DL offers attention visualizations.",
    icon: "explain",
  },
  {
    title: "Scalable Architecture",
    description: "Dual FastAPI services can scale independently based on workload characteristics.",
    icon: "scale",
  },
];

const iconSvgs = {
  speed: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
    </svg>
  ),
  accuracy: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  explain: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  scale: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" />
    </svg>
  ),
};

export function FeaturesList() {
  return (
    <section className="py-24 px-6 bg-marrow/30">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="Capabilities"
          title="Why This Approach"
          description="Purpose-built for clinical regulatory document processing"
        />

        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} delay={index * 0.1} />
          ))}
        </div>
      </div>
    </section>
  );
}

interface FeatureCardProps {
  feature: Feature;
  delay: number;
}

function FeatureCard({ feature, delay }: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className="flex gap-4"
    >
      <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-plasma/10 text-plasma flex items-center justify-center">
        {iconSvgs[feature.icon]}
      </div>
      <div>
        <h3 className="font-heading text-parchment mb-1">{feature.title}</h3>
        <p className="text-cortex text-sm leading-relaxed">{feature.description}</p>
      </div>
    </motion.div>
  );
}
