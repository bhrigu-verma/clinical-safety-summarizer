"use client";

import { motion, AnimatePresence } from "framer-motion";

/* ============================================
   APPROACH SELECTOR
   ============================================
   Toggle between ML (fast) and DL (quality) pipelines.
   Each has distinct characteristics displayed.
*/

export type Approach = "ml" | "dl";

interface ApproachSelectorProps {
  value: Approach;
  onChange: (value: Approach) => void;
}

export function ApproachSelector({ value, onChange }: ApproachSelectorProps) {
  return (
    <div className="space-y-4">
      <div className="text-xs text-cortex uppercase tracking-wider">
        Processing Pipeline
      </div>
      
      <div className="flex gap-3">
        <ApproachButton
          approach="ml"
          selected={value === "ml"}
          onClick={() => onChange("ml")}
          label="ML Pipeline"
          badge="Fast"
          description="LightGBM + KNN retrieval"
          stats={{ latency: "<100ms", rouge: "0.71" }}
        />
        <ApproachButton
          approach="dl"
          selected={value === "dl"}
          onClick={() => onChange("dl")}
          label="DL Pipeline"
          badge="Quality"
          description="Fine-tuned Flan-T5-XL"
          stats={{ latency: "~2s", rouge: "0.76" }}
        />
      </div>
    </div>
  );
}

interface ApproachButtonProps {
  approach: Approach;
  selected: boolean;
  onClick: () => void;
  label: string;
  badge: string;
  description: string;
  stats: { latency: string; rouge: string };
}

function ApproachButton({
  approach,
  selected,
  onClick,
  label,
  badge,
  description,
  stats,
}: ApproachButtonProps) {
  const accentColor = approach === "ml" ? "serum" : "saline";
  
  return (
    <motion.button
      onClick={onClick}
      whileTap={{ scale: 0.98 }}
      className={`
        flex-1 p-4 rounded-lg border text-left transition-colors
        ${selected
          ? `border-${accentColor} bg-${accentColor}/5`
          : "border-parchment/10 hover:border-parchment/20"
        }
      `}
      style={{
        borderColor: selected 
          ? (approach === "ml" ? "var(--color-serum)" : "var(--color-saline)")
          : undefined,
        backgroundColor: selected
          ? (approach === "ml" ? "rgba(201, 168, 76, 0.05)" : "rgba(107, 163, 190, 0.05)")
          : undefined,
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-heading text-parchment">{label}</span>
        <span
          className="text-xs px-2 py-0.5 rounded"
          style={{
            backgroundColor: approach === "ml" 
              ? "rgba(201, 168, 76, 0.15)" 
              : "rgba(107, 163, 190, 0.15)",
            color: approach === "ml" ? "var(--color-serum)" : "var(--color-saline)",
          }}
        >
          {badge}
        </span>
      </div>
      
      <div className="text-sm text-cortex mb-3">
        {description}
      </div>
      
      <div className="flex gap-4 text-xs">
        <div>
          <span className="text-cortex">Latency: </span>
          <span className="text-parchment font-mono">{stats.latency}</span>
        </div>
        <div>
          <span className="text-cortex">ROUGE-1: </span>
          <span className="text-parchment font-mono">{stats.rouge}</span>
        </div>
      </div>

      {/* Selection indicator */}
      <AnimatePresence>
        {selected && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="absolute top-2 right-2 w-2 h-2 rounded-full"
            style={{
              backgroundColor: approach === "ml" ? "var(--color-serum)" : "var(--color-saline)",
            }}
          />
        )}
      </AnimatePresence>
    </motion.button>
  );
}

/* ============================================
   APPROACH INFO PANEL
   ============================================
   Displays detailed info about selected approach
*/

interface ApproachInfoProps {
  approach: Approach;
}

export function ApproachInfo({ approach }: ApproachInfoProps) {
  const info = approach === "ml" ? mlInfo : dlInfo;
  const accentColor = approach === "ml" ? "var(--color-serum)" : "var(--color-saline)";

  return (
    <motion.div
      key={approach}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="clinical-card border border-parchment/10"
    >
      <div className="flex items-center gap-2 mb-3">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: accentColor }}
        />
        <h3 className="font-heading text-parchment">{info.title}</h3>
      </div>
      
      <p className="text-cortex text-sm mb-4">{info.description}</p>
      
      <div className="space-y-2">
        <h4 className="text-xs text-cortex uppercase tracking-wider">Pipeline Steps</h4>
        <ol className="space-y-1">
          {info.steps.map((step, i) => (
            <li key={i} className="text-sm text-parchment/80 flex items-start gap-2">
              <span
                className="text-xs font-mono mt-0.5"
                style={{ color: accentColor }}
              >
                {String(i + 1).padStart(2, "0")}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </div>
    </motion.div>
  );
}

const mlInfo = {
  title: "Machine Learning Pipeline",
  description:
    "Fast, interpretable approach using classical ML. Best for high-throughput processing where latency matters.",
  steps: [
    "Extract features from table structure and content",
    "LightGBM classifies content relevance (recall: 0.94)",
    "KNN retrieves similar examples from training corpus",
    "Template-based generation fills slots from matched examples",
  ],
};

const dlInfo = {
  title: "Deep Learning Pipeline",
  description:
    "State-of-the-art quality using fine-tuned transformer. Best when summary quality is the priority.",
  steps: [
    "Tokenize input with SentencePiece vocabulary",
    "Encode table content through Flan-T5-XL encoder",
    "Apply LoRA adapter weights (rank 64, trained on clinical data)",
    "Autoregressive decoding with beam search (n=4)",
  ],
};
