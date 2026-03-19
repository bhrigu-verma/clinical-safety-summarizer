"use client";

import { motion } from "framer-motion";

/* ============================================
   CENTRIFUGE LOADER
   ============================================
   Biology-inspired loading state.
   A spinning centrifuge separating data.
*/

interface CentrifugeLoaderProps {
  size?: "sm" | "md" | "lg";
  text?: string;
}

const sizeMap = {
  sm: { container: "w-16 h-16", ring: 56, dot: 4 },
  md: { container: "w-24 h-24", ring: 84, dot: 6 },
  lg: { container: "w-32 h-32", ring: 112, dot: 8 },
};

export function CentrifugeLoader({ size = "md", text }: CentrifugeLoaderProps) {
  const { container } = sizeMap[size];

  return (
    <div className="flex flex-col items-center gap-4">
      <div className={`${container} relative`}>
        {/* Outer ring */}
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-plasma/30"
          animate={{ rotate: 360 }}
          transition={{
            duration: 3,
            ease: "linear",
            repeat: Infinity,
          }}
        />

        {/* Middle ring */}
        <motion.div
          className="absolute inset-2 rounded-full border border-saline/40"
          animate={{ rotate: -360 }}
          transition={{
            duration: 2,
            ease: "linear",
            repeat: Infinity,
          }}
        />

        {/* Inner spinning element */}
        <motion.div
          className="absolute inset-4 rounded-full border border-serum/50"
          animate={{ rotate: 360 }}
          transition={{
            duration: 1.5,
            ease: "linear",
            repeat: Infinity,
          }}
        >
          {/* Sample dots */}
          {[0, 90, 180, 270].map((angle) => (
            <motion.div
              key={angle}
              className="absolute w-2 h-2 rounded-full bg-plasma"
              style={{
                top: "50%",
                left: "50%",
                transform: `rotate(${angle}deg) translateY(-50%)`,
                transformOrigin: "center",
              }}
              animate={{
                scale: [1, 1.3, 1],
                opacity: [0.6, 1, 0.6],
              }}
              transition={{
                duration: 1,
                delay: angle / 360,
                repeat: Infinity,
              }}
            />
          ))}
        </motion.div>

        {/* Center core */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          animate={{ scale: [1, 1.1, 1] }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
          }}
        >
          <div className="w-3 h-3 rounded-full bg-plasma/60" />
        </motion.div>
      </div>

      {text && (
        <motion.p
          className="text-cortex text-sm font-mono"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{
            duration: 2,
            repeat: Infinity,
          }}
        >
          {text}
        </motion.p>
      )}
    </div>
  );
}

/* ============================================
   PROCESSING INDICATOR
   ============================================
   Shows pipeline progress steps
*/

interface ProcessingStep {
  label: string;
  status: "pending" | "active" | "complete";
}

interface ProcessingIndicatorProps {
  steps: ProcessingStep[];
  currentStep?: number;
}

export function ProcessingIndicator({ steps }: ProcessingIndicatorProps) {
  return (
    <div className="space-y-3">
      {steps.map((step, index) => (
        <motion.div
          key={step.label}
          className="flex items-center gap-3"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          {/* Status indicator */}
          <div className="relative w-5 h-5 flex items-center justify-center">
            {step.status === "complete" && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-5 h-5 rounded-full bg-verified/20 flex items-center justify-center"
              >
                <svg className="w-3 h-3 text-verified" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                </svg>
              </motion.div>
            )}
            {step.status === "active" && (
              <motion.div
                className="w-5 h-5 rounded-full border-2 border-plasma"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                <motion.div
                  className="w-full h-full rounded-full bg-plasma/30"
                  animate={{ opacity: [0.3, 0.8, 0.3] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              </motion.div>
            )}
            {step.status === "pending" && (
              <div className="w-5 h-5 rounded-full border border-cortex/50" />
            )}
          </div>

          {/* Label */}
          <span
            className={`text-sm font-mono ${
              step.status === "complete"
                ? "text-verified"
                : step.status === "active"
                ? "text-plasma"
                : "text-cortex"
            }`}
          >
            {step.label}
          </span>
        </motion.div>
      ))}
    </div>
  );
}
