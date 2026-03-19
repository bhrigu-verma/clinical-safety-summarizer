"use client";

import { motion } from "framer-motion";

/* ============================================
   ECG LINE - Signature Clinical Animation
   ============================================
   The heartbeat of the Clinical Ink design system.
   A subtle, continuous pulse that shows the app is alive.
*/

interface EcgLineProps {
  className?: string;
  color?: "plasma" | "verified" | "saline" | "serum";
  width?: "sm" | "md" | "lg" | "full";
  animated?: boolean;
}

const colorMap = {
  plasma: "stroke-plasma",
  verified: "stroke-verified",
  saline: "stroke-saline",
  serum: "stroke-serum",
};

const widthMap = {
  sm: "w-32",
  md: "w-48",
  lg: "w-64",
  full: "w-full",
};

export function EcgLine({
  className = "",
  color = "plasma",
  width = "md",
  animated = true,
}: EcgLineProps) {
  // Standard ECG waveform path
  const pathD = "M0 20 L10 20 L14 20 L16 8 L20 32 L24 8 L26 20 L30 20 L40 20";

  return (
    <div className={`${widthMap[width]} ${className}`}>
      <svg
        viewBox="0 0 40 40"
        className={`w-full h-auto ${colorMap[color]}`}
        fill="none"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        preserveAspectRatio="xMidYMid meet"
      >
        {animated ? (
          <motion.path
            d={pathD}
            initial={{ pathLength: 0, opacity: 0-5 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{
              duration: 1.5,
              ease: "easeInOut",
              repeat: Infinity,
              repeatType: "loop",
              repeatDelay: 0.5,
            }}
          />
        ) : (
          <path d={pathD} />
        )}
      </svg>
    </div>
  );
}

/* ============================================
   ECG DIVIDER - Horizontal rule variant
   ============================================ */

interface EcgDividerProps {
  className?: string;
  color?: "plasma" | "verified" | "saline" | "serum";
}

export function EcgDivider({ className = "", color = "plasma" }: EcgDividerProps) {
  return (
    <div className={`flex items-center gap-4 ${className}`}>
      <div className="flex-1 h-px bg-white/8" />
      <EcgLine color={color} width="sm" animated={false} />
      <div className="flex-1 h-px bg-white/8" />
    </div>
  );
}

/* ============================================
   ECG PULSE - Loading indicator
   ============================================ */

interface EcgPulseProps {
  size?: "sm" | "md" | "lg";
  color?: "plasma" | "verified" | "saline" | "serum";
}

const sizeMap = {
  sm: "w-16 h-8",
  md: "w-24 h-12",
  lg: "w-32 h-16",
};

export function EcgPulse({ size = "md", color = "plasma" }: EcgPulseProps) {
  return (
    <div className={`${sizeMap[size]} relative overflow-hidden`}>
      <motion.div
        className="absolute inset-0"
        animate={{ x: ["-100%", "100%"] }}
        transition={{
          duration: 2,
          ease: "linear",
          repeat: Infinity,
        }}
      >
        <svg
          viewBox="0 0 80 40"
          className={`w-full h-full ${colorMap[color]}`}
          fill="none"
          strokeWidth="2"
          strokeLinecap="round"
        >
          <path d="M0 20 L20 20 L28 20 L32 8 L40 32 L48 8 L52 20 L60 20 L80 20" />
        </svg>
      </motion.div>
    </div>
  );
}
