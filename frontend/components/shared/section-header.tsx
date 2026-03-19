"use client";

import { motion } from "framer-motion";

/* ============================================
   SECTION HEADER - Consistent page sections
   ============================================
   Medical literature uses clear hierarchies.
   This component provides that structure.
*/

interface SectionHeaderProps {
  label?: string;
  badge?: string;
  title: string;
  description?: string;
  align?: "left" | "center";
  className?: string;
}

export function SectionHeader({
  label,
  badge,
  title,
  description,
  align = "left",
  className = "",
}: SectionHeaderProps) {
  const alignClass = align === "center" ? "text-center mx-auto" : "text-left";
  const headerLabel = label ?? badge;

  return (
    <motion.div
      className={`max-w-2xl ${alignClass} ${className}`}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.5 }}
    >
      {headerLabel && (
        <span className="inline-block text-xs font-mono uppercase tracking-wider text-plasma mb-3">
          {headerLabel}
        </span>
      )}
      <h2 className="font-display text-3xl md:text-4xl text-parchment mb-4">
        {title}
      </h2>
      {description && (
        <p className="text-cortex text-base md:text-lg leading-relaxed">
          {description}
        </p>
      )}
    </motion.div>
  );
}

/* ============================================
   STAT BLOCK - Metric display
   ============================================ */

interface StatBlockProps {
  value: string;
  label: string;
  sublabel?: string;
  variant?: "default" | "plasma" | "verified" | "serum";
}

const variantMap = {
  default: "text-parchment",
  plasma: "text-plasma",
  verified: "text-verified",
  serum: "text-serum",
};

export function StatBlock({
  value,
  label,
  sublabel,
  variant = "default",
}: StatBlockProps) {
  return (
    <motion.div
      className="text-center"
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
    >
      <div className={`font-display text-4xl md:text-5xl ${variantMap[variant]} mb-2`}>
        {value}
      </div>
      <div className="font-heading font-medium text-parchment text-sm">
        {label}
      </div>
      {sublabel && (
        <div className="text-cortex text-xs mt-1">{sublabel}</div>
      )}
    </motion.div>
  );
}

/* ============================================
   BADGE - Status indicators
   ============================================ */

interface BadgeProps {
  children: React.ReactNode;
  variant?: "verified" | "hazard" | "neutral" | "plasma";
  size?: "sm" | "md";
}

export function Badge({ 
  children, 
  variant = "neutral",
  size = "sm" 
}: BadgeProps) {
  const baseClasses = "inline-flex items-center gap-1.5 font-mono uppercase tracking-wider";
  
  const variantClasses = {
    verified: "badge-verified",
    hazard: "badge-hazard", 
    neutral: "badge-neutral",
    plasma: "bg-plasma/15 text-plasma border border-plasma/30",
  };
  
  const sizeClasses = {
    sm: "px-2 py-1 text-[10px]",
    md: "px-3 py-1.5 text-xs",
  };

  return (
    <span className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} rounded`}>
      {children}
    </span>
  );
}

/* ============================================
   CLINICAL CARD - Content container
   ============================================ */

interface ClinicalCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
}

export function ClinicalCard({ 
  children, 
  className = "",
  hover = true 
}: ClinicalCardProps) {
  return (
    <motion.div
      className={`clinical-card ${hover ? "hover:border-plasma/40" : ""} ${className}`}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
      whileHover={hover ? { y: -4 } : undefined}
    >
      {children}
    </motion.div>
  );
}
