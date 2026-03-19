"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { EcgLine } from "../shared/ecg-line";

/* ============================================
   RESULT CARD
   ============================================
   Displays generated summary with metadata
*/

interface ResultCardProps {
  summary: string;
  approach: string;
  processingTime?: number;
  confidence?: number;
}

const APPROACH_META: Record<string, { label: string; className: string }> = {
  ml: { label: "ML Pipeline", className: "bg-serum/15 text-serum border border-serum/30" },
  base_dl: { label: "DL Base", className: "bg-saline/15 text-saline border border-saline/30" },
  finetuned: { label: "DL Fine-Tuned", className: "bg-verified/15 text-verified border border-verified/30" },
  compare: { label: "Compare", className: "bg-plasma/15 text-plasma border border-plasma/30" },
};

export function ResultCard({
  summary,
  approach,
  processingTime,
  confidence,
}: ResultCardProps) {
  const [copied, setCopied] = useState(false);
  const approachMeta = APPROACH_META[approach] ?? {
    label: approach.toUpperCase(),
    className: "bg-white/10 text-parchment border border-white/20",
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="clinical-card"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-mono uppercase tracking-wider ${approachMeta.className}`}
          >
            {approachMeta.label}
          </span>
          {processingTime && (
            <span className="text-xs text-cortex">
              {processingTime < 1000
                ? `${processingTime}ms`
                : `${(processingTime / 1000).toFixed(1)}s`}
            </span>
          )}
        </div>

        <button
          onClick={handleCopy}
          className="text-xs text-parchment/60 hover:text-parchment transition-colors flex items-center gap-1.5"
        >
          <AnimatePresence mode="wait">
            {copied ? (
              <motion.span
                key="check"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="text-verified"
              >
                <CheckIcon />
              </motion.span>
            ) : (
              <motion.span
                key="copy"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <CopyIcon />
              </motion.span>
            )}
          </AnimatePresence>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      {/* Summary content */}
      <div className="prose prose-invert prose-sm max-w-none">
        <p className="text-parchment/90 leading-relaxed whitespace-pre-wrap">
          {summary}
        </p>
      </div>

      {/* Metadata footer */}
      {confidence && (
        <div className="mt-4 pt-4 border-t border-parchment/10">
          <div className="flex items-center justify-between text-xs">
            <span className="text-cortex">Confidence Score</span>
            <ConfidenceMeter value={confidence} />
          </div>
        </div>
      )}
    </motion.div>
  );
}

/* ============================================
   RESULT SKELETON
   ============================================
   Loading state while generating
*/

export function ResultSkeleton() {
  return (
    <div className="clinical-card">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-24 h-5 bg-parchment/5 rounded animate-pulse" />
        <div className="w-12 h-5 bg-parchment/5 rounded animate-pulse" />
      </div>

      <div className="space-y-3">
        <div className="w-full h-4 bg-parchment/5 rounded animate-pulse" />
        <div className="w-11/12 h-4 bg-parchment/5 rounded animate-pulse" />
        <div className="w-4/5 h-4 bg-parchment/5 rounded animate-pulse" />
        <div className="w-full h-4 bg-parchment/5 rounded animate-pulse" />
        <div className="w-3/4 h-4 bg-parchment/5 rounded animate-pulse" />
      </div>

      <div className="flex justify-center mt-6">
        <EcgLine animated className="w-32" />
      </div>
    </div>
  );
}

/* ============================================
   EMPTY STATE
   ============================================
   Shown before any summary is generated
*/

export function EmptyResult() {
  return (
    <div className="clinical-card border border-dashed border-parchment/20">
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <div className="w-12 h-12 rounded-full bg-parchment/5 flex items-center justify-center mb-4">
          <DocumentIcon />
        </div>
        <h3 className="font-heading text-parchment mb-1">No Summary Yet</h3>
        <p className="text-sm text-cortex max-w-xs">
          Paste a clinical safety table and click &quot;Generate Summary&quot; to see
          results from the selected pipeline.
        </p>
      </div>
    </div>
  );
}

/* ============================================
   ERROR STATE
   ============================================ */

interface ErrorResultProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorResult({ message, onRetry }: ErrorResultProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="clinical-card border border-plasma/30 bg-plasma/5"
    >
      <div className="flex items-start gap-3">
        <div className="p-1 rounded bg-plasma/20 text-plasma shrink-0">
          <AlertIcon />
        </div>
        <div className="flex-1">
          <h3 className="font-heading text-plasma mb-1">Generation Failed</h3>
          <p className="text-sm text-parchment/70">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 text-sm text-plasma hover:text-plasma/80 underline underline-offset-2"
            >
              Try Again
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

/* ============================================
   CONFIDENCE METER
   ============================================ */

function ConfidenceMeter({ value }: { value: number }) {
  const percentage = Math.round(value * 100);
  const color =
    percentage >= 80
      ? "bg-verified"
      : percentage >= 60
      ? "bg-serum"
      : "bg-hazard";

  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-1.5 bg-parchment/10 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className={`h-full ${color} rounded-full`}
        />
      </div>
      <span className="text-parchment/70 tabular-nums">{percentage}%</span>
    </div>
  );
}

/* ============================================
   ICONS
   ============================================ */

function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20,6 9,17 4,12" />
    </svg>
  );
}

function DocumentIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-cortex">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14,2 14,8 20,8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <line x1="10" y1="9" x2="8" y2="9" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}
