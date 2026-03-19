"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { EcgLine } from "@/components/shared/ecg-line";

/* ============================================
   HERO SECTION
   ============================================
   Landing page hero with statement and CTA.
   Biological precision, zero-slop design.
*/

export function Hero() {
  return (
    <section className="relative min-h-[80vh] flex items-center">
      {/* Background subtle grid */}
      <div 
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: `
            linear-gradient(var(--parchment) 1px, transparent 1px),
            linear-gradient(90deg, var(--parchment) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />

      <div className="max-w-4xl mx-auto text-center relative z-10 px-6">
        {/* Top accent */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8 flex justify-center"
        >
          <span className="badge-serum text-sm tracking-wider uppercase">
            Clinical Table Summarization
          </span>
        </motion.div>

        {/* Main headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="font-serif text-5xl md:text-6xl lg:text-7xl text-parchment mb-6 leading-tight"
        >
          Transform Clinical Data
          <br />
          <span className="text-plasma">Into Insight</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-cortex text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed"
        >
          Dual-pathway ML/DL system for generating precise, clinically-accurate 
          summaries from complex safety tables. Built for researchers and 
          regulatory professionals.
        </motion.p>

        {/* CTA buttons */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <Link href="/summarizer" className="btn-primary text-base">
            Start Summarizing
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
          <Link href="/research" className="btn-secondary text-base">
            View Research
          </Link>
        </motion.div>

        {/* ECG accent line */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-16"
        >
          <EcgLine color="plasma" className="opacity-40" />
        </motion.div>
      </div>
    </section>
  );
}

/* ============================================
   HERO STATS BAR
   ============================================
   Key metrics displayed below hero
*/

interface HeroStat {
  value: string;
  label: string;
}

export function HeroStats({ stats }: { stats: HeroStat[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="bg-marrow/50 border-t border-b border-parchment/5"
    >
      <div className="max-w-6xl mx-auto py-8 px-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              className="text-center"
            >
              <div className="font-mono text-3xl text-parchment mb-1">
                {stat.value}
              </div>
              <div className="text-cortex text-sm uppercase tracking-wider">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
