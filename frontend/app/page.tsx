"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Terminal,
  Zap,
  BrainCircuit,
  FileText,
  ChevronRight,
  CheckCircle2,
  Activity,
  ArrowRight,
  ShieldCheck,
  Search,
  Sparkles,
  XCircle,
  Clock,
  AlertTriangle,
  Layers
} from "lucide-react";

type ApproachType = "ml" | "base_dl" | "finetuned" | "compare";

interface ResultData {
  summary: string;
  model_used: string;
  verified: boolean;
  numeric_accuracy: number;
  inference_time_ms: number;
  warnings: string[];
  tokens_generated?: number;
}

export default function Home() {
  const [tableText, setTableText] = useState("");
  const [approach, setApproach] = useState<ApproachType>("finetuned");
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [results, setResults] = useState<Record<string, ResultData> | null>(null);

  const handleSummarize = async () => {
    if (!tableText.trim()) return;

    setIsSummarizing(true);
    setResults(null);

    try {
      if (approach === "compare") {
        // Use the compare endpoint on port 8001
        const response = await fetch("http://localhost:8001/summarize-compare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ table_text: tableText }),
        });
        const data = await response.json();
        setTimeout(() => {
          setResults({
            ml: data.ml,
            base_dl: data.base_dl,
            finetuned: data.finetuned_dl,
          });
          setIsSummarizing(false);
        }, 400);
      } else {
        // Single mode
        const port = approach === "ml" ? 8000 : 8001;
        const url = `http://localhost:${port}/summarize`;
        const body: any = { table_text: tableText };
        if (port === 8001) {
          body.mode = approach;
        }

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await response.json();
        setTimeout(() => {
          setResults({ [approach]: data });
          setIsSummarizing(false);
        }, 400);
      }
    } catch (error) {
      console.error("Inference failed:", error);
      setTimeout(() => {
        setResults({
          [approach]: {
            summary: "Backend connection failed. Ensure servers are running on ports 8000/8001.",
            model_used: "error",
            verified: false,
            numeric_accuracy: 0,
            inference_time_ms: 0,
            warnings: ["Connection failed"],
          },
        });
        setIsSummarizing(false);
      }, 400);
    }
  };

  const clear = () => {
    setTableText("");
    setResults(null);
  };

  const approachConfig = {
    ml: {
      label: "ML Ensemble",
      subtitle: "LightGBM + Jinja2",
      icon: <Zap className="w-4 h-4" />,
      color: "blue",
      activeClass: "bg-blue-600/10 border-blue-500/50 text-blue-400",
    },
    base_dl: {
      label: "DL Base",
      subtitle: "Flan-T5 (Unfinetuned)",
      icon: <BrainCircuit className="w-4 h-4" />,
      color: "purple",
      activeClass: "bg-purple-600/10 border-purple-500/50 text-purple-400",
    },
    finetuned: {
      label: "DL Fine-tuned",
      subtitle: "LoRA Adapter (830MB)",
      icon: <Sparkles className="w-4 h-4" />,
      color: "emerald",
      activeClass: "bg-emerald-600/10 border-emerald-500/50 text-emerald-400",
    },
    compare: {
      label: "Compare All",
      subtitle: "Side-by-Side",
      icon: <Layers className="w-4 h-4" />,
      color: "amber",
      activeClass: "bg-amber-600/10 border-amber-500/50 text-amber-400",
    },
  };

  const renderResultCard = (key: string, data: ResultData) => {
    const isVerified = data.verified && data.numeric_accuracy >= 0.95;
    const colorMap: Record<string, string> = {
      ml: "blue",
      base_dl: "purple",
      finetuned: "emerald",
    };
    const color = colorMap[key] || "blue";
    const labelMap: Record<string, string> = {
      ml: "ML Ensemble",
      base_dl: "DL Base (Unfinetuned)",
      finetuned: "DL Fine-tuned (LoRA)",
    };

    return (
      <motion.div
        key={key}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: key === "ml" ? 0 : key === "base_dl" ? 0.15 : 0.3 }}
        className="minimal-card relative overflow-hidden group"
      >
        <div className={`absolute top-0 right-0 w-32 h-32 bg-${color}-500/5 blur-3xl -mr-16 -mt-16 group-hover:bg-${color}-500/10 transition-all`} />

        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isVerified ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
              {isVerified ? (
                <ShieldCheck className="w-5 h-5 text-green-500" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-red-400" />
              )}
            </div>
            <div>
              <h3 className="font-bold text-sm">{labelMap[key] || key}</h3>
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                {isVerified ? (
                  <><CheckCircle2 className="w-3 h-3 text-green-500" /> Hallucination Check Passed</>
                ) : (
                  <><XCircle className="w-3 h-3 text-red-400" /> Hallucination Detected</>
                )}
              </p>
            </div>
          </div>
          <span className={`text-[10px] px-2 py-1 rounded-full font-bold uppercase tracking-wider bg-${color}-500/10 text-${color}-400 border border-${color}-500/20`}>
            {data.model_used}
          </span>
        </div>

        <div className="mb-4">
          <p className="text-sm leading-relaxed text-white/85">
            {data.summary}
          </p>
        </div>

        {data.warnings.length > 0 && (
          <div className="mb-4 p-2 rounded-lg bg-amber-500/5 border border-amber-500/10">
            {data.warnings.map((w, i) => (
              <p key={i} className="text-[11px] text-amber-400/80">{w}</p>
            ))}
          </div>
        )}

        <div className="grid grid-cols-3 gap-3 pt-4 border-t border-white/5">
          <div>
            <p className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Inference</p>
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-blue-400" />
              <p className="text-sm font-mono font-bold text-blue-400">{data.inference_time_ms.toFixed(0)}ms</p>
            </div>
          </div>
          <div>
            <p className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Num. Accuracy</p>
            <p className={`text-sm font-mono font-bold ${data.numeric_accuracy >= 0.95 ? 'text-green-400' : 'text-red-400'}`}>
              {(data.numeric_accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Status</p>
            <div className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full ${isVerified ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <span className="text-xs font-semibold">{isVerified ? 'Verified' : 'Failed'}</span>
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

      {/* Left Column: Input */}
      <div className="lg:col-span-5 space-y-6">
        <div className="flex flex-col gap-2">
          <h2 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
            Generate Summary
          </h2>
          <p className="text-muted-foreground text-sm">
            Input linearized clinical tables for regulatory-grade summarization.
          </p>
        </div>

        <div className="minimal-card space-y-4">
          <div className="flex items-center justify-between text-sm">
            <label className="font-medium flex items-center gap-2">
              <Terminal className="w-4 h-4 text-blue-400" />
              Linearized Table Input
            </label>
            <button
              onClick={clear}
              className="text-xs text-muted-foreground hover:text-white transition-colors"
            >
              Clear
            </button>
          </div>

          <textarea
            value={tableText}
            onChange={(e) => setTableText(e.target.value)}
            placeholder="start_table [TABLE_TITLE: Table 1] [HEADERS: | Placebo | Drug] [ROW] TEAE | 10 (5%) | 12 (6%) ... end_table"
            className="w-full h-48 bg-black/40 border border-white/10 rounded-xl p-4 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50 transition-all resize-none"
          />

          <div className="space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-white/40">Select Approach</span>
            <div className="grid grid-cols-2 gap-2">
              {(Object.entries(approachConfig) as [ApproachType, any][]).map(([key, cfg]) => (
                <button
                  key={key}
                  onClick={() => setApproach(key)}
                  className={`flex items-center gap-2 py-2.5 px-3 rounded-xl border transition-all text-left ${approach === key
                      ? cfg.activeClass
                      : "bg-white/5 border-white/5 text-white/60 hover:bg-white/10"
                    }`}
                >
                  {cfg.icon}
                  <div>
                    <div className="text-xs font-semibold">{cfg.label}</div>
                    <div className="text-[10px] text-white/30">{cfg.subtitle}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleSummarize}
            disabled={isSummarizing || !tableText.trim()}
            className="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl font-semibold shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 hover:scale-[1.01] transition-all disabled:opacity-50 disabled:scale-100 flex items-center justify-center gap-2 mt-2"
          >
            {isSummarizing ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                {approach === "compare" ? "Running All 3 Engines..." : "Synthesizing..."}
              </>
            ) : (
              <>
                {approach === "compare" ? "Compare All Engines" : "Synthesize Natural Language"}
                <ChevronRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Right Column: Results */}
      <div className="lg:col-span-7 space-y-4">
        <AnimatePresence mode="wait">
          {!results && !isSummarizing ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="h-full min-h-[500px] flex flex-col items-center justify-center minimal-card border-dashed"
            >
              <div className="w-16 h-16 rounded-3xl bg-white/5 flex items-center justify-center mb-6">
                <Activity className="w-8 h-8 text-white/20" />
              </div>
              <h3 className="text-muted-foreground font-medium">Results will appear here</h3>
              <p className="text-white/20 text-xs mt-2">Enter table data and select an approach to begin</p>
            </motion.div>
          ) : isSummarizing ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="min-h-[500px] flex flex-col items-center justify-center minimal-card"
            >
              <div className="w-16 h-16 rounded-3xl bg-blue-500/10 flex items-center justify-center mb-6">
                <div className="w-8 h-8 border-3 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
              </div>
              <h3 className="text-white/80 font-medium mb-1">
                {approach === "compare" ? "Running ML → DL Base → DL Fine-tuned..." : "Processing..."}
              </h3>
              <p className="text-white/30 text-xs">Generating clinical narrative with hallucination verification</p>
            </motion.div>
          ) : results ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              {approach === "compare" && (
                <div className="text-center mb-2">
                  <span className="text-xs font-bold uppercase tracking-widest text-amber-400/60 bg-amber-500/5 px-4 py-1.5 rounded-full border border-amber-500/10">
                    3-Way Comparison
                  </span>
                </div>
              )}
              {Object.entries(results).map(([key, data]) => renderResultCard(key, data))}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );
}
