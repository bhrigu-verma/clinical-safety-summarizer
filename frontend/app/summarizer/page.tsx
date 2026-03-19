"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  ArrowLeft,
  FlaskConical,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Sparkles,
  Download,
  Copy,
  RefreshCw,
} from "lucide-react";

import { TableInput, TablePreview } from "@/components/summarizer/table-input";
import { ResultCard, EmptyResult, ErrorResult } from "@/components/summarizer/result-card";
import { CentrifugeLoader } from "@/components/shared/loaders";


type SummaryMode = "ml" | "base_dl" | "finetuned" | "compare";

type ResultItem = {
  mode: Exclude<SummaryMode, "compare">;
  summary: string;
  inferenceMs?: number;
  numericAccuracy?: number;
  warnings: string[];
};

type SingleApiResponse = {
  summary: string;
  inference_time_ms?: number;
  numeric_accuracy?: number;
  hallucination_flags?: string[];
};

type CompareApiResponse = {
  results: {
    ml?: SingleApiResponse;
    base_dl?: SingleApiResponse;
    finetuned?: SingleApiResponse;
    finetuned_dl?: SingleApiResponse;
  };
};

type HealthState = {
  ml: boolean;
  dl: boolean;
  loading: boolean;
  lastChecked?: string;
};

type HistoryEntry = {
  id: string;
  createdAt: string;
  mode: SummaryMode;
  inputPreview: string;
  tableText: string;
  results: ResultItem[];
};

type ToastItem = {
  id: string;
  message: string;
  tone: "success" | "error" | "info";
};

const HISTORY_KEY = "clinicalsafe.summarizer.history.v1";

const MODE_META: Record<SummaryMode, { title: string; tone: string; description: string }> = {
  ml: {
    title: "ML",
    tone: "text-serum",
    description: "Fast deterministic generation from feature pipeline.",
  },
  base_dl: {
    title: "DL Base",
    tone: "text-saline",
    description: "Transformer generation without domain fine-tuning.",
  },
  finetuned: {
    title: "DL Fine-Tuned",
    tone: "text-verified",
    description: "LoRA-adapted generation with verification guardrails.",
  },
  compare: {
    title: "Compare",
    tone: "text-plasma",
    description: "Run ML, DL Base, and DL Fine-Tuned side by side.",
  },
};

function normalizeResult(mode: Exclude<SummaryMode, "compare">, data: SingleApiResponse): ResultItem {
  return {
    mode,
    summary: data.summary,
    inferenceMs: data.inference_time_ms,
    numericAccuracy: data.numeric_accuracy,
    warnings: data.hallucination_flags ?? [],
  };
}

export default function SummarizerPage() {
  const [tableText, setTableText] = useState("");
  const [mode, setMode] = useState<SummaryMode>("ml");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthState>({ ml: false, dl: false, loading: true });
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyQuery, setHistoryQuery] = useState("");
  const [historyMode, setHistoryMode] = useState<"all" | SummaryMode>("all");
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const canSubmit = tableText.trim().length > 0 && !isLoading;

  const sortedResults = useMemo(
    () =>
      [...results].sort(
        (a, b) =>
          ["ml", "base_dl", "finetuned"].indexOf(a.mode) -
          ["ml", "base_dl", "finetuned"].indexOf(b.mode)
      ),
    [results]
  );

  const filteredHistory = useMemo(() => {
    return history.filter((entry) => {
      const modeMatch = historyMode === "all" || entry.mode === historyMode;
      const query = historyQuery.trim().toLowerCase();
      const queryMatch =
        query.length === 0 ||
        entry.inputPreview.toLowerCase().includes(query) ||
        MODE_META[entry.mode].title.toLowerCase().includes(query);
      return modeMatch && queryMatch;
    });
  }, [history, historyMode, historyQuery]);

  const pushToast = useCallback((message: string, tone: ToastItem["tone"]) => {
    const toast: ToastItem = { id: `${Date.now()}-${Math.random()}`, message, tone };
    setToasts((prev) => [...prev, toast]);
  }, []);

  useEffect(() => {
    if (toasts.length === 0) return;
    const timer = setTimeout(() => {
      setToasts((prev) => prev.slice(1));
    }, 2200);
    return () => clearTimeout(timer);
  }, [toasts]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as HistoryEntry[];
      if (Array.isArray(parsed)) {
        setHistory(parsed.slice(0, 8));
      }
    } catch {
      setHistory([]);
    }
  }, []);

  const persistHistory = useCallback((entries: HistoryEntry[]) => {
    setHistory(entries);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, 8)));
  }, []);

  const checkBackends = useCallback(async () => {
    setHealth((prev) => ({ ...prev, loading: true }));
    const [mlOk, dlOk] = await Promise.all([
      fetch("http://localhost:8000/health").then((r) => r.ok).catch(() => false),
      fetch("http://localhost:8001/health").then((r) => r.ok).catch(() => false),
    ]);

    setHealth({
      ml: mlOk,
      dl: dlOk,
      loading: false,
      lastChecked: new Date().toISOString(),
    });
    if (mlOk && dlOk) {
      pushToast("Both backends are online.", "success");
    } else {
      pushToast("One or more backends are offline.", "error");
    }
  }, [pushToast]);

  useEffect(() => {
    void checkBackends();
  }, [checkBackends]);

  const runSummarization = useCallback(async () => {
    if (!tableText.trim()) {
      setError("Please add clinical table content first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults([]);

    try {
      let outputResults: ResultItem[] = [];

      if (mode === "ml") {
        const res = await fetch("http://localhost:8000/summarize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ table_text: tableText }),
        });

        if (!res.ok) throw new Error(`ML backend request failed (${res.status})`);
        const data: SingleApiResponse = await res.json();
        outputResults = [normalizeResult("ml", data)];
      }

      if (mode === "base_dl" || mode === "finetuned") {
        const res = await fetch(`http://localhost:8001/summarize?mode=${mode}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ table_text: tableText }),
        });

        if (!res.ok) throw new Error(`DL backend request failed (${res.status})`);
        const data: SingleApiResponse = await res.json();
        outputResults = [normalizeResult(mode, data)];
      }

      if (mode === "compare") {
        const res = await fetch("http://localhost:8001/summarize-compare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ table_text: tableText }),
        });

        if (!res.ok) throw new Error(`Compare request failed (${res.status})`);
        const data: CompareApiResponse = await res.json();

        const comparisonResults: ResultItem[] = [];
        if (data.results.ml) comparisonResults.push(normalizeResult("ml", data.results.ml));
        if (data.results.base_dl) comparisonResults.push(normalizeResult("base_dl", data.results.base_dl));
        if (data.results.finetuned) comparisonResults.push(normalizeResult("finetuned", data.results.finetuned));
        if (data.results.finetuned_dl) comparisonResults.push(normalizeResult("finetuned", data.results.finetuned_dl));

        outputResults = comparisonResults;
      }

      setResults(outputResults);
      const newEntry: HistoryEntry = {
        id: `${Date.now()}`,
        createdAt: new Date().toISOString(),
        mode,
        inputPreview: tableText.slice(0, 120),
        tableText,
        results: outputResults,
      };
      persistHistory([newEntry, ...history].slice(0, 8));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error while generating summary.");
    } finally {
      setIsLoading(false);
    }
  }, [history, mode, persistHistory, tableText]);

  const exportMarkdown = useMemo(() => {
    if (sortedResults.length === 0) return "";
    const header = `# ClinicalSafe Run\n\n- Mode: ${MODE_META[mode].title}\n- Timestamp: ${new Date().toISOString()}\n\n`;
    const body = sortedResults
      .map((item) => {
        const accuracy = typeof item.numericAccuracy === "number" ? `${Math.round(item.numericAccuracy * 100)}%` : "N/A";
        const warnings = item.warnings.length > 0 ? item.warnings.map((w) => `  - ${w}`).join("\n") : "  - None";
        return `## ${item.mode}\n\n${item.summary}\n\n- Inference: ${item.inferenceMs ?? "N/A"}ms\n- Numeric accuracy: ${accuracy}\n- Warnings:\n${warnings}`;
      })
      .join("\n\n");
    return `${header}${body}`;
  }, [mode, sortedResults]);

  const handleCopyExport = useCallback(async () => {
    if (!exportMarkdown) return;
    await navigator.clipboard.writeText(exportMarkdown);
    pushToast("Run markdown copied.", "success");
  }, [exportMarkdown, pushToast]);

  const handleDownloadExport = useCallback(() => {
    if (!exportMarkdown) return;
    const blob = new Blob([exportMarkdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `clinicalsafe-run-${Date.now()}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
    pushToast("Run markdown downloaded.", "success");
  }, [exportMarkdown, pushToast]);

  const exportSingleResult = useCallback(
    async (item: ResultItem, action: "copy" | "download") => {
      const accuracy = typeof item.numericAccuracy === "number" ? `${Math.round(item.numericAccuracy * 100)}%` : "N/A";
      const warnings = item.warnings.length > 0 ? item.warnings.map((w) => `- ${w}`).join("\n") : "- None";
      const md = `# ClinicalSafe Result\n\n- Mode: ${item.mode}\n- Inference: ${item.inferenceMs ?? "N/A"}ms\n- Numeric accuracy: ${accuracy}\n\n## Summary\n\n${item.summary}\n\n## Warnings\n${warnings}\n`;

      if (action === "copy") {
        await navigator.clipboard.writeText(md);
        pushToast(`${MODE_META[item.mode].title} result copied.`, "info");
        return;
      }

      const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `clinicalsafe-${item.mode}-${Date.now()}.md`;
      anchor.click();
      URL.revokeObjectURL(url);
      pushToast(`${MODE_META[item.mode].title} result downloaded.`, "info");
    },
    [pushToast]
  );

  return (
    <div className="min-h-screen bg-ink">
      <div className="border-b border-white/10">
        <div className="section-container py-4">
          <Link href="/" className="inline-flex items-center gap-2 text-cortex hover:text-parchment transition-colors text-sm">
            <ArrowLeft className="w-4 h-4" />
            Back to Overview
          </Link>
        </div>
      </div>

      <main className="section-container py-10 space-y-8">
        <header className="space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full border border-plasma/30 bg-plasma/10 px-3 py-1 text-xs text-plasma">
            <FlaskConical className="h-3.5 w-3.5" />
            Summarization Workspace
          </div>
          <h1 className="font-display text-4xl md:text-5xl text-parchment">Clinical Table to Narrative</h1>
          <p className="max-w-3xl text-cortex">
            Select a model pathway, submit your safety table, and inspect generated outputs with numeric verification metadata.
          </p>

          <div className="clinical-card p-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-wider text-cortex">Backend Health</p>
                <p className="mt-1 text-sm text-parchment/80">
                  ML (8000): <span className={health.ml ? "text-verified" : "text-hazard"}>{health.ml ? "Online" : "Offline"}</span>
                  {" | "}
                  DL (8001): <span className={health.dl ? "text-verified" : "text-hazard"}>{health.dl ? "Online" : "Offline"}</span>
                </p>
                {health.lastChecked ? (
                  <p className="mt-1 text-xs text-cortex">Last checked: {new Date(health.lastChecked).toLocaleTimeString()}</p>
                ) : null}
              </div>
              <button type="button" className="btn-secondary" onClick={checkBackends} disabled={health.loading}>
                <RefreshCw className={`h-4 w-4 ${health.loading ? "animate-spin" : ""}`} />
                Refresh
              </button>
            </div>
          </div>
        </header>

        <section className="grid gap-8 lg:grid-cols-[1.2fr_1fr]">
          <div className="space-y-6">
            <div className="clinical-card p-5">
              <p className="mb-4 text-xs uppercase tracking-wider text-cortex">Model Path</p>
              <div className="grid gap-3 sm:grid-cols-2">
                {(Object.keys(MODE_META) as SummaryMode[]).map((key) => {
                  const meta = MODE_META[key];
                  const selected = mode === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setMode(key)}
                      className={`rounded-lg border p-3 text-left transition-colors ${
                        selected ? "border-white/40 bg-white/5" : "border-white/10 hover:border-white/20"
                      }`}
                    >
                      <p className={`font-heading text-sm ${meta.tone}`}>{meta.title}</p>
                      <p className="mt-1 text-xs text-cortex">{meta.description}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="clinical-card p-5">
              <TableInput value={tableText} onChange={setTableText} disabled={isLoading} />
              <div className="mt-4 flex gap-3">
                <button onClick={runSummarization} disabled={!canSubmit} className="btn-primary">
                  <Sparkles className="h-4 w-4" />
                  Generate
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  disabled={isLoading || (!tableText && results.length === 0)}
                  onClick={() => {
                    setTableText("");
                    setResults([]);
                    setError(null);
                  }}
                >
                  Reset
                </button>
              </div>
            </div>

            {tableText.trim().length > 0 ? (
              <div className="clinical-card p-5">
                <TablePreview content={tableText} />
              </div>
            ) : null}
          </div>

          <div className="space-y-5">
            <div className="clinical-card p-5">
              <p className="text-xs uppercase tracking-wider text-cortex">Run Status</p>
              <p className="mt-2 text-sm text-parchment/80">
                Current mode: <span className={MODE_META[mode].tone}>{MODE_META[mode].title}</span>
              </p>
              <p className="mt-2 text-xs text-cortex">Backends expected: ML on port 8000, DL on port 8001.</p>

              {sortedResults.length > 0 ? (
                <div className="mt-4 flex gap-2">
                  <button type="button" className="btn-secondary" onClick={handleCopyExport}>
                    <Copy className="h-4 w-4" />
                    Copy Markdown
                  </button>
                  <button type="button" className="btn-secondary" onClick={handleDownloadExport}>
                    <Download className="h-4 w-4" />
                    Download
                  </button>
                </div>
              ) : null}
            </div>

            <div className="clinical-card p-5">
              <p className="text-xs uppercase tracking-wider text-cortex">Recent Runs</p>
              <div className="mt-3 grid grid-cols-1 gap-2">
                <input
                  value={historyQuery}
                  onChange={(e) => setHistoryQuery(e.target.value)}
                  placeholder="Search runs..."
                  className="input-clinical py-2 text-xs"
                />
                <select
                  value={historyMode}
                  onChange={(e) => setHistoryMode(e.target.value as "all" | SummaryMode)}
                  className="input-clinical py-2 text-xs"
                >
                  <option value="all">All modes</option>
                  <option value="ml">ML</option>
                  <option value="base_dl">DL Base</option>
                  <option value="finetuned">DL Fine-Tuned</option>
                  <option value="compare">Compare</option>
                </select>
              </div>

              {filteredHistory.length === 0 ? (
                <p className="mt-2 text-sm text-cortex">No runs in this browser session yet.</p>
              ) : (
                <ul className="mt-3 space-y-2">
                  {filteredHistory.map((entry) => (
                    <li key={entry.id} className="rounded-lg border border-white/10 p-3">
                      <p className="text-xs text-cortex">{new Date(entry.createdAt).toLocaleString()}</p>
                      <p className="mt-1 text-sm text-parchment/90">{MODE_META[entry.mode].title}</p>
                      <p className="mt-1 text-xs text-cortex line-clamp-2">{entry.inputPreview}</p>
                      <button
                        type="button"
                        className="mt-2 text-xs text-saline hover:text-saline/80"
                        onClick={() => {
                          setMode(entry.mode);
                          setTableText(entry.tableText);
                          setResults(entry.results);
                          setError(null);
                        }}
                      >
                        Load Run
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <AnimatePresence mode="wait">
              {isLoading ? (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="clinical-card p-8">
                  <CentrifugeLoader size="md" text="Processing request" />
                </motion.div>
              ) : error ? (
                <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <ErrorResult message={error} onRetry={runSummarization} />
                </motion.div>
              ) : sortedResults.length === 0 ? (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <EmptyResult />
                </motion.div>
              ) : (
                <motion.div key="results" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                  {sortedResults.map((item) => (
                    <div key={item.mode} className="space-y-3">
                      <ResultCard
                        summary={item.summary}
                        approach={item.mode}
                        processingTime={item.inferenceMs}
                        confidence={item.numericAccuracy}
                      />

                      <div className="flex gap-2">
                        <button type="button" className="btn-secondary" onClick={() => void exportSingleResult(item, "copy")}>
                          <Copy className="h-4 w-4" />
                          Copy Result
                        </button>
                        <button type="button" className="btn-secondary" onClick={() => void exportSingleResult(item, "download")}>
                          <Download className="h-4 w-4" />
                          Download Result
                        </button>
                      </div>

                      <div className="clinical-card p-4">
                        <div className="mb-3 flex items-center justify-between">
                          <p className="text-xs uppercase tracking-wider text-cortex">Verification</p>
                          {item.warnings.length === 0 ? (
                            <span className="inline-flex items-center gap-1 text-xs text-verified">
                              <CheckCircle2 className="h-3.5 w-3.5" />
                              Clean
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-xs text-hazard">
                              <XCircle className="h-3.5 w-3.5" />
                              Review
                            </span>
                          )}
                        </div>

                        {typeof item.numericAccuracy === "number" ? (
                          <p className="text-sm text-parchment/80">
                            Numeric accuracy: <span className="font-mono text-parchment">{Math.round(item.numericAccuracy * 100)}%</span>
                          </p>
                        ) : (
                          <p className="text-sm text-cortex">Numeric accuracy metadata unavailable for this mode.</p>
                        )}

                        {item.warnings.length > 0 ? (
                          <ul className="mt-3 space-y-2">
                            {item.warnings.map((warning, idx) => (
                              <li key={`${item.mode}-${idx}`} className="flex items-start gap-2 text-xs text-hazard/90">
                                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
                                {warning}
                              </li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>

        <div className="pointer-events-none fixed bottom-5 right-5 z-[90] flex w-80 flex-col gap-2">
          <AnimatePresence>
            {toasts.map((toast) => (
              <motion.div
                key={toast.id}
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 30 }}
                className={`rounded-lg border p-3 text-sm shadow-lg ${
                  toast.tone === "success"
                    ? "border-verified/40 bg-verified/15 text-verified"
                    : toast.tone === "error"
                    ? "border-hazard/40 bg-hazard/10 text-hazard"
                    : "border-saline/40 bg-saline/10 text-saline"
                }`}
              >
                {toast.message}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
