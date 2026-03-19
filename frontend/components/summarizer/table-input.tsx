"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";

/* ============================================
   TABLE INPUT
   ============================================
   Accepts clinical safety table data in various formats.
   Supports paste, file upload, and example loading.
*/

interface TableInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function TableInput({ value, onChange, disabled }: TableInputProps) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        readFile(file, onChange);
      }
    },
    [onChange]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        readFile(e.target.files[0], onChange);
      }
    },
    [onChange]
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs text-cortex uppercase tracking-wider">
          Clinical Safety Table
        </label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => onChange(exampleTable)}
            disabled={disabled}
            className="text-xs text-saline hover:text-saline/80 transition-colors disabled:opacity-50"
          >
            Load Example
          </button>
          <span className="text-parchment/20">|</span>
          <label className="text-xs text-plasma hover:text-plasma/80 transition-colors cursor-pointer">
            Upload File
            <input
              type="file"
              accept=".txt,.csv,.json"
              onChange={handleFileChange}
              disabled={disabled}
              className="hidden"
            />
          </label>
        </div>
      </div>

      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className="relative"
      >
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="Paste clinical safety table content here...

Supported formats:
• Pipe-delimited tables (Header1|Header2|...)
• CSV format
• Plain text with consistent delimiters"
          rows={12}
          className={`
            input-clinical w-full font-mono text-sm resize-none
            ${dragActive ? "border-saline" : ""}
          `}
        />

        {/* Drag overlay */}
        {dragActive && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-saline/5 border-2 border-dashed border-saline rounded-lg flex items-center justify-center"
          >
            <div className="text-saline font-heading">Drop file here</div>
          </motion.div>
        )}
      </div>

      {/* Character count */}
      <div className="flex justify-between text-xs text-cortex">
        <span>
          {value.length === 0
            ? "No content"
            : `${value.length.toLocaleString()} characters`}
        </span>
        {value.length > 0 && (
          <button
            onClick={() => onChange("")}
            disabled={disabled}
            className="text-plasma/70 hover:text-plasma transition-colors disabled:opacity-50"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
}

/* ============================================
   TABLE PREVIEW
   ============================================
   Shows parsed table structure before processing
*/

interface TablePreviewProps {
  content: string;
}

export function TablePreview({ content }: TablePreviewProps) {
  const rows = parseTableRows(content);
  
  if (rows.length === 0) {
    return null;
  }

  const headers = rows[0];
  const dataRows = rows.slice(1, 6); // Show first 5 data rows
  const hiddenCount = rows.length - 6;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="clinical-card border border-parchment/10"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs text-cortex uppercase tracking-wider">
          Table Preview
        </h3>
        <span className="text-xs text-parchment/50">
          {rows.length - 1} rows × {headers.length} columns
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-parchment/10">
              {headers.map((header, i) => (
                <th
                  key={i}
                  className="text-left px-2 py-1.5 text-parchment font-heading text-xs"
                >
                  {truncate(header, 20)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataRows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="border-b border-parchment/5 last:border-0"
              >
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="px-2 py-1.5 text-cortex text-xs font-mono"
                  >
                    {truncate(cell, 30)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {hiddenCount > 0 && (
        <div className="text-xs text-cortex mt-2 text-center">
          + {hiddenCount} more rows
        </div>
      )}
    </motion.div>
  );
}

/* ============================================
   HELPERS
   ============================================ */

function readFile(file: File, callback: (content: string) => void) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const content = e.target?.result as string;
    callback(content);
  };
  reader.readAsText(file);
}

function parseTableRows(content: string): string[][] {
  if (!content.trim()) return [];
  
  const lines = content.trim().split("\n").filter(Boolean);
  
  // Detect delimiter
  const firstLine = lines[0];
  let delimiter = "|";
  if (firstLine.includes("\t")) delimiter = "\t";
  else if (firstLine.includes(",") && !firstLine.includes("|")) delimiter = ",";
  
  return lines.map((line) =>
    line.split(delimiter).map((cell) => cell.trim())
  );
}

function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - 1) + "…";
}

const exampleTable = `Adverse Event|Total Patients|Drug Group|Placebo Group|Risk Ratio
Headache|145|89 (12.3%)|56 (7.8%)|1.58
Nausea|98|67 (9.3%)|31 (4.3%)|2.16
Fatigue|76|45 (6.2%)|31 (4.3%)|1.45
Dizziness|52|34 (4.7%)|18 (2.5%)|1.89
Insomnia|41|28 (3.9%)|13 (1.8%)|2.15
Dry Mouth|38|26 (3.6%)|12 (1.7%)|2.17
Constipation|29|19 (2.6%)|10 (1.4%)|1.90`;
