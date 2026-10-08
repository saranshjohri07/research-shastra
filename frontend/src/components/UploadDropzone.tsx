"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, FileText, X, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { uploadPaper } from "@/lib/api";
import type { UploadPaperResponse } from "@/types/api";

interface UploadDropzoneProps {
  onSuccess: (paper: UploadPaperResponse) => void;
}

type UploadState =
  | { status: "idle" }
  | { status: "selected"; file: File }
  | { status: "uploading"; file: File }
  | { status: "success"; paper: UploadPaperResponse }
  | { status: "error"; message: string };

export function UploadDropzone({ onSuccess }: UploadDropzoneProps) {
  const [state, setState] = useState<UploadState>({ status: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function acceptFile(file: File) {
    if (file.type !== "application/pdf") {
      setState({ status: "error", message: "Only PDF files are supported." });
      return;
    }
    setState({ status: "selected", file });
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) acceptFile(file);
  }, []);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => setDragging(false);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) acceptFile(file);
  };

  async function handleUpload() {
    if (state.status !== "selected") return;
    const { file } = state;

    setState({ status: "uploading", file });

    try {
      const paper = await uploadPaper(file);
      setState({ status: "success", paper });
      onSuccess(paper);
    } catch (err) {
      setState({
        status: "error",
        message: err instanceof Error ? err.message : "Upload failed. Please try again.",
      });
    }
  }

  function reset() {
    setState({ status: "idle" });
    if (inputRef.current) inputRef.current.value = "";
  }

  // ── Success state ──────────────────────────────────────────
  if (state.status === "success") {
    return (
      <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-emerald-900">Paper uploaded successfully</p>
            <p className="mt-0.5 text-sm text-emerald-700 truncate">
              {state.paper.title}
            </p>
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-emerald-700">
              <span>{state.paper.page_count} pages</span>
              <span>{state.paper.chunk_count} chunks indexed</span>
            </div>
          </div>
          <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={reset}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  // ── Error state ────────────────────────────────────────────
  if (state.status === "error") {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-5">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
          <div className="flex-1">
            <p className="font-semibold text-red-900">Upload failed</p>
            <p className="mt-0.5 text-sm text-red-700">{state.message}</p>
          </div>
          <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={reset}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  // ── Uploading state ────────────────────────────────────────
  if (state.status === "uploading") {
    return (
      <div className="rounded-lg border border-border bg-muted/30 p-8 text-center">
        <Spinner size="lg" className="mx-auto text-primary" />
        <p className="mt-3 text-sm font-medium text-foreground">
          Uploading and indexing…
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {state.file.name}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          This may take a moment while the paper is processed.
        </p>
      </div>
    );
  }

  // ── Selected state ─────────────────────────────────────────
  if (state.status === "selected") {
    return (
      <div className="rounded-lg border border-primary/40 bg-primary/[0.03] p-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md border border-primary/20 bg-primary/10">
            <FileText className="h-5 w-5 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-foreground truncate">
              {state.file.name}
            </p>
            <p className="text-xs text-muted-foreground">
              {(state.file.size / 1024 / 1024).toFixed(2)} MB · PDF
            </p>
          </div>
          <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={reset}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="mt-4 flex gap-2">
          <Button onClick={handleUpload} className="flex-1">
            <Upload className="mr-2 h-4 w-4" />
            Upload &amp; Index
          </Button>
          <Button variant="outline" onClick={reset}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  // ── Idle / drop zone ───────────────────────────────────────
  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={cn(
        "rounded-lg border-2 border-dashed p-10 text-center transition-colors",
        dragging
          ? "border-primary bg-primary/[0.04]"
          : "border-border bg-background hover:border-primary/40 hover:bg-muted/20",
      )}
    >
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted">
        <Upload className="h-6 w-6 text-muted-foreground" />
      </div>
      <p className="mt-3 text-sm font-medium text-foreground">
        Drag a PDF here, or{" "}
        <button
          className="text-primary hover:underline focus:outline-none"
          onClick={() => inputRef.current?.click()}
        >
          browse to select
        </button>
      </p>
      <p className="mt-1 text-xs text-muted-foreground">PDF files only</p>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={handleFileInput}
      />
    </div>
  );
}
