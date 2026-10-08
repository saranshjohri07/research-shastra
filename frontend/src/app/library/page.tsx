"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Library,
  RefreshCw,
  Upload,
  ChevronRight,
  BookOpen,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { PaperCard } from "@/components/PaperCard";
import { UploadDropzone } from "@/components/UploadDropzone";
import { listPapers } from "@/lib/api";
import type { Paper, UploadPaperResponse } from "@/types/api";

export default function LibraryPage() {
  const router = useRouter();

  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);

  const fetchPapers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listPapers();
      setPapers(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not load papers. Is the backend running?",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPapers();
  }, [fetchPapers]);

  function handleUploadSuccess(uploaded: UploadPaperResponse) {
    setShowUpload(false);
    setSelectedPaperId(uploaded.id);
    fetchPapers();
  }

  function handleSelectAndResearch(paper: Paper) {
    router.push(`/research?paper_id=${paper.id}&title=${encodeURIComponent(paper.title)}`);
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Library className="h-5 w-5 text-primary" />
            <h1 className="text-xl font-semibold">Paper Library</h1>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload and manage the research papers you want to query.
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchPapers}
            disabled={loading}
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
          <Button
            size="sm"
            onClick={() => setShowUpload((v) => !v)}
            variant={showUpload ? "secondary" : "default"}
          >
            <Upload className="mr-1.5 h-4 w-4" />
            {showUpload ? "Cancel" : "Upload Paper"}
          </Button>
        </div>
      </div>

      {/* Upload panel */}
      {showUpload && (
        <div>
          <UploadDropzone onSuccess={handleUploadSuccess} />
        </div>
      )}

      <Separator />

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-16 text-muted-foreground">
          <Spinner className="mr-2" />
          <span className="text-sm">Loading papers…</span>
        </div>
      )}

      {/* Error state */}
      {!loading && error && (
        <Alert variant="destructive">
          <AlertTitle>Could not load library</AlertTitle>
          <AlertDescription>
            {error}
            <Button
              variant="link"
              size="sm"
              className="ml-2 h-auto p-0 text-red-800 underline"
              onClick={fetchPapers}
            >
              Try again
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Empty state */}
      {!loading && !error && papers.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-16 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-muted">
            <BookOpen className="h-7 w-7 text-muted-foreground" />
          </div>
          <h2 className="mt-4 text-base font-semibold">No papers yet</h2>
          <p className="mt-1 max-w-xs text-sm text-muted-foreground">
            Upload a PDF to get started. Each paper is indexed so you can ask
            questions about it.
          </p>
          <Button
            className="mt-5"
            onClick={() => setShowUpload(true)}
            size="sm"
          >
            <Upload className="mr-1.5 h-4 w-4" />
            Upload your first paper
          </Button>
        </div>
      )}

      {/* Paper grid */}
      {!loading && !error && papers.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            {papers.length} paper{papers.length !== 1 ? "s" : ""} — select one
            to start researching
          </p>

          {papers.map((paper) => (
            <div key={paper.id} className="group relative">
              <PaperCard
                paper={paper}
                selected={selectedPaperId === paper.id}
                onSelect={(p) =>
                  setSelectedPaperId((prev) =>
                    prev === p.id ? null : p.id,
                  )
                }
              />

              {/* Research action that appears when selected */}
              {selectedPaperId === paper.id && (
                <div className="mt-2 flex justify-end">
                  <Button
                    size="sm"
                    onClick={() => handleSelectAndResearch(paper)}
                  >
                    Ask a question about this paper
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
