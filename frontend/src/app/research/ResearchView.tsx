"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  FlaskConical,
  FileText,
  ArrowLeft,
  Send,
  ChevronDown,
  BookOpen,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Spinner } from "@/components/ui/spinner";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { PaperCard } from "@/components/PaperCard";
import { AnswerDisplay } from "@/components/AnswerDisplay";
import { AbstentionDisplay } from "@/components/AbstentionDisplay";
import { listPapers, queryResearch } from "@/lib/api";
import type { Paper, ResearchQueryResponse } from "@/types/api";

type QueryState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; result: ResearchQueryResponse }
  | { status: "error"; message: string };

export function ResearchView() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [papers, setPapers] = useState<Paper[]>([]);
  const [papersLoading, setPapersLoading] = useState(true);
  const [papersError, setPapersError] = useState<string | null>(null);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [showPaperPicker, setShowPaperPicker] = useState(false);

  const [question, setQuestion] = useState("");
  const [queryState, setQueryState] = useState<QueryState>({ status: "idle" });
  const resultRef = useRef<HTMLDivElement>(null);

  const fetchPapers = useCallback(async () => {
    setPapersLoading(true);
    setPapersError(null);
    try {
      const data = await listPapers();
      setPapers(data);
      const urlPaperId = searchParams.get("paper_id");
      if (urlPaperId) {
        const match = data.find((p) => p.id === urlPaperId);
        if (match) setSelectedPaper(match);
      }
    } catch (err) {
      setPapersError(
        err instanceof Error ? err.message : "Could not load papers.",
      );
    } finally {
      setPapersLoading(false);
    }
  }, [searchParams]);

  useEffect(() => {
    fetchPapers();
  }, [fetchPapers]);

  function selectPaper(paper: Paper) {
    setSelectedPaper(paper);
    setShowPaperPicker(false);
    setQueryState({ status: "idle" });
    router.replace(
      `/research?paper_id=${paper.id}&title=${encodeURIComponent(paper.title)}`,
      { scroll: false },
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedPaper || !question.trim()) return;
    setQueryState({ status: "loading" });
    try {
      const result = await queryResearch({
        paper_id: selectedPaper.id,
        question: question.trim(),
        top_k: 5,
      });
      setQueryState({ status: "success", result });
      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      setQueryState({
        status: "error",
        message:
          err instanceof Error
            ? err.message
            : "Something went wrong. Please try again.",
      });
    }
  }

  const isSubmittable =
    selectedPaper !== null &&
    question.trim().length > 0 &&
    queryState.status !== "loading";

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <div className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-semibold">Research</h1>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Ask a question. The answer will be grounded in the paper&apos;s actual
          content.
        </p>
      </div>

      {/* ── Step 1: Paper selection ── */}
      <section className="rounded-lg border border-border bg-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs">
              1
            </span>
            Select a paper
          </div>

          {selectedPaper && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={() => setShowPaperPicker((v) => !v)}
            >
              Change
              <ChevronDown className="ml-1 h-3.5 w-3.5" />
            </Button>
          )}
        </div>

        {selectedPaper && !showPaperPicker && (
          <PaperCard paper={selectedPaper} selected />
        )}

        {(!selectedPaper || showPaperPicker) && (
          <div className="space-y-2">
            {papersLoading && (
              <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
                <Spinner size="sm" />
                Loading papers…
              </div>
            )}

            {!papersLoading && papersError && (
              <Alert variant="destructive">
                <AlertTitle>Could not load papers</AlertTitle>
                <AlertDescription>{papersError}</AlertDescription>
              </Alert>
            )}

            {!papersLoading && !papersError && papers.length === 0 && (
              <div className="rounded-md border border-dashed border-border py-8 text-center">
                <BookOpen className="mx-auto h-8 w-8 text-muted-foreground" />
                <p className="mt-2 text-sm text-muted-foreground">
                  No papers in library yet.
                </p>
                <Link href="/library">
                  <Button size="sm" className="mt-3" variant="outline">
                    <ArrowLeft className="mr-1.5 h-3.5 w-3.5" />
                    Go to Library to upload
                  </Button>
                </Link>
              </div>
            )}

            {!papersLoading && !papersError && papers.length > 0 && (
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1 scrollbar-thin">
                {papers.map((paper) => (
                  <PaperCard
                    key={paper.id}
                    paper={paper}
                    selected={selectedPaper?.id === paper.id}
                    onSelect={selectPaper}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── Step 2: Question ── */}
      <section
        className={`rounded-lg border bg-card p-5 space-y-3 transition-opacity ${
          selectedPaper ? "opacity-100" : "pointer-events-none opacity-40"
        }`}
      >
        <div className="flex items-center gap-2 text-sm font-semibold">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs">
            2
          </span>
          Ask your question
        </div>

        {selectedPaper && (
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <FileText className="h-3.5 w-3.5 shrink-0" />
            <span className="truncate">
              Asking about:{" "}
              <strong>{selectedPaper.title}</strong>
            </span>
          </p>
        )}

        <form onSubmit={handleSubmit} className="space-y-3">
          <Textarea
            placeholder="e.g. What is the main contribution of this paper?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={4}
            maxLength={2000}
            disabled={!selectedPaper || queryState.status === "loading"}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                (e.ctrlKey || e.metaKey) &&
                isSubmittable
              ) {
                handleSubmit(e as unknown as React.FormEvent);
              }
            }}
          />

          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              {question.length}/2000
              {question.length > 0 && (
                <span className="ml-2 text-muted-foreground/60">
                  Ctrl+Enter to submit
                </span>
              )}
            </span>
            <Button type="submit" disabled={!isSubmittable} size="sm">
              {queryState.status === "loading" ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  Searching…
                </>
              ) : (
                <>
                  <Send className="mr-1.5 h-3.5 w-3.5" />
                  Ask
                </>
              )}
            </Button>
          </div>
        </form>
      </section>

      {/* ── Step 3: Result ── */}
      {queryState.status !== "idle" && (
        <>
          <Separator />
          <div ref={resultRef} className="scroll-mt-20">
            <div className="mb-3 flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs">
                3
              </span>
              <span className="text-sm font-semibold">Result</span>

              {queryState.status === "success" && (
                <Badge
                  variant={
                    queryState.result.answerable ? "success" : "warning"
                  }
                  className="ml-1 text-xs"
                >
                  {queryState.result.answerable
                    ? "Answered"
                    : "Insufficient evidence"}
                </Badge>
              )}
            </div>

            {queryState.status === "loading" && (
              <div className="flex items-center justify-center py-16 text-muted-foreground">
                <Spinner className="mr-2 text-primary" />
                <span className="text-sm">
                  Retrieving evidence and generating answer…
                </span>
              </div>
            )}

            {queryState.status === "error" && (
              <Alert variant="destructive">
                <AlertTitle>Query failed</AlertTitle>
                <AlertDescription>{queryState.message}</AlertDescription>
              </Alert>
            )}

            {queryState.status === "success" &&
              queryState.result.answerable && (
                <AnswerDisplay result={queryState.result} />
              )}

            {queryState.status === "success" &&
              !queryState.result.answerable && (
                <AbstentionDisplay result={queryState.result} />
              )}
          </div>
        </>
      )}
    </div>
  );
}
