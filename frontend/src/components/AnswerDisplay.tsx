"use client";

import ReactMarkdown from "react-markdown";
import { CitationList } from "@/components/CitationList";
import { EvidenceCard } from "@/components/EvidenceCard";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { MessageSquare, Layers } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ResearchQueryResponse } from "@/types/api";

interface AnswerDisplayProps {
  result: ResearchQueryResponse;
}

// ─── Evidence marker chip ─────────────────────────────────────────────────────

interface EvidenceChipProps {
  evidenceId: string;
}

function EvidenceChip({ evidenceId }: EvidenceChipProps) {
  function scrollTo() {
    const el = document.getElementById(`evidence-${evidenceId}`);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  return (
    <button
      onClick={scrollTo}
      className="mx-0.5 inline-flex h-5 min-w-[2rem] items-center justify-center rounded-full bg-primary/15 px-1.5 text-xs font-semibold text-primary hover:bg-primary/25 transition-colors align-baseline"
      title={`Jump to ${evidenceId}`}
    >
      {evidenceId}
    </button>
  );
}

// ─── Inline [En] marker renderer ─────────────────────────────────────────────
//
// react-markdown v9 does not support a "text" component override.
// We pre-process the raw answer text by splitting on [E1], [E2], … markers
// and rendering each segment individually — plain text segments go through
// ReactMarkdown, marker segments become EvidenceChip buttons.
//
// This keeps full markdown formatting (bold, italics, lists, headings) intact
// while making evidence markers interactive.

function renderSegments(text: string): React.ReactNode[] {
  // Split into alternating [plain text, [En] marker, plain text, …]
  const parts = text.split(/(\[E\d+\])/g);

  return parts.map((part, i) => {
    const match = part.match(/^\[E(\d+)\]$/);
    if (match) {
      return <EvidenceChip key={i} evidenceId={`E${match[1]}`} />;
    }
    if (!part) return null;

    // Render each plain-text segment through ReactMarkdown so markdown
    // formatting (bold, lists, etc.) is preserved.
    return (
      <ReactMarkdown
        key={i}
        components={{
          // Remove the wrapping <p> that ReactMarkdown adds for inline text
          // so chips stay inline with the surrounding text.
          p: ({ children }) => (
            <span className="leading-relaxed">{children}</span>
          ),
        }}
      >
        {part}
      </ReactMarkdown>
    );
  });
}

// ─── AnswerText ───────────────────────────────────────────────────────────────

function AnswerText({ text }: { text: string }) {
  // If there are no [En] markers, render directly through ReactMarkdown
  // for cleaner block-level formatting (paragraphs, lists, headings).
  const hasMarkers = /\[E\d+\]/.test(text);

  if (!hasMarkers) {
    return (
      <div className="prose prose-sm max-w-none text-foreground leading-relaxed">
        <ReactMarkdown
          components={{
            p: ({ children }) => (
              <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>
            ),
          }}
        >
          {text}
        </ReactMarkdown>
      </div>
    );
  }

  // Has markers — render inline with chips interspersed.
  return (
    <div className="text-sm text-foreground leading-relaxed space-y-1">
      {renderSegments(text)}
    </div>
  );
}

// ─── AnswerDisplay ────────────────────────────────────────────────────────────

export function AnswerDisplay({ result }: AnswerDisplayProps) {
  function scrollToEvidence(evidenceId: string) {
    const el = document.getElementById(`evidence-${evidenceId}`);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <div className="space-y-6">
      {/* Question echo */}
      <div className="flex items-start gap-2 rounded-md border border-border bg-muted/40 px-4 py-3">
        <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <p className="text-sm text-foreground">{result.question}</p>
      </div>

      {/* Answer body */}
      <div className="rounded-lg border border-border bg-card p-5">
        <div className="mb-3 flex items-center gap-2">
          <h2 className="text-base font-semibold text-foreground">Answer</h2>
          <Badge variant="success" className="text-xs">
            Supported by evidence
          </Badge>
        </div>

        {result.answer && <AnswerText text={result.answer} />}

        {/* Citations */}
        {result.citations.length > 0 && (
          <>
            <Separator className="my-4" />
            <CitationList
              citations={result.citations}
              onCitationClick={scrollToEvidence}
            />
          </>
        )}
      </div>

      {/* Evidence section */}
      {result.evidence.length > 0 && (
        <div>
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-foreground">
            <Layers className="h-4 w-4 text-primary" />
            Retrieved Evidence
            <span className="ml-1 text-xs font-normal text-muted-foreground">
              ({result.evidence.length} passage
              {result.evidence.length !== 1 ? "s" : ""})
            </span>
          </div>

          <div className={cn("space-y-3")}>
            {result.evidence.map((item) => (
              <EvidenceCard key={item.chunk_id} item={item} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
