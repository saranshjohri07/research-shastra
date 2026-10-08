"use client";

import { BookMarked, BookOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { CitationItem } from "@/types/api";

interface CitationListProps {
  citations: CitationItem[];
  /** Callback when a citation badge is clicked — scrolls to evidence card */
  onCitationClick?: (evidenceId: string) => void;
}

function pageLabel(start: number | null, end: number | null): string {
  if (start == null) return "";
  if (end == null || start === end) return `p. ${start}`;
  return `pp. ${start}–${end}`;
}

export function CitationList({ citations, onCitationClick }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div>
      <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
        <BookMarked className="h-4 w-4 text-primary" />
        Citations
        <span className="ml-1 text-xs font-normal text-muted-foreground">
          ({citations.length})
        </span>
      </div>

      <Separator className="my-2" />

      <ol className="space-y-2">
        {citations.map((citation, index) => (
          <li key={citation.chunk_id} className="flex items-start gap-3 text-sm">
            {/* Number */}
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-semibold text-muted-foreground">
              {index + 1}
            </span>

            {/* Content */}
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-1.5">
                {/* Evidence ID — clickable to jump to evidence card */}
                <button
                  onClick={() => onCitationClick?.(citation.evidence_id)}
                  className="inline-flex h-5 min-w-[2rem] items-center justify-center rounded-full bg-primary/10 px-2 text-xs font-semibold text-primary hover:bg-primary/20 transition-colors"
                  title="Jump to evidence"
                >
                  {citation.evidence_id}
                </button>

                <span className="font-medium text-foreground">{citation.title}</span>

                {citation.section && (
                  <Badge variant="secondary" className="text-xs font-normal">
                    {citation.section}
                  </Badge>
                )}

                {citation.page_start != null && (
                  <Badge variant="outline" className="gap-1 text-xs font-normal">
                    <BookOpen className="h-3 w-3" />
                    {pageLabel(citation.page_start, citation.page_end)}
                  </Badge>
                )}
              </div>

              <p className="mt-0.5 text-xs text-muted-foreground">
                {citation.filename}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
