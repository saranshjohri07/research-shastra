"use client";

import { BookOpen, Hash, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { EvidenceItem } from "@/types/api";

interface EvidenceCardProps {
  item: EvidenceItem;
  /** Highlight this card — used when an [En] marker is hovered in the answer */
  highlighted?: boolean;
}

function pageLabel(start: number | null, end: number | null): string {
  if (start == null) return "Unknown page";
  if (end == null || start === end) return `p. ${start}`;
  return `pp. ${start}–${end}`;
}

function similarityColor(score: number): string {
  if (score >= 0.7) return "success";
  if (score >= 0.45) return "warning";
  return "muted";
}

export function EvidenceCard({ item, highlighted = false }: EvidenceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const PREVIEW_LENGTH = 280;
  const isLong = item.text.length > PREVIEW_LENGTH;
  const displayText =
    expanded || !isLong ? item.text : item.text.slice(0, PREVIEW_LENGTH) + "…";

  return (
    <div
      id={`evidence-${item.evidence_id}`}
      className={cn(
        "rounded-lg border p-4 transition-all",
        highlighted
          ? "border-primary/50 bg-primary/[0.04] shadow-sm"
          : "border-border bg-card",
      )}
    >
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Evidence ID chip */}
        <span className="inline-flex h-6 min-w-[2.25rem] items-center justify-center rounded-full bg-primary px-2 text-xs font-bold text-primary-foreground">
          {item.evidence_id}
        </span>

        {/* Similarity score */}
        <Badge
          variant={similarityColor(item.similarity) as "success" | "warning" | "muted"}
          className="text-xs"
        >
          {(item.similarity * 100).toFixed(0)}% match
        </Badge>

        {/* Page range */}
        <Badge variant="outline" className="gap-1 text-xs font-normal">
          <BookOpen className="h-3 w-3" />
          {pageLabel(item.page_start, item.page_end)}
        </Badge>

        {/* Section */}
        {item.section && (
          <Badge variant="secondary" className="max-w-[14rem] truncate text-xs font-normal">
            {item.section}
          </Badge>
        )}

        {/* Chunk index */}
        <span className="ml-auto text-xs text-muted-foreground">
          <Hash className="mr-0.5 inline h-3 w-3" />
          chunk {item.chunk_index}
        </span>
      </div>

      {/* Paper name */}
      <p className="mt-2 text-xs font-medium text-muted-foreground">
        {item.title} — <span className="font-normal">{item.filename}</span>
      </p>

      {/* Passage text */}
      <div className="mt-3">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
          {displayText}
        </p>

        {isLong && (
          <Button
            variant="ghost"
            size="sm"
            className="mt-1 h-7 px-2 text-xs text-muted-foreground"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? (
              <>
                <ChevronUp className="mr-1 h-3 w-3" /> Show less
              </>
            ) : (
              <>
                <ChevronDown className="mr-1 h-3 w-3" /> Show full passage
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  );
}
