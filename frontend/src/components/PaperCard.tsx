"use client";

import { FileText, Calendar, Layers, CheckCircle2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Paper } from "@/types/api";

interface PaperCardProps {
  paper: Paper;
  selected?: boolean;
  onSelect?: (paper: Paper) => void;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function PaperCard({ paper, selected = false, onSelect }: PaperCardProps) {
  const isClickable = Boolean(onSelect);

  return (
    <Card
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
      aria-pressed={isClickable ? selected : undefined}
      onClick={() => onSelect?.(paper)}
      onKeyDown={(e) => {
        if (isClickable && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          onSelect?.(paper);
        }
      }}
      className={cn(
        "transition-all duration-150",
        isClickable && "cursor-pointer hover:shadow-md hover:border-primary/40",
        selected && "border-primary ring-1 ring-primary/30 bg-primary/[0.03]",
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div
            className={cn(
              "mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md border",
              selected
                ? "border-primary/30 bg-primary/10 text-primary"
                : "border-border bg-muted text-muted-foreground",
            )}
          >
            <FileText className="h-4 w-4" />
          </div>

          {/* Content */}
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-2">
              <h3
                className={cn(
                  "text-sm font-semibold leading-snug",
                  selected ? "text-primary" : "text-foreground",
                )}
                title={paper.title}
              >
                {paper.title}
              </h3>
              {selected && (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
              )}
            </div>

            <p
              className="mt-0.5 truncate text-xs text-muted-foreground"
              title={paper.filename}
            >
              {paper.filename}
            </p>

            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Badge variant="muted" className="gap-1 text-xs font-normal">
                <Calendar className="h-3 w-3" />
                {formatDate(paper.created_at)}
              </Badge>

              {paper.page_count != null && (
                <Badge variant="muted" className="gap-1 text-xs font-normal">
                  <Layers className="h-3 w-3" />
                  {paper.page_count} {paper.page_count === 1 ? "page" : "pages"}
                </Badge>
              )}

              <Badge variant="success" className="text-xs font-normal">
                Indexed
              </Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
