"use client";

import { ShieldAlert, Layers, MessageSquare } from "lucide-react";
import { EvidenceCard } from "@/components/EvidenceCard";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import type { ResearchQueryResponse } from "@/types/api";

interface AbstentionDisplayProps {
  result: ResearchQueryResponse;
}

/** Human-readable labels for machine abstention reason codes */
function formatAbstentionReason(reason: string | null): string {
  if (!reason) return "The available evidence was not sufficient to answer this question.";
  if (reason === "no_evidence")
    return "No relevant passages were found in this paper for the given question.";
  if (reason === "low_similarity")
    return "The retrieved passages were not similar enough to the question to produce a reliable answer.";
  // Otherwise it's a natural-language reason from the LLM — use it directly
  return reason;
}

export function AbstentionDisplay({ result }: AbstentionDisplayProps) {
  const reasonText = formatAbstentionReason(result.abstention_reason);

  return (
    <div className="space-y-6">
      {/* Question echo */}
      <div className="flex items-start gap-2 rounded-md border border-border bg-muted/40 px-4 py-3">
        <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <p className="text-sm text-foreground">{result.question}</p>
      </div>

      {/* Insufficient evidence notice */}
      <Alert variant="warning">
        <ShieldAlert className="h-4 w-4" />
        <AlertTitle>Insufficient evidence</AlertTitle>
        <AlertDescription>{reasonText}</AlertDescription>
      </Alert>

      {/* If there IS retrieved evidence, show it — but make clear it didn't support an answer */}
      {result.evidence.length > 0 && (
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-muted-foreground">
            <Layers className="h-4 w-4" />
            Retrieved passages
            <span className="ml-1 text-xs font-normal">
              (did not support an answer)
            </span>
          </div>

          <div className="space-y-3">
            {result.evidence.map((item) => (
              <EvidenceCard key={item.chunk_id} item={item} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
