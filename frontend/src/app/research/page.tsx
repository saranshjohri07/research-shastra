import { Suspense } from "react";
import { Spinner } from "@/components/ui/spinner";
import { ResearchView } from "./ResearchView";

// Next.js 14 app router: useSearchParams() requires a Suspense boundary.
// The actual page logic lives in ResearchView so the boundary wraps it cleanly.
export default function ResearchPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-24 text-muted-foreground">
          <Spinner className="mr-2 text-primary" />
          <span className="text-sm">Loading…</span>
        </div>
      }
    >
      <ResearchView />
    </Suspense>
  );
}
