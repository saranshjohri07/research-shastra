"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Upload,
  FlaskConical,
  Library,
  ArrowRight,
  Search,
  FileText,
  Quote,
  ShieldCheck,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { PaperCard } from "@/components/PaperCard";
import { Spinner } from "@/components/ui/spinner";
import { Separator } from "@/components/ui/separator";
import { listPapers, checkHealth } from "@/lib/api";
import type { Paper } from "@/types/api";

// ─── How it works steps ──────────────────────────────────────

const HOW_IT_WORKS = [
  {
    icon: Upload,
    label: "Upload a paper",
    description:
      "Add any research paper as a PDF. It is extracted, chunked, and indexed automatically.",
  },
  {
    icon: Search,
    label: "Ask a question",
    description:
      "Type a question in plain language. The system searches the paper's content for relevant passages.",
  },
  {
    icon: Quote,
    label: "Read the answer",
    description:
      "Get a grounded answer backed by the actual text — with source passages and page references.",
  },
  {
    icon: ShieldCheck,
    label: "Trust the evidence",
    description:
      "When the paper doesn't contain enough information, ResearchShastra says so — no invented answers.",
  },
];

// ─── Component ───────────────────────────────────────────────

export default function HomePage() {
  const [recentPapers, setRecentPapers] = useState<Paper[]>([]);
  const [papersLoading, setPapersLoading] = useState(true);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));

    listPapers()
      .then((papers) => setRecentPapers(papers.slice(0, 3)))
      .catch(() => setRecentPapers([]))
      .finally(() => setPapersLoading(false));
  }, []);

  return (
    <div className="space-y-14">
      {/* ── Hero ──────────────────────────────────────────── */}
      <section className="pt-4 pb-2">
        <div className="flex items-center gap-2 mb-4">
          {backendOnline === true && (
            <span className="flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700">
              <CheckCircle2 className="h-3 w-3" />
              Backend connected
            </span>
          )}
          {backendOnline === false && (
            <span className="flex items-center gap-1.5 rounded-full border border-red-200 bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700">
              Backend offline — start the FastAPI server
            </span>
          )}
        </div>

        <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          Research with evidence,
          <br />
          <span className="text-primary">not just answers.</span>
        </h1>

        <p className="mt-4 max-w-xl text-base text-muted-foreground leading-relaxed">
          ResearchShastra helps you understand research papers by answering your
          questions from the paper&apos;s own content — with source passages,
          page references, and honest abstention when the evidence isn&apos;t
          there.
        </p>

        <div className="mt-6 flex flex-wrap gap-3">
          <Button asChild size="lg">
            <Link href="/library">
              <Upload className="mr-2 h-4 w-4" />
              Upload a paper
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link href="/research">
              <FlaskConical className="mr-2 h-4 w-4" />
              Ask a question
            </Link>
          </Button>
        </div>
      </section>

      <Separator />

      {/* ── How it works ──────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-semibold text-foreground mb-5">
          How it works
        </h2>
        <div className="grid gap-5 sm:grid-cols-2">
          {HOW_IT_WORKS.map(({ icon: Icon, label, description }, i) => (
            <div
              key={label}
              className="flex items-start gap-4 rounded-lg border border-border bg-card p-4"
            >
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                <Icon className="h-4 w-4" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-muted-foreground">
                    Step {i + 1}
                  </span>
                </div>
                <p className="mt-0.5 text-sm font-semibold text-foreground">
                  {label}
                </p>
                <p className="mt-1 text-sm text-muted-foreground leading-relaxed">
                  {description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <Separator />

      {/* ── Recent papers ─────────────────────────────────── */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-foreground">
            Recent papers
          </h2>
          <Button asChild variant="ghost" size="sm">
            <Link href="/library">
              <Library className="mr-1.5 h-3.5 w-3.5" />
              View all
              <ArrowRight className="ml-1 h-3.5 w-3.5" />
            </Link>
          </Button>
        </div>

        {papersLoading && (
          <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
            <Spinner size="sm" />
            Loading…
          </div>
        )}

        {!papersLoading && recentPapers.length === 0 && (
          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-10 text-center">
            <FileText className="h-8 w-8 text-muted-foreground" />
            <p className="mt-2 text-sm text-muted-foreground">
              No papers uploaded yet.
            </p>
            <Button asChild size="sm" className="mt-3">
              <Link href="/library">
                <Upload className="mr-1.5 h-3.5 w-3.5" />
                Upload your first paper
              </Link>
            </Button>
          </div>
        )}

        {!papersLoading && recentPapers.length > 0 && (
          <div className="space-y-3">
            {recentPapers.map((paper) => (
              <Link
                key={paper.id}
                href={`/research?paper_id=${paper.id}&title=${encodeURIComponent(paper.title)}`}
                className="block"
              >
                <PaperCard paper={paper} />
              </Link>
            ))}
            {recentPapers.length === 3 && (
              <div className="pt-1 text-center">
                <Button asChild variant="ghost" size="sm">
                  <Link href="/library">
                    See all papers
                    <ArrowRight className="ml-1 h-3.5 w-3.5" />
                  </Link>
                </Button>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
