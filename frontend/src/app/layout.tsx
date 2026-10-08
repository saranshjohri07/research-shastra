import type { Metadata } from "next";
import "./globals.css";
import { SiteNav } from "@/components/SiteNav";

export const metadata: Metadata = {
  title: "ResearchShastra",
  description: "Research with evidence, not just answers.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <SiteNav />
        <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
          {children}
        </main>
      </body>
    </html>
  );
}
