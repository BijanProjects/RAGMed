"use client";

import type { Source } from "@/lib/types";

// MedQuAD's answers come mostly from NIH consumer health pages. This copy of the corpus does not
// keep each page's address, so the outside link searches MedlinePlus for the same topic.
const MEDLINEPLUS_SEARCH =
  "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?v%3Aproject=medlineplus&v%3Asources=medlineplus-bundle&query=";

/** The "**Sources** (MedQuAD):" list the backend writes into the text; shown as cards instead. */
const SOURCES_BLOCK = /\n*\*\*Sources\*\* \(MedQuAD\):\n(?:\d+\. .*(?:\n|$))+/;

export function withoutSourcesBlock(text: string): string {
  return text.replace(SOURCES_BLOCK, "\n\n").trim();
}

/** Turn "[1]" and "[1, 2]" into links to the matching source cards (`#<prefix>-<n>`). */
export function linkCitations(text: string, prefix: string, sources: Source[]): string {
  const known = new Set(sources.map((s) => s.number));
  return text.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (whole, list: string) => {
    const numbers = list.split(",").map((n) => Number(n.trim()));
    if (!numbers.every((n) => known.has(n))) return whole;
    return numbers.map((n) => `[\\[${n}\\]](#${prefix}-${n})`).join("");
  });
}

/** A short subject for the outside search: the passage's question without its question words. */
function topic(question: string): string {
  return question
    .replace(/…$/, "")
    // Longest openings first: "what are the symptoms of" before "what are".
    .replace(/^(what are the (symptoms|treatments|complications|signs) (of|for)|what (is|are)( \(are\))?|what causes|how (to|can|is|are|many|much)|who is at risk for|do you have information about)\s+/i, "")
    .replace(/\?.*$/, "")
    .trim()
    .slice(0, 120);
}

/** "What causes Asthma? The exact cause…" → "What causes Asthma?": the passage is shown below. */
function title(question: string): string {
  const mark = question.indexOf("? ");
  return mark > 0 && question.endsWith("…") ? question.slice(0, mark + 1) : question;
}

/** Numbered sources; each opens to the passage that was read, with a link to read more. */
export function Sources({
  sources,
  prefix,
  open,
  onToggle,
}: {
  sources: Source[];
  prefix: string;
  open: Set<number>;
  onToggle: (number: number, isOpen: boolean) => void;
}) {
  if (sources.length === 0) return null;
  return (
    <section className="space-y-1.5" aria-label="Sources">
      <p className="text-sm font-medium">Sources</p>
      {sources.map((source) => (
        <details
          key={source.number}
          id={`${prefix}-${source.number}`}
          open={open.has(source.number)}
          onToggle={(event) => onToggle(source.number, event.currentTarget.open)}
          className="scroll-mt-24 rounded-lg border border-black/10 text-sm open:bg-white/60 dark:border-white/10 dark:open:bg-white/[0.03]"
        >
          <summary className="flex cursor-pointer gap-2 px-3 py-2 hover:bg-black/[0.03] dark:hover:bg-white/5">
            <span className="font-mono text-xs leading-5 text-sky-700 dark:text-sky-300">[{source.number}]</span>
            <span className="flex-1">{title(source.question)}</span>
          </summary>
          <div className="space-y-2 border-t border-black/5 px-3 py-2 dark:border-white/10">
            {source.excerpt ? (
              <blockquote className="border-l-2 border-emerald-500/60 pl-3 text-[13px] leading-relaxed text-neutral-700 dark:text-neutral-300">
                {source.excerpt}
              </blockquote>
            ) : (
              <p className="text-xs text-neutral-500">The passage text is not available for this source.</p>
            )}
            <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-500">
              <span>From MedQuAD, a collection of answers from NIH health websites</span>
              <a
                href={MEDLINEPLUS_SEARCH + encodeURIComponent(topic(source.question))}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900 dark:text-sky-300"
              >
                Read more on MedlinePlus ↗
              </a>
            </p>
          </div>
        </details>
      ))}
    </section>
  );
}
