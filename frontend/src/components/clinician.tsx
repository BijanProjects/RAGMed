"use client";

import Link from "next/link";
import { useState } from "react";
import ReactMarkdown from "react-markdown";

import { formatWhen } from "@/components/chat";
import { Sources } from "@/components/sources";
import type { ClinicAppointment } from "@/lib/types";

const URGENCY: Record<string, string> = {
  emergency: "bg-red-600 text-white",
  urgent: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
  routine: "bg-sky-100 text-sky-900 dark:bg-sky-900/40 dark:text-sky-200",
  self_care: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200",
};

export function ClinicianView() {
  const [code, setCode] = useState("");
  const [visits, setVisits] = useState<ClinicAppointment[] | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function load(event?: React.FormEvent) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/clinician", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? "Request failed.");
      setVisits(body as ClinicAppointment[]);
    } catch (problem) {
      setVisits(null);
      setError(problem instanceof Error ? problem.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto w-full max-w-4xl space-y-6 px-4 py-8">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">RAGMed Clinic · Clinician view</h1>
          <p className="text-sm text-neutral-500">
            Upcoming visits with each patient&apos;s draft pre-visit report, most urgent first. Reports are
            patient-reported and machine-drafted: not diagnoses.
          </p>
        </div>
        <Link href="/" className="shrink-0 text-sm underline">
          Patient chat
        </Link>
      </header>

      {visits === null ? (
        <form onSubmit={load} className="flex max-w-sm gap-2">
          <label htmlFor="code" className="sr-only">
            Access code
          </label>
          <input
            id="code"
            type="password"
            value={code}
            onChange={(event) => setCode(event.target.value)}
            placeholder="Access code"
            autoComplete="current-password"
            className="flex-1 rounded-xl border border-black/15 bg-transparent px-4 py-2 outline-none focus:border-sky-500 dark:border-white/15"
          />
          <button
            type="submit"
            disabled={loading || !code}
            className="rounded-xl bg-sky-600 px-4 py-2 font-medium text-white disabled:opacity-40"
          >
            Open
          </button>
        </form>
      ) : (
        <div className="flex items-center justify-between">
          <p className="text-sm text-neutral-500">
            {visits.length} upcoming {visits.length === 1 ? "visit" : "visits"}
          </p>
          <button onClick={() => load()} disabled={loading} className="text-sm underline disabled:opacity-40">
            Refresh
          </button>
        </div>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {visits?.map((visit) => (
        <article
          key={visit.reference}
          data-urgency={visit.urgency}
          className="space-y-3 rounded-2xl border border-black/10 p-5 dark:border-white/10"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${URGENCY[visit.urgency] ?? ""}`}>
              {visit.urgency.replace("_", " ")}
            </span>
            {visit.report.urgency_score && (
              <span className="rounded-full border border-black/15 px-2 py-0.5 text-xs font-medium dark:border-white/20">
                Score {visit.report.urgency_score}/5
              </span>
            )}
            <span className="font-mono text-sm">{visit.reference}</span>
            {visit.patient_name && <span className="text-sm font-medium">{visit.patient_name}</span>}
            <span className="text-sm text-neutral-500">
              {formatWhen(visit.start)} · {Math.round(visit.minutes)} min · {visit.physician} ({visit.clinician_type})
            </span>
          </div>
          {visit.report.impression && (
            <div className="rounded-xl bg-sky-50 p-3 text-sm dark:bg-sky-950/40">
              <p className="font-medium">Care Coordinator&apos;s handoff</p>
              <p>{visit.report.impression}</p>
              {visit.report.score_reason && (
                <p className="mt-1 text-neutral-600 dark:text-neutral-400">
                  Urgency {visit.report.urgency_score}/5: {visit.report.score_reason}
                </p>
              )}
              {(visit.report.open_questions ?? []).length > 0 && (
                <ul className="mt-1 list-disc pl-5">
                  {visit.report.open_questions!.map((question) => (
                    <li key={question}>{question}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            {Object.entries(visit.report.case).map(([label, value]) => (
              <div key={label} className="contents">
                <dt className="text-neutral-500">{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
          {visit.report.assessment && (
            <div className="prose-chat text-sm">
              <p className="font-medium">Draft assessment (from MedQuAD)</p>
              <ReactMarkdown>{visit.report.assessment}</ReactMarkdown>
            </div>
          )}
          {visit.report.possible_conditions.length > 0 && (
            <p className="text-sm">
              <span className="font-medium">To consider:</span> {visit.report.possible_conditions.join(", ")}
            </p>
          )}
          <ClinicianSources visit={visit} />
          <p className="text-xs text-neutral-500">{visit.report.notice}</p>
        </article>
      ))}
    </div>
  );
}

function ClinicianSources({ visit }: { visit: ClinicAppointment }) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  return (
    <Sources
      sources={visit.report.sources}
      prefix={`source-${visit.reference}`}
      open={open}
      onToggle={(number, isOpen) =>
        setOpen((all) => {
          const next = new Set(all);
          if (isOpen) next.add(number);
          else next.delete(number);
          return next;
        })
      }
    />
  );
}
