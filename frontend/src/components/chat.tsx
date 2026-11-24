"use client";

import Link from "next/link";
import Script from "next/script";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";

import { Sources, linkCitations, withoutSourcesBlock } from "@/components/sources";

import type { Appointment, ChatAction, ChatEvent, Offer, ReplyData, Slot, Step } from "@/lib/types";

const CLINIC_TIME_ZONE = process.env.NEXT_PUBLIC_CLINIC_TIME_ZONE ?? "America/St_Johns";
const EMAIL = /^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$/;
const TURNSTILE_SITE_KEY = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;
const CHECK_ENABLED = Boolean(TURNSTILE_SITE_KEY);

declare global {
  interface Window {
    turnstile?: {
      render: (element: HTMLElement, options: Record<string, unknown>) => string;
      reset: (widget?: string) => void;
    };
  }
}

/** Cloudflare's bot check, shown only when Cloudflare wants an interaction. */
function useTurnstile() {
  const container = useRef<HTMLDivElement>(null);
  const widget = useRef<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const render = useCallback(() => {
    if (!TURNSTILE_SITE_KEY || !window.turnstile || !container.current || widget.current) return;
    widget.current = window.turnstile.render(container.current, {
      sitekey: TURNSTILE_SITE_KEY,
      appearance: "interaction-only",
      callback: (value: string) => setToken(value),
      "expired-callback": () => setToken(null),
      "error-callback": () => setToken(null),
    });
  }, []);
  const reset = useCallback(() => {
    setToken(null);
    if (widget.current) window.turnstile?.reset(widget.current);
  }, []);
  useEffect(render, [render]);
  return [container, token, render, reset] as const;
}

interface Message {
  id: string;
  role: "patient" | "assistant";
  text: string;
  data?: ReplyData | null;
  progress: Step[]; // what each agent did while this reply was prepared
  pending: boolean;
  failed?: boolean;
}

interface Outgoing {
  text?: string;
  action?: ChatAction;
  label?: string; // what the patient's bubble shows for a button press
}

const EXAMPLES = [
  "I've had headaches on one side for a month",
  "My 3-year-old has a fever and a rash",
  "I need to renew my blood pressure prescription",
  "What causes asthma?",
];

/** The care team: each agent's colour and what it does. */
const AGENTS: Record<string, { dot: string; role: string }> = {
  "Care Coordinator": { dot: "bg-sky-500", role: "Talks with you, decides what to ask, and briefs your clinician" },
  "Diagnosis agent": { dot: "bg-violet-500", role: "Weighs the medical evidence as you talk" },
  "Knowledge base": { dot: "bg-emerald-500", role: "Searches the clinic's medical sources (MedQuAD)" },
  Scheduling: { dot: "bg-amber-500", role: "Finds open times and books your visit" },
};
const COORDINATOR = "Care Coordinator";

const URGENCY: Record<string, { label: string; style: string }> = {
  self_care: { label: "Self-care", style: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200" },
  routine: { label: "Routine", style: "bg-sky-100 text-sky-900 dark:bg-sky-900/40 dark:text-sky-200" },
  urgent: { label: "Urgent: today", style: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200" },
  emergency: { label: "Emergency", style: "bg-red-600 text-white" },
};

const dayFormat = new Intl.DateTimeFormat("en-CA", {
  timeZone: CLINIC_TIME_ZONE,
  weekday: "long",
  month: "long",
  day: "numeric",
});
const timeFormat = new Intl.DateTimeFormat("en-CA", {
  timeZone: CLINIC_TIME_ZONE,
  hour: "numeric",
  minute: "2-digit",
});

export function formatWhen(iso: string): string {
  const moment = new Date(iso);
  return `${dayFormat.format(moment)}, ${timeFormat.format(moment)}`;
}

async function* readEvents(response: Response): AsyncGenerator<ChatEvent> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) yield JSON.parse(line) as ChatEvent;
  }
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [contextId, setContextId] = useState<string | undefined>();
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [verified, setVerified] = useState(false);
  const [checkBox, checkToken, renderCheck, resetCheck] = useTurnstile();
  const waitingForCheck = CHECK_ENABLED && !verified && !checkToken;
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const update = (id: string, change: (m: Message) => Message) =>
    setMessages((all) => all.map((m) => (m.id === id ? change(m) : m)));

  async function send({ text = "", action, label }: Outgoing) {
    const message = text.trim();
    if ((!message && !action) || busy || waitingForCheck) return;
    const turnstileToken = CHECK_ENABLED && !verified ? checkToken : undefined;
    setInput("");
    setBusy(true);
    const replyId = crypto.randomUUID();
    setMessages((all) => [
      ...all,
      { id: crypto.randomUUID(), role: "patient", text: label ?? message, progress: [], pending: false },
      { id: replyId, role: "assistant", text: "", progress: [], pending: true },
    ]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, contextId, action, turnstileToken }),
      });
      if (turnstileToken) resetCheck(); // each token works once
      if (!response.ok || !response.body) {
        const error = await response.json().catch(() => ({ error: "Request failed." }));
        if (error.code === "verification_required") setVerified(false);
        throw new Error(error.error ?? "Request failed.");
      }
      if (CHECK_ENABLED) setVerified(true); // the server set a session cookie
      for await (const event of readEvents(response)) {
        if (event.type === "progress") {
          const step = { agent: event.agent ?? COORDINATOR, text: event.text };
          update(replyId, (m) => ({ ...m, progress: [...m.progress, step] }));
        } else if (event.type === "reply") {
          if (event.contextId) setContextId(event.contextId);
          update(replyId, (m) => ({
            ...m,
            text: event.text,
            data: event.data,
            pending: false,
            failed: event.state !== "completed",
          }));
        } else {
          update(replyId, (m) => ({ ...m, text: event.text, pending: false, failed: true }));
        }
      }
    } catch (error) {
      const text = error instanceof Error ? error.message : "Something went wrong.";
      update(replyId, (m) => ({ ...m, text, pending: false, failed: true }));
    } finally {
      setBusy(false);
    }
  }

  function restart() {
    setMessages([]);
    setContextId(undefined);
  }

  // The newest reply's buttons are the live ones. After a booking step fails (say, a mistyped
  // email), the time picker or code box just before it stays usable so the patient can retry.
  const last = messages.at(-1);
  const retryable = last?.data?.kind === "no_answer" || (last?.failed && !last.pending);
  const card = messages.at(-3); // the card, the patient's button press, then the error
  const isCard = card?.data?.kind === "slots" || card?.data?.kind === "verify_email";
  const live = new Set([last?.id, retryable && isCard ? card.id : undefined]);
  const working = busy ? (last?.progress.at(-1)?.agent ?? COORDINATOR) : undefined;

  return (
    <div
      className="mx-auto flex h-dvh w-full max-w-3xl flex-col px-4"
      data-check={waitingForCheck ? "pending" : "ready"}
    >
      {CHECK_ENABLED && (
        <Script
          src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit"
          strategy="afterInteractive"
          onReady={renderCheck}
        />
      )}
      <header className="flex items-center justify-between gap-4 border-b border-black/10 py-4 dark:border-white/10">
        <div>
          <h1 className="text-lg font-semibold">RAGMed Clinic</h1>
          <p className="text-sm text-neutral-500">
            Tell us what&apos;s wrong. Our care team asks only what matters, briefs your clinician, and books your
            visit.
          </p>
          <CareTeam working={working} />
        </div>
        {messages.length > 0 && (
          <button
            onClick={restart}
            disabled={busy}
            className="shrink-0 rounded-lg border border-black/15 px-3 py-1.5 text-sm hover:bg-black/5 disabled:opacity-50 dark:border-white/15 dark:hover:bg-white/10"
          >
            New conversation
          </button>
        )}
      </header>

      <main className="flex-1 space-y-6 overflow-y-auto py-6" aria-live="polite">
        {messages.length === 0 ? (
          <div className="space-y-4 pt-10 text-center">
            <p className="text-neutral-500">What would you like to be seen for?</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  disabled={waitingForCheck}
                  onClick={() => send({ text: example })}
                  className="rounded-xl border border-black/10 p-3 text-left text-sm hover:bg-black/5 dark:border-white/10 dark:hover:bg-white/5"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <Bubble
              key={message.id}
              message={message}
              active={live.has(message.id) && !busy}
              onSend={send}
            />
          ))
        )}
        <div ref={endRef} />
      </main>

      <div ref={checkBox} className="flex justify-center empty:hidden" />
      <form
        onSubmit={(event) => {
          event.preventDefault();
          send({ text: input });
        }}
        className="flex gap-2 border-t border-black/10 py-4 dark:border-white/10"
      >
        <label htmlFor="message" className="sr-only">
          Your message
        </label>
        <textarea
          id="message"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send({ text: input });
            }
          }}
          rows={1}
          maxLength={2000}
          placeholder="Type your answer or describe your symptoms…"
          className="min-h-11 flex-1 resize-none rounded-xl border border-black/15 bg-transparent px-4 py-2.5 outline-none focus:border-sky-500 dark:border-white/15"
        />
        <button
          type="submit"
          disabled={busy || !input.trim() || waitingForCheck}
          title={waitingForCheck ? "Running a quick security check…" : undefined}
          className="rounded-xl bg-sky-600 px-5 font-medium text-white hover:bg-sky-700 disabled:opacity-40"
        >
          Send
        </button>
      </form>
      <p className="pb-3 text-center text-xs text-neutral-500">
        Educational demo, not a medical device; physician names are placeholders. Please don&apos;t enter real
        health data. In an emergency, call your local emergency number. ·{" "}
        <Link href="/clinician" className="underline hover:text-neutral-700 dark:hover:text-neutral-300">
          Clinician view
        </Link>
      </p>
    </div>
  );
}

function Bubble({
  message,
  active,
  onSend,
}: {
  message: Message;
  active: boolean;
  onSend: (outgoing: Outgoing) => void;
}) {
  const [openSources, setOpenSources] = useState<Set<number>>(new Set());
  const openSource = (number: number) => {
    setOpenSources((all) => new Set(all).add(number));
    requestAnimationFrame(() =>
      document.getElementById(`source-${message.id}-${number}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  };

  if (message.role === "patient") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-sky-600 px-4 py-2.5 text-white">
          {message.text}
        </div>
      </div>
    );
  }

  const data = message.data;
  const sources = data?.sources ?? [];
  const prefix = `source-${message.id}`;
  const emergency = data?.kind === "emergency" || data?.urgency === "emergency";
  const booked = data?.kind === "booked";
  const urgency = data?.urgency ? URGENCY[data.urgency] : undefined;
  // A general answer is not about the patient: only an urgent one earns a badge.
  const pressing = data?.urgency === "urgent" || data?.urgency === "emergency";
  const showBadge = urgency && (data?.kind === "report" || emergency || (data?.kind === "answer" && pressing));

  return (
    <div className="flex" data-kind={data?.kind ?? (message.pending ? "pending" : "error")}>
      <div
        className={`w-full max-w-[95%] space-y-3 rounded-2xl rounded-bl-sm px-4 py-3 ${
          emergency
            ? "border-2 border-red-600 bg-red-50 dark:bg-red-950/40"
            : booked
              ? "border border-emerald-500/60 bg-emerald-50 dark:bg-emerald-950/30"
              : message.failed
                ? "border border-amber-400/60 bg-amber-50 dark:bg-amber-950/30"
                : "bg-black/[0.04] dark:bg-white/[0.06]"
        }`}
      >
        {message.pending &&
          (message.progress.length > 0 ? (
            <StepList steps={message.progress} live />
          ) : (
            <p className="flex items-center gap-2 text-sm text-neutral-500">
              <span className="size-2 animate-pulse rounded-full bg-sky-500" />
              Thinking…
            </p>
          ))}
        {showBadge && (
          <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${urgency.style}`}>
            {urgency.label}
          </span>
        )}
        {booked && data?.appointment ? (
          <BookedCard appointment={data.appointment} />
        ) : (
          message.text && (
            <div className="prose-chat text-[15px] leading-relaxed">
              <ReactMarkdown components={markdownLinks(prefix, openSource)}>
                {sources.length ? linkCitations(withoutSourcesBlock(message.text), prefix, sources) : message.text}
              </ReactMarkdown>
            </div>
          )
        )}
        <Sources
          sources={sources}
          prefix={prefix}
          open={openSources}
          onToggle={(number, isOpen) =>
            setOpenSources((all) => {
              const next = new Set(all);
              if (isOpen) next.add(number);
              else next.delete(number);
              return next;
            })
          }
        />
        {(data?.kind === "question" || data?.kind === "emergency") &&
          data.quick_replies.length > 0 &&
          (data.multi_select ? (
            <MultipleChoice options={data.quick_replies} active={active} onSend={onSend} />
          ) : (
            <QuickReplies replies={data.quick_replies} active={active} onSend={onSend} />
          ))}
        {(data?.kind === "report" || data?.kind === "answer") && data.offer && (
          <OfferButtons offer={data.offer} active={active} onSend={onSend} />
        )}
        {data?.kind === "slots" && data.offer && (
          <SlotPicker slots={data.slots} offer={data.offer} active={active} onSend={onSend} />
        )}
        {data?.kind === "verify_email" && data.verification && (
          <CodeEntry active={active} onSend={onSend} />
        )}
        {!message.pending && message.progress.length > 1 && <TeamLog steps={message.progress} />}
      </div>
    </div>
  );
}

/** Citation links ("#source-…-2") open that source in place; other links open a new tab. */
function markdownLinks(prefix: string, openSource: (number: number) => void): Components {
  return {
    a: ({ href = "", children }) => {
      if (href.startsWith(`#${prefix}-`)) {
        const number = Number(href.slice(prefix.length + 2));
        return (
          <a
            href={href}
            onClick={(event) => {
              event.preventDefault();
              openSource(number);
            }}
            className="align-super text-[11px] font-medium text-sky-700 no-underline hover:underline dark:text-sky-300"
            aria-label={`Source ${number}`}
          >
            {children}
          </a>
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" className="underline">
          {children}
        </a>
      );
    },
  };
}

function dotFor(agent: string): string {
  return AGENTS[agent]?.dot ?? "bg-neutral-400";
}

/** The four agents, with the one working right now lit up. */
function CareTeam({ working }: { working?: string }) {
  return (
    <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Care team">
      {Object.entries(AGENTS).map(([name, agent]) => {
        const on = working === name;
        return (
          <li
            key={name}
            title={agent.role}
            data-working={on}
            className={`flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs transition-colors ${
              on
                ? "border-black/25 bg-black/5 text-neutral-900 dark:border-white/30 dark:bg-white/10 dark:text-neutral-100"
                : "border-black/10 text-neutral-500 dark:border-white/10"
            }`}
          >
            <span className={`size-1.5 rounded-full ${agent.dot} ${on ? "animate-pulse" : "opacity-60"}`} />
            {name}
          </li>
        );
      })}
    </ul>
  );
}

/** Each step as "Agent · what it did"; while `live`, the newest one pulses. */
function StepList({ steps, live = false }: { steps: Step[]; live?: boolean }) {
  return (
    <ol className="space-y-1.5 text-sm" aria-label="Care team activity">
      {steps.map((step, index) => {
        const current = live && index === steps.length - 1;
        return (
          <li key={index} className={`flex gap-2 ${current ? "" : "text-neutral-500"}`}>
            <span
              className={`mt-[0.45rem] size-2 shrink-0 rounded-full ${dotFor(step.agent)} ${current ? "animate-pulse" : ""}`}
            />
            <span>
              <span className="font-medium text-neutral-700 dark:text-neutral-300">{step.agent}</span> · {step.text}
              {current && "…"}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/** After a reply: what the care team did to prepare it, folded away. */
function TeamLog({ steps }: { steps: Step[] }) {
  const agents = [...new Set(steps.map((step) => step.agent))];
  return (
    <details className="border-t border-black/5 pt-2 dark:border-white/10">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-xs text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300">
        <span className="flex -space-x-0.5">
          {agents.map((agent) => (
            <span key={agent} className={`size-2 rounded-full ring-2 ring-white dark:ring-neutral-900 ${dotFor(agent)}`} />
          ))}
        </span>
        How the care team worked on this · {steps.length} steps
      </summary>
      <div className="mt-2">
        <StepList steps={steps} />
      </div>
    </details>
  );
}

const chip =
  "rounded-full border border-sky-600/40 px-3 py-1 text-sm text-sky-800 hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-40 dark:text-sky-200 dark:hover:bg-sky-950";
const primary =
  "rounded-xl bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-40";
const secondary =
  "rounded-xl border border-black/15 px-4 py-2 text-sm hover:bg-black/5 disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/15 dark:hover:bg-white/10";

function QuickReplies({
  replies,
  active,
  onSend,
}: {
  replies: string[];
  active: boolean;
  onSend: (outgoing: Outgoing) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="Suggested answers">
      {replies.map((reply) => (
        <button key={reply} className={chip} disabled={!active} onClick={() => onSend({ text: reply })}>
          {reply}
        </button>
      ))}
    </div>
  );
}

const NONE = "None of these";

/** Tick any number of options (or "None of these"), then send them as one answer. */
function MultipleChoice({
  options,
  active,
  onSend,
}: {
  options: string[];
  active: boolean;
  onSend: (outgoing: Outgoing) => void;
}) {
  const [picked, setPicked] = useState<string[]>([]);
  const toggle = (option: string) =>
    setPicked((all) =>
      option === NONE
        ? all.includes(NONE) ? [] : [NONE]
        : all.includes(option)
          ? all.filter((o) => o !== option)
          : [...all.filter((o) => o !== NONE), option],
    );
  const answer = picked.includes(NONE) ? "None of these" : picked.join(", ");
  return (
    <div className="space-y-2" role="group" aria-label="Choose all that apply">
      <p className="text-xs text-neutral-500">Choose all that apply, or type your own answer.</p>
      <div className="flex flex-wrap gap-2">
        {[...options, NONE].map((option) => {
          const on = picked.includes(option);
          return (
            <button
              key={option}
              role="checkbox"
              aria-checked={on}
              disabled={!active}
              onClick={() => toggle(option)}
              className={`${chip} ${on ? "border-sky-600 bg-sky-600 text-white hover:bg-sky-700 dark:text-white dark:hover:bg-sky-700" : ""}`}
            >
              {on ? "✓ " : ""}
              {option}
            </button>
          );
        })}
      </div>
      <button className={primary} disabled={!active || picked.length === 0} onClick={() => onSend({ text: answer })}>
        {picked.length === 0 ? "Select one or more" : picked.includes(NONE) ? "Send: none of these" : `Send ${picked.length} selected`}
      </button>
    </div>
  );
}

function OfferButtons({
  offer,
  active,
  onSend,
}: {
  offer: Offer;
  active: boolean;
  onSend: (outgoing: Outgoing) => void;
}) {
  const minutes = Math.round(offer.visit_minutes);
  return (
    <div className="flex flex-wrap gap-2 border-t border-black/10 pt-3 dark:border-white/10">
      <button
        className={primary}
        disabled={!active}
        onClick={() =>
          onSend({ action: { type: "show_slots" }, label: `Show times with ${offer.clinician_type}` })
        }
      >
        See available times · {minutes} min with {offer.clinician_type}
      </button>
      {offer.alternative && (
        <button
          className={secondary}
          disabled={!active}
          onClick={() =>
            onSend({
              action: { type: "show_slots", clinician_type: offer.alternative! },
              label: `Show times with a ${offer.alternative!.toLowerCase()}`,
            })
          }
        >
          A {offer.alternative.toLowerCase()} instead
        </button>
      )}
    </div>
  );
}

function SlotPicker({
  slots,
  offer,
  active,
  onSend,
}: {
  slots: Slot[];
  offer: Offer;
  active: boolean;
  onSend: (outgoing: Outgoing) => void;
}) {
  const [chosen, setChosen] = useState<Slot | null>(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const emailOk = EMAIL.test(email.trim());
  const ready = name.trim().length > 0 && emailOk;
  const days = new Map<string, Slot[]>();
  for (const slot of slots) {
    const day = dayFormat.format(new Date(slot.start));
    days.set(day, [...(days.get(day) ?? []), slot]);
  }
  const minutes = Math.round(offer.visit_minutes);

  return (
    <div className="space-y-3 border-t border-black/10 pt-3 dark:border-white/10">
      {[...days.entries()].map(([day, daySlots]) => (
        <div key={day}>
          <p className="mb-1.5 text-sm font-medium">{day}</p>
          <div className="flex flex-wrap gap-2">
            {daySlots.map((slot) => (
              <button
                key={slot.start}
                disabled={!active}
                onClick={() => setChosen(slot)}
                aria-pressed={chosen?.start === slot.start}
                className={`rounded-lg border px-3 py-1.5 text-left text-sm disabled:cursor-not-allowed disabled:opacity-40 ${
                  chosen?.start === slot.start
                    ? "border-sky-600 bg-sky-600 text-white"
                    : "border-black/15 hover:border-sky-500 dark:border-white/15"
                }`}
              >
                <span className="font-medium">{timeFormat.format(new Date(slot.start))}</span>
                <span className="block text-xs opacity-75">{slot.physician}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
      {chosen && active && (
        <form
          className="space-y-3 rounded-xl bg-sky-50 p-3 dark:bg-sky-950/40"
          aria-label="Confirm booking"
          onSubmit={(event) => {
            event.preventDefault();
            if (!ready) return;
            onSend({
              action: {
                type: "book",
                start: chosen.start,
                clinician_type: chosen.clinician_type,
                name: name.trim(),
                email: email.trim(),
              },
              label: `Book ${formatWhen(chosen.start)} with ${chosen.physician}`,
            });
          }}
        >
          <p className="text-sm">
            Book <strong>{formatWhen(chosen.start)}</strong> with {chosen.physician} ({chosen.clinician_type},{" "}
            {minutes} min). We&apos;ll email you a code to confirm it&apos;s you, then send your invitation.
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            <label className="text-sm">
              <span className="mb-1 block text-neutral-600 dark:text-neutral-400">Your name</span>
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                maxLength={80}
                autoComplete="name"
                required
                className="w-full rounded-lg border border-black/15 bg-white px-3 py-1.5 outline-none focus:border-sky-500 dark:border-white/15 dark:bg-transparent"
              />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-neutral-600 dark:text-neutral-400">Email for the invitation</span>
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                maxLength={254}
                autoComplete="email"
                required
                aria-invalid={email.length > 0 && !emailOk}
                className="w-full rounded-lg border border-black/15 bg-white px-3 py-1.5 outline-none focus:border-sky-500 aria-[invalid=true]:border-red-500 dark:border-white/15 dark:bg-transparent"
              />
            </label>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="submit" className={primary} disabled={!ready}>
              Email me a code
            </button>
            <button type="button" className={secondary} onClick={() => setChosen(null)}>
              Choose another
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

function CodeEntry({ active, onSend }: { active: boolean; onSend: (outgoing: Outgoing) => void }) {
  const [code, setCode] = useState("");
  const ready = /^\d{6}$/.test(code);
  if (!active) return null;
  return (
    <form
      className="space-y-3 rounded-xl bg-sky-50 p-3 dark:bg-sky-950/40"
      aria-label="Enter your code"
      onSubmit={(event) => {
        event.preventDefault();
        if (ready) onSend({ action: { type: "confirm", code }, label: "Entered the code" });
      }}
    >
      <label className="block text-sm">
        <span className="mb-1 block text-neutral-600 dark:text-neutral-400">6-digit code from the email</span>
        <input
          value={code}
          onChange={(event) => setCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
          inputMode="numeric"
          autoComplete="one-time-code"
          placeholder="123456"
          className="w-40 rounded-lg border border-black/15 bg-white px-3 py-1.5 font-mono text-lg tracking-[0.3em] outline-none focus:border-sky-500 dark:border-white/15 dark:bg-transparent"
        />
      </label>
      <div className="flex flex-wrap gap-2">
        <button type="submit" className={primary} disabled={!ready}>
          Confirm booking
        </button>
        <button
          type="button"
          className={secondary}
          onClick={() => onSend({ action: { type: "resend" }, label: "Send a new code" })}
        >
          Send a new code
        </button>
      </div>
      <p className="text-xs text-neutral-500">Not there? Check your spam folder.</p>
    </form>
  );
}

function BookedCard({ appointment }: { appointment: Appointment }) {
  return (
    <div className="space-y-2">
      <p className="text-base font-semibold text-emerald-800 dark:text-emerald-300">Your visit is booked</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-neutral-500">When</dt>
        <dd>{formatWhen(appointment.start)}</dd>
        <dt className="text-neutral-500">With</dt>
        <dd>
          {appointment.physician} · {appointment.clinician_type}
        </dd>
        <dt className="text-neutral-500">Length</dt>
        <dd>{Math.round(appointment.minutes)} minutes</dd>
        <dt className="text-neutral-500">Reference</dt>
        <dd className="font-mono font-semibold">{appointment.reference}</dd>
      </dl>
      <p className="text-sm text-neutral-600 dark:text-neutral-400">
        Your calendar invitation is on its way by email, and your draft pre-visit report has been sent to{" "}
        {appointment.physician}. It is not a diagnosis; your clinician will assess you at the visit.
      </p>
    </div>
  );
}
