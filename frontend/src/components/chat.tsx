"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import type { Appointment, ChatAction, ChatEvent, Offer, ReplyData, Slot } from "@/lib/types";

const CLINIC_TIME_ZONE = process.env.NEXT_PUBLIC_CLINIC_TIME_ZONE ?? "America/St_Johns";
const EMAIL = /^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$/;

interface Message {
  id: string;
  role: "patient" | "assistant";
  text: string;
  data?: ReplyData | null;
  progress: string[];
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
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const update = (id: string, change: (m: Message) => Message) =>
    setMessages((all) => all.map((m) => (m.id === id ? change(m) : m)));

  async function send({ text = "", action, label }: Outgoing) {
    const message = text.trim();
    if ((!message && !action) || busy) return;
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
        body: JSON.stringify({ message, contextId, action }),
      });
      if (!response.ok || !response.body) {
        const error = await response.json().catch(() => ({ error: "Request failed." }));
        throw new Error(error.error ?? "Request failed.");
      }
      for await (const event of readEvents(response)) {
        if (event.type === "progress") {
          update(replyId, (m) => ({ ...m, progress: [...m.progress, event.text] }));
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

  const latest = messages.at(-1)?.id;

  return (
    <div className="mx-auto flex h-dvh w-full max-w-3xl flex-col px-4">
      <header className="flex items-center justify-between gap-4 border-b border-black/10 py-4 dark:border-white/10">
        <div>
          <h1 className="text-lg font-semibold">RAGMed Clinic</h1>
          <p className="text-sm text-neutral-500">
            Tell us what&apos;s wrong. We ask a few questions, draft a note for your clinician, and book your visit.
          </p>
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
              active={message.id === latest && !busy}
              onSend={send}
            />
          ))
        )}
        <div ref={endRef} />
      </main>

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
          disabled={busy || !input.trim()}
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
  const emergency = data?.kind === "emergency" || data?.urgency === "emergency";
  const booked = data?.kind === "booked";
  const urgency = data?.urgency ? URGENCY[data.urgency] : undefined;
  const showBadge = urgency && (data?.kind === "report" || data?.kind === "answer" || emergency);

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
        {message.pending && (
          <p className="flex items-center gap-2 text-sm text-neutral-500">
            <span className="size-2 animate-pulse rounded-full bg-sky-500" />
            {message.progress.at(-1) ?? "Thinking"}…
          </p>
        )}
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
              <ReactMarkdown>{message.text}</ReactMarkdown>
            </div>
          )
        )}
        {data?.kind === "question" && data.quick_replies.length > 0 && (
          <QuickReplies replies={data.quick_replies} active={active} onSend={onSend} />
        )}
        {data?.kind === "report" && data.offer && (
          <OfferButtons offer={data.offer} active={active} onSend={onSend} />
        )}
        {data?.kind === "slots" && data.offer && (
          <SlotPicker slots={data.slots} offer={data.offer} active={active} onSend={onSend} />
        )}
      </div>
    </div>
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
            {minutes} min). We&apos;ll email your invitation.
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
              Confirm booking
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
