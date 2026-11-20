"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import type { ChatEvent, ReplyData } from "@/lib/types";

interface Message {
  id: string;
  role: "patient" | "assistant";
  text: string;
  data?: ReplyData | null;
  progress: string[];
  pending: boolean;
  failed?: boolean;
}

const EXAMPLES = [
  "How is asthma treated?",
  "My 4-year-old has had a high fever and a red rash for two days",
  "What are the early signs of type 2 diabetes?",
  "I get headaches behind my eyes every afternoon",
];

const URGENCY: Record<string, { label: string; style: string }> = {
  self_care: { label: "Self-care", style: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200" },
  routine: { label: "See a clinician soon", style: "bg-sky-100 text-sky-900 dark:bg-sky-900/40 dark:text-sky-200" },
  urgent: { label: "See a clinician today", style: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200" },
  emergency: { label: "Emergency", style: "bg-red-600 text-white" },
};

async function* readEvents(response: Response): AsyncGenerator<ChatEvent> {
  const reader = response.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
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

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);
    const replyId = crypto.randomUUID();
    setMessages((all) => [
      ...all,
      { id: crypto.randomUUID(), role: "patient", text: message, progress: [], pending: false },
      { id: replyId, role: "assistant", text: "", progress: [], pending: true },
    ]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, contextId }),
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

  return (
    <div className="mx-auto flex h-dvh w-full max-w-3xl flex-col px-4">
      <header className="flex items-center justify-between border-b border-black/10 py-4 dark:border-white/10">
        <div>
          <h1 className="text-lg font-semibold">RAGMed</h1>
          <p className="text-sm text-neutral-500">Health questions answered from NIH sources, with citations</p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={restart}
            disabled={busy}
            className="rounded-lg border border-black/15 px-3 py-1.5 text-sm hover:bg-black/5 disabled:opacity-50 dark:border-white/15 dark:hover:bg-white/10"
          >
            New conversation
          </button>
        )}
      </header>

      <main className="flex-1 space-y-6 overflow-y-auto py-6" aria-live="polite">
        {messages.length === 0 ? (
          <div className="space-y-4 pt-10 text-center">
            <p className="text-neutral-500">Describe a symptom or ask a health question.</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  onClick={() => send(example)}
                  className="rounded-xl border border-black/10 p-3 text-left text-sm hover:bg-black/5 dark:border-white/10 dark:hover:bg-white/5"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => <Bubble key={message.id} message={message} />)
        )}
        <div ref={endRef} />
      </main>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          send(input);
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
              send(input);
            }
          }}
          rows={1}
          maxLength={2000}
          placeholder="Describe your symptoms or ask a question…"
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
        Educational project, not a medical device. In an emergency, call your local emergency number.
      </p>
    </div>
  );
}

function Bubble({ message }: { message: Message }) {
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
  const urgency = data?.urgency ? URGENCY[data.urgency] : undefined;

  return (
    <div className="flex">
      <div
        className={`max-w-[95%] space-y-3 rounded-2xl rounded-bl-sm px-4 py-3 ${
          emergency
            ? "border-2 border-red-600 bg-red-50 dark:bg-red-950/40"
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
        {urgency && data?.kind !== "greeting" && (
          <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${urgency.style}`}>
            {urgency.label}
          </span>
        )}
        {message.text && (
          <div className="prose-chat text-[15px] leading-relaxed">
            <ReactMarkdown>{message.text}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
