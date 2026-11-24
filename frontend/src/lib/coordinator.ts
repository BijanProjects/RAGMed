import "server-only";

import type { ChatAction, ChatEvent, ReplyData } from "./types";

// Server-side only: the browser never sees the backend URL or its token.
const COORDINATOR_URL = process.env.COORDINATOR_URL ?? "http://127.0.0.1:8000";
const COORDINATOR_TOKEN = process.env.COORDINATOR_TOKEN;
const TIMEOUT_MS = 90_000;

type Part = { text?: string; data?: unknown };
type A2AResult = {
  task?: { id: string; contextId: string };
  statusUpdate?: {
    contextId: string;
    status: { state: string; message?: { parts?: Part[] } };
  };
  artifactUpdate?: { contextId: string; artifact: { parts?: Part[] } };
};

const FINAL_STATES: Record<string, string> = {
  TASK_STATE_COMPLETED: "completed",
  TASK_STATE_FAILED: "failed",
  TASK_STATE_REJECTED: "rejected",
  TASK_STATE_CANCELED: "canceled",
};

function textOf(parts: Part[] | undefined): string {
  return (parts ?? []).map((p) => p.text ?? "").join("").trim();
}

/** Who is acting, from a status message's data part ({"agent": "..."}), if it says. */
function agentOf(parts: Part[] | undefined): string | undefined {
  for (const part of parts ?? []) {
    const agent = (part.data as { agent?: unknown } | undefined)?.agent;
    if (typeof agent === "string" && agent.length <= 40) return agent;
  }
  return undefined;
}

/** Parse a Server-Sent Events stream into the JSON payload of each `data:` event. */
async function* sseEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.search(/\r?\n\r?\n/)) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary).replace(/^\r?\n\r?\n/, "");
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) yield JSON.parse(data);
    }
  }
}

/**
 * Send one message (text, a button action, or both) to the Care Coordinator over A2A
 * (JSON-RPC, streamed) and yield its progress, then exactly one reply or error.
 */
export async function* askCoordinator(
  message: string,
  contextId?: string,
  action?: ChatAction,
  clientIp?: string,
): AsyncGenerator<ChatEvent> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
    "A2A-Version": "1.0",
  };
  if (COORDINATOR_TOKEN) headers.Authorization = `Bearer ${COORDINATOR_TOKEN}`;
  // The Coordinator's rate limits are per visitor; it trusts this header only from this app.
  if (clientIp) headers["X-RAGMed-Client-IP"] = clientIp;

  const request = {
    jsonrpc: "2.0",
    id: crypto.randomUUID(),
    method: "SendStreamingMessage",
    params: {
      message: {
        messageId: crypto.randomUUID(),
        role: "ROLE_USER",
        // A2A refuses empty text parts: a button press with no text sends only its data.
        parts: [
          ...(message ? [{ text: message }] : []),
          ...(action ? [{ data: { action } }] : []),
        ],
        ...(contextId ? { contextId } : {}),
      },
      configuration: {},
    },
  };

  let response: Response;
  try {
    response = await fetch(`${COORDINATOR_URL}/a2a`, {
      method: "POST",
      headers,
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch {
    yield { type: "error", text: "The RAGMed service is not reachable right now." };
    return;
  }
  if (!response.ok || !response.body) {
    yield { type: "error", text: `The RAGMed service returned ${response.status}.` };
    return;
  }

  let text = "";
  let data: ReplyData | null = null;
  let context = contextId ?? "";
  let state = "failed";
  for await (const event of sseEvents(response.body)) {
    const envelope = event as { result?: A2AResult; error?: { message?: string } };
    if (envelope.error) {
      yield { type: "error", text: envelope.error.message ?? "The request failed." };
      return;
    }
    const result = envelope.result ?? {};
    if (result.task) context = result.task.contextId;
    if (result.artifactUpdate) {
      const parts = result.artifactUpdate.artifact.parts ?? [];
      text = textOf(parts) || text;
      const structured = parts.find((p) => p.data !== undefined)?.data;
      if (structured) data = structured as ReplyData;
    }
    if (result.statusUpdate) {
      context = result.statusUpdate.contextId || context;
      const { state: raw, message: statusMessage } = result.statusUpdate.status;
      const messageText = textOf(statusMessage?.parts);
      if (raw in FINAL_STATES) {
        state = FINAL_STATES[raw];
        if (messageText && !text) text = messageText;
      } else if (messageText) {
        yield { type: "progress", text: messageText, agent: agentOf(statusMessage?.parts) };
      }
    }
  }
  yield { type: "reply", text, data, contextId: context, state };
}
