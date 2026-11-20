import { askCoordinator } from "@/lib/coordinator";
import type { ChatAction, ChatEvent } from "@/lib/types";

const MAX_MESSAGE_CHARS = 2000;
const ISO_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?([+-]\d{2}:\d{2}|Z)$/;

/** Only the two known buttons, with only their known fields, reach the backend. */
function parseAction(value: unknown): ChatAction | undefined | null {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const kind = typeof raw.clinician_type === "string" ? raw.clinician_type.slice(0, 60) : undefined;
  if (raw.type === "show_slots") return { type: "show_slots", clinician_type: kind };
  if (raw.type === "book" && typeof raw.start === "string" && ISO_TIME.test(raw.start)) {
    return { type: "book", start: raw.start, clinician_type: kind };
  }
  return null;
}

/**
 * POST { message?, contextId?, action? } → a stream of ChatEvent objects, one per line (NDJSON).
 * The browser only ever talks to this route; the backend stays private.
 */
export async function POST(request: Request): Promise<Response> {
  let body: { message?: unknown; contextId?: unknown; action?: unknown };
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Send JSON: { message?, contextId?, action? }" }, { status: 400 });
  }
  const message = typeof body.message === "string" ? body.message.trim() : "";
  const contextId = typeof body.contextId === "string" ? body.contextId.slice(0, 100) : undefined;
  const action = parseAction(body.action);
  if (action === null) {
    return Response.json({ error: "Unknown action." }, { status: 400 });
  }
  if ((!message && !action) || message.length > MAX_MESSAGE_CHARS) {
    return Response.json(
      { error: `A message must be 1 to ${MAX_MESSAGE_CHARS} characters.` },
      { status: 400 },
    );
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: ChatEvent) =>
        controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
      try {
        for await (const event of askCoordinator(message, contextId, action)) send(event);
      } catch {
        send({ type: "error", text: "Something went wrong. Please try again." });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
