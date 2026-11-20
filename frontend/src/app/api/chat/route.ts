import { askCoordinator } from "@/lib/coordinator";
import type { ChatEvent } from "@/lib/types";

const MAX_MESSAGE_CHARS = 2000;

/**
 * POST { message, contextId? } → a stream of ChatEvent objects, one per line (NDJSON).
 * The browser only ever talks to this route; the backend stays private.
 */
export async function POST(request: Request): Promise<Response> {
  let body: { message?: unknown; contextId?: unknown };
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Send JSON: { message, contextId? }" }, { status: 400 });
  }
  const message = typeof body.message === "string" ? body.message.trim() : "";
  const contextId = typeof body.contextId === "string" ? body.contextId : undefined;
  if (!message || message.length > MAX_MESSAGE_CHARS) {
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
        for await (const event of askCoordinator(message, contextId)) send(event);
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
