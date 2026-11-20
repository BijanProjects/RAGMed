import { askCoordinator } from "@/lib/coordinator";
import { SESSION_COOKIE, isValidSession, issueSession, readCookie, sessionCookie } from "@/lib/session";
import { clientIp, verifyTurnstile } from "@/lib/turnstile";
import type { ChatAction, ChatEvent } from "@/lib/types";

const MAX_MESSAGE_CHARS = 2000;
// The bot check is on when a Turnstile secret is configured (always in production).
const TURNSTILE_SECRET = process.env.TURNSTILE_SECRET_KEY;
const SESSION_SECRET = process.env.SESSION_SECRET;
const EMAIL = /^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$/;
const CODE = /^\d{6}$/;
const ISO_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?([+-]\d{2}:\d{2}|Z)$/;

/** Only the known buttons, with only their known fields, reach the backend. */
function parseAction(value: unknown): ChatAction | undefined | null {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const kind = typeof raw.clinician_type === "string" ? raw.clinician_type.slice(0, 60) : undefined;
  if (raw.type === "show_slots") return { type: "show_slots", clinician_type: kind };
  if (raw.type === "resend") return { type: "resend" };
  if (raw.type === "confirm") {
    const code = typeof raw.code === "string" ? raw.code.trim() : "";
    return CODE.test(code) ? { type: "confirm", code } : null;
  }
  if (raw.type === "book" && typeof raw.start === "string" && ISO_TIME.test(raw.start)) {
    const name = typeof raw.name === "string" ? raw.name.trim().slice(0, 80) : "";
    const email = typeof raw.email === "string" ? raw.email.trim() : "";
    if (!name || email.length > 254 || !EMAIL.test(email)) return null;
    return { type: "book", start: raw.start, clinician_type: kind, name, email };
  }
  return null;
}

/**
 * POST { message?, contextId?, action? } → a stream of ChatEvent objects, one per line (NDJSON).
 * The browser only ever talks to this route; the backend stays private.
 */
export async function POST(request: Request): Promise<Response> {
  let body: { message?: unknown; contextId?: unknown; action?: unknown; turnstileToken?: unknown };
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

  const ip = clientIp(request.headers);
  let setCookie: string | undefined;
  if (TURNSTILE_SECRET) {
    if (!SESSION_SECRET || SESSION_SECRET.length < 32) {
      return Response.json({ error: "The chat is not configured." }, { status: 500 });
    }
    const verified = isValidSession(readCookie(request.headers.get("cookie"), SESSION_COOKIE), SESSION_SECRET);
    if (!verified) {
      const token = typeof body.turnstileToken === "string" ? body.turnstileToken : "";
      if (!token || !(await verifyTurnstile(token, TURNSTILE_SECRET, ip))) {
        return Response.json(
          { error: "Please complete the quick check, then send again.", code: "verification_required" },
          { status: 403 },
        );
      }
      setCookie = sessionCookie(issueSession(SESSION_SECRET));
    }
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: ChatEvent) =>
        controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
      try {
        for await (const event of askCoordinator(message, contextId, action, ip)) send(event);
      } catch {
        send({ type: "error", text: "Something went wrong. Please try again." });
      } finally {
        controller.close();
      }
    },
  });

  const headers = new Headers({
    "Content-Type": "application/x-ndjson; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  if (setCookie) headers.set("Set-Cookie", setCookie);
  return new Response(stream, { headers });
}
