import { timingSafeEqual } from "node:crypto";

// Server-side only: the browser sends an access code, never the backend token.
const SCHEDULING_URL = process.env.SCHEDULING_URL ?? "http://127.0.0.1:9002";
const CLINICIAN_TOKEN = process.env.CLINICIAN_TOKEN;
const ACCESS_CODE = process.env.CLINICIAN_ACCESS_CODE;

function sameCode(given: string, expected: string): boolean {
  const a = Buffer.from(given);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

/** POST { code } → the clinic's upcoming visits with their draft reports, most urgent first. */
export async function POST(request: Request): Promise<Response> {
  if (!ACCESS_CODE || !CLINICIAN_TOKEN) {
    return Response.json({ error: "The clinician view is not configured." }, { status: 503 });
  }
  const body = await request.json().catch(() => ({}));
  const code = typeof body.code === "string" ? body.code : "";
  if (!sameCode(code, ACCESS_CODE)) {
    return Response.json({ error: "Wrong access code." }, { status: 401 });
  }
  try {
    const response = await fetch(`${SCHEDULING_URL}/clinician/appointments`, {
      headers: { Authorization: `Bearer ${CLINICIAN_TOKEN}` },
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) {
      return Response.json({ error: `The clinic service returned ${response.status}.` }, { status: 502 });
    }
    return Response.json(await response.json(), { headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json({ error: "The clinic service is not reachable." }, { status: 502 });
  }
}
