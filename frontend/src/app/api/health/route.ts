/**
 * GET /api/health → 200 {"status":"ok"} when the web app and the backend services it needs are
 * up, 503 {"status":"degraded"} otherwise. Used by the uptime monitor; it reveals nothing else.
 */
const COORDINATOR_URL = process.env.COORDINATOR_URL ?? "http://127.0.0.1:8000";
const SCHEDULING_URL = process.env.SCHEDULING_URL ?? "http://127.0.0.1:9002";

async function healthy(url: string): Promise<boolean> {
  try {
    const response = await fetch(`${url}/healthz`, { signal: AbortSignal.timeout(5000), cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export async function GET(): Promise<Response> {
  const checks = await Promise.all([healthy(COORDINATOR_URL), healthy(SCHEDULING_URL)]);
  const ok = checks.every(Boolean);
  return Response.json(
    { status: ok ? "ok" : "degraded" },
    { status: ok ? 200 : 503, headers: { "Cache-Control": "no-store" } },
  );
}
