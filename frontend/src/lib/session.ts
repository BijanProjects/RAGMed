import "server-only";

import { createHmac, timingSafeEqual } from "node:crypto";

/** A signed, HTTP-only cookie marking a browser that passed the bot check. */
export const SESSION_COOKIE = "ragmed_verified";
export const SESSION_HOURS = 12;

function sign(payload: string, secret: string): string {
  return createHmac("sha256", secret).update(payload).digest("base64url");
}

export function issueSession(secret: string, now = Date.now()): string {
  const payload = String(now + SESSION_HOURS * 3600_000);
  return `${payload}.${sign(payload, secret)}`;
}

export function isValidSession(value: string | undefined, secret: string, now = Date.now()): boolean {
  if (!value) return false;
  const [payload, signature] = value.split(".");
  if (!payload || !signature) return false;
  const expected = Buffer.from(sign(payload, secret));
  const given = Buffer.from(signature);
  if (expected.length !== given.length || !timingSafeEqual(expected, given)) return false;
  return Number(payload) > now;
}

export function sessionCookie(value: string): string {
  const secure = process.env.NODE_ENV === "production" ? "; Secure" : "";
  return `${SESSION_COOKIE}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_HOURS * 3600}${secure}`;
}

export function readCookie(header: string | null, name: string): string | undefined {
  for (const part of (header ?? "").split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return undefined;
}
