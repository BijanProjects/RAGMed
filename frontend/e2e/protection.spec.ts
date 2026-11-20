import { expect, test } from "@playwright/test";

// These call the real /api/chat route. Cloudflare's test secret accepts its dummy token.
const DUMMY_TOKEN = "XXXX.DUMMY.TOKEN.XXXX";

test("a first message without the bot check is refused", async ({ request }) => {
  const response = await request.post("/api/chat", { data: { message: "hello" } });

  expect(response.status()).toBe(403);
  expect((await response.json()).code).toBe("verification_required");
});

test("passing the check sets a signed session cookie that later messages use", async ({ playwright, baseURL }) => {
  const api = await playwright.request.newContext({ baseURL });

  const first = await api.post("/api/chat", { data: { message: "hello", turnstileToken: DUMMY_TOKEN } });
  const cookie = first.headers()["set-cookie"] ?? "";
  const second = await api.post("/api/chat", { data: { message: "hello again" } });

  expect(first.status()).toBe(200);
  expect(cookie).toContain("ragmed_verified=");
  expect(cookie).toContain("HttpOnly");
  expect(second.status()).toBe(200); // no token needed: the cookie came back
  await api.dispose();
});

test("a forged session cookie is refused", async ({ request }) => {
  const response = await request.post("/api/chat", {
    data: { message: "hello" },
    headers: { Cookie: "ragmed_verified=99999999999999.forged-signature" },
  });

  expect(response.status()).toBe(403);
});

test("unknown actions and bad booking emails are refused before the check", async ({ request }) => {
  const unknown = await request.post("/api/chat", { data: { action: { type: "delete_everything" } } });
  const badEmail = await request.post("/api/chat", {
    data: { action: { type: "book", start: "2026-11-23T09:00:00-03:30", name: "A", email: "nope" } },
  });

  expect(unknown.status()).toBe(400);
  expect(badEmail.status()).toBe(400);
});
