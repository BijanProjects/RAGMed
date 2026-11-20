import { expect, test, type Page, type Route } from "@playwright/test";

const SLOTS = [
  { start: "2026-11-23T09:00:00-03:30", end: "2026-11-23T09:30:00-03:30", physician: "Dr. Isabel Cohen", clinician_type: "Neurology", visit_minutes: 30 },
  { start: "2026-11-23T10:30:00-03:30", end: "2026-11-23T11:00:00-03:30", physician: "Dr. Daniel Osei", clinician_type: "Neurology", visit_minutes: 30 },
  { start: "2026-11-24T14:00:00-03:30", end: "2026-11-24T14:30:00-03:30", physician: "Dr. Isabel Cohen", clinician_type: "Neurology", visit_minutes: 30 },
];
const OFFER = { clinician_type: "Neurology", visit_minutes: 30, urgency: "routine", alternative: "General physician" };

function reply(kind: string, text: string, extra: Record<string, unknown> = {}) {
  return {
    type: "reply",
    text,
    contextId: "ctx-1",
    state: "completed",
    data: {
      kind, text, urgency: null, recommend_visit: false, possible_conditions: [], sources: [],
      red_flag: null, quick_replies: [], report: null, offer: null, slots: [], appointment: null,
      ...extra,
    },
  };
}

/** Answers /api/chat from a script and records every request the page sends. */
async function mockChat(page: Page, script: (body: Record<string, unknown>) => object[]) {
  const requests: Record<string, unknown>[] = [];
  await page.route("**/api/chat", async (route: Route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    requests.push(body);
    const lines = script(body).map((event) => JSON.stringify(event)).join("\n") + "\n";
    await route.fulfill({ status: 200, contentType: "application/x-ndjson", body: lines });
  });
  return requests;
}

test("the whole intake: questions, report, times, confirmation, booking", async ({ page }) => {
  const requests = await mockChat(page, (body) => {
    const action = body.action as { type: string; start?: string; clinician_type?: string } | undefined;
    if (action?.type === "book") {
      return [reply("booked", "**Booked.**", {
        appointment: { reference: "RC-TEST42", start: action.start, end: action.start, minutes: 30,
          clinician_type: "Neurology", physician: "Dr. Daniel Osei", urgency: "routine", status: "booked" },
      })];
    }
    if (action?.type === "show_slots") {
      return [reply("slots", "Here are open times for a 30-minute visit with Neurology.", { slots: SLOTS, offer: OFFER })];
    }
    if (body.message === "About a month") {
      return [
        { type: "progress", text: "Preparing your draft report" },
        reply("report", "## Draft pre-visit report\n\n_This is not a diagnosis._\n\n**Possibilities to discuss with your clinician:** migraine", {
          urgency: "routine", offer: OFFER,
        }),
      ];
    }
    return [reply("question", "How long have you had these headaches?", { quick_replies: ["A day", "About a month"] })];
  });

  await page.goto("/");
  await page.getByRole("button", { name: /headaches on one side/ }).click();
  await expect(page.getByText("How long have you had these headaches?")).toBeVisible();

  await page.getByRole("group", { name: "Suggested answers" }).getByRole("button", { name: "About a month" }).click();
  await expect(page.getByRole("heading", { name: "Draft pre-visit report" })).toBeVisible();
  await expect(page.getByText("Routine", { exact: true })).toBeVisible();
  // Earlier buttons are no longer active once the conversation has moved on.
  await expect(page.getByRole("button", { name: "A day" })).toBeDisabled();

  await page.getByRole("button", { name: /See available times · 30 min with Neurology/ }).click();
  await expect(page.getByText("Monday, November 23")).toBeVisible();
  await expect(page.getByText("Tuesday, November 24")).toBeVisible();

  await page.getByRole("button", { name: /10:30.*Dr\. Daniel Osei/ }).click();
  const confirm = page.getByRole("dialog", { name: "Confirm booking" });
  await expect(confirm).toContainText("Monday, November 23, 10:30");
  await confirm.getByRole("button", { name: "Confirm booking" }).click();

  await expect(page.getByText("Your visit is booked")).toBeVisible();
  await expect(page.getByText("RC-TEST42")).toBeVisible();
  await expect(page.getByText(/draft pre-visit report has been sent to Dr\. Daniel Osei/)).toBeVisible();

  expect(requests.map((r) => r.action ?? r.message)).toEqual([
    "I've had headaches on one side for a month",
    "About a month",
    { type: "show_slots" },
    { type: "book", start: "2026-11-23T10:30:00-03:30", clinician_type: "Neurology" },
  ]);
  expect(requests.slice(1).every((r) => r.contextId === "ctx-1")).toBe(true);
});

test("choosing a general physician instead asks for their times", async ({ page }) => {
  const requests = await mockChat(page, (body) =>
    body.action
      ? [reply("slots", "Open times with General physician.", { slots: [], offer: { ...OFFER, clinician_type: "General physician" } })]
      : [reply("report", "## Draft pre-visit report", { urgency: "routine", offer: OFFER })],
  );

  await page.goto("/");
  await page.getByLabel("Your message").fill("my head hurts");
  await page.getByLabel("Your message").press("Enter");
  await page.getByRole("button", { name: "A general physician instead" }).click();

  await expect(page.getByText("Open times with General physician.")).toBeVisible();
  expect(requests[1].action).toEqual({ type: "show_slots", clinician_type: "General physician" });
});

test("an emergency is shown as a red card with the emergency badge", async ({ page }) => {
  await mockChat(page, () => [
    reply("emergency", "**This may be a medical emergency. Call your local emergency number now**", {
      urgency: "emergency", red_flag: "chest_pain",
    }),
  ]);

  await page.goto("/");
  await page.getByLabel("Your message").fill("crushing chest pain");
  await page.getByRole("button", { name: "Send" }).click();

  const card = page.locator('[data-kind="emergency"]');
  await expect(card).toContainText("This may be a medical emergency");
  await expect(card.getByText("Emergency", { exact: true })).toBeVisible();
});

test("progress is shown while the agents work, and errors are readable", async ({ page }) => {
  await mockChat(page, () => [{ type: "progress", text: "Searching the medical knowledge base" }, { type: "error", text: "The RAGMed service is not reachable right now." }]);

  await page.goto("/");
  await page.getByLabel("Your message").fill("hello");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("The RAGMed service is not reachable right now.")).toBeVisible();
});

test("a new conversation starts fresh", async ({ page }) => {
  const requests = await mockChat(page, () => [reply("greeting", "Hi, I'm the intake assistant.")]);

  await page.goto("/");
  await page.getByLabel("Your message").fill("hello");
  await page.getByLabel("Your message").press("Enter");
  await expect(page.getByText("Hi, I'm the intake assistant.")).toBeVisible();
  await page.getByRole("button", { name: "New conversation" }).click();
  await page.getByRole("button", { name: /What causes asthma/ }).click();

  await expect(page.getByText("Hi, I'm the intake assistant.")).toHaveCount(1);
  expect(requests[1].contextId).toBeUndefined();
});
