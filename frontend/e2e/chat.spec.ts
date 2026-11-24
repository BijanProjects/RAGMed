import { expect, test, type Page, type Route } from "@playwright/test";

const SLOTS = [
  { start: "2026-11-23T09:00:00-03:30", end: "2026-11-23T09:30:00-03:30", physician: "Dr. Fred Nurk", clinician_type: "Neurology", visit_minutes: 30 },
  { start: "2026-11-23T10:30:00-03:30", end: "2026-11-23T11:00:00-03:30", physician: "Dr. Joe Public", clinician_type: "Neurology", visit_minutes: 30 },
  { start: "2026-11-24T14:00:00-03:30", end: "2026-11-24T14:30:00-03:30", physician: "Dr. Fred Nurk", clinician_type: "Neurology", visit_minutes: 30 },
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

/** Opens the chat and waits for the bot check (Cloudflare's test key always passes). */
async function openChat(page: Page) {
  await page.goto("/");
  await expect(page.locator('[data-check="ready"]')).toBeAttached({ timeout: 20_000 });
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
    const action = body.action as { type: string; start?: string; clinician_type?: string; code?: string } | undefined;
    if (action?.type === "book") {
      return [reply("verify_email", "We sent a 6-digit code to **alex@example.org**.", {
        verification: { email: "alex@example.org", start: action.start, clinician_type: "Neurology", minutes: 10 },
      })];
    }
    if (action?.type === "confirm" && action.code !== "482913") {
      return [reply("verify_email", "That code is not right. You have 4 tries left.", {
        verification: { email: "alex@example.org", start: SLOTS[1].start, clinician_type: "Neurology", minutes: 10 },
      })];
    }
    if (action?.type === "confirm") {
      return [reply("booked", "**Booked.**", {
        appointment: { reference: "RC-TEST42", start: SLOTS[1].start, end: SLOTS[1].end, minutes: 30,
          clinician_type: "Neurology", physician: "Dr. Joe Public", urgency: "routine", status: "booked" },
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

  await openChat(page);
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

  await page.getByRole("button", { name: /10:30.*Dr\. Joe Public/ }).click();
  const confirm = page.getByRole("form", { name: "Confirm booking" });
  await expect(confirm).toContainText("Monday, November 23, 10:30");
  const book = confirm.getByRole("button", { name: "Email me a code" });
  await expect(book).toBeDisabled(); // name and email first
  await confirm.getByLabel("Your name").fill("Alex Sample");
  await confirm.getByLabel("Email for the invitation").fill("alex@example");
  await expect(book).toBeDisabled(); // not a complete address
  await confirm.getByLabel("Email for the invitation").fill("alex@example.org");
  await book.click();

  // Nothing is booked until the emailed code is entered.
  await expect(page.getByText(/We sent a 6-digit code/)).toBeVisible();
  let codeBox = page.getByRole("form", { name: "Enter your code" });
  const confirmCode = codeBox.getByRole("button", { name: "Confirm booking" });
  await expect(confirmCode).toBeDisabled();
  await codeBox.getByLabel(/6-digit code/).fill("12a34"); // letters are dropped
  await expect(codeBox.getByLabel(/6-digit code/)).toHaveValue("1234");
  await codeBox.getByLabel(/6-digit code/).fill("111111");
  await confirmCode.click();
  await expect(page.getByText("That code is not right. You have 4 tries left.")).toBeVisible();

  codeBox = page.getByRole("form", { name: "Enter your code" }); // a fresh box under the new reply
  await expect(codeBox).toHaveCount(1);
  await codeBox.getByLabel(/6-digit code/).fill("482913");
  await codeBox.getByRole("button", { name: "Confirm booking" }).click();

  await expect(page.getByText("Your visit is booked")).toBeVisible();
  await expect(page.getByText("RC-TEST42")).toBeVisible();
  await expect(page.getByText(/invitation is on its way by email/)).toBeVisible();

  expect(requests[0].turnstileToken).toBeTruthy(); // the first message carries the check
  expect(requests.slice(1).every((r) => !r.turnstileToken)).toBe(true); // then the cookie does
  expect(requests.map((r) => r.action ?? r.message)).toEqual([
    "I've had headaches on one side for a month",
    "About a month",
    { type: "show_slots" },
    {
      type: "book", start: "2026-11-23T10:30:00-03:30", clinician_type: "Neurology",
      name: "Alex Sample", email: "alex@example.org",
    },
    { type: "confirm", code: "111111" },
    { type: "confirm", code: "482913" },
  ]);
  expect(requests.slice(1).every((r) => r.contextId === "ctx-1")).toBe(true);
});

test("choosing a general physician instead asks for their times", async ({ page }) => {
  const requests = await mockChat(page, (body) =>
    body.action
      ? [reply("slots", "Open times with General physician.", { slots: [], offer: { ...OFFER, clinician_type: "General physician" } })]
      : [reply("report", "## Draft pre-visit report", { urgency: "routine", offer: OFFER })],
  );

  await openChat(page);
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

  await openChat(page);
  await page.getByLabel("Your message").fill("crushing chest pain");
  await page.getByRole("button", { name: "Send" }).click();

  const card = page.locator('[data-kind="emergency"]');
  await expect(card).toContainText("This may be a medical emergency");
  await expect(card.getByText("Emergency", { exact: true })).toBeVisible();
});

test("progress is shown while the agents work, and errors are readable", async ({ page }) => {
  await mockChat(page, () => [{ type: "progress", text: "Searching the medical knowledge base" }, { type: "error", text: "The RAGMed service is not reachable right now." }]);

  await openChat(page);
  await page.getByLabel("Your message").fill("hello");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("The RAGMed service is not reachable right now.")).toBeVisible();
});

test("a new conversation starts fresh", async ({ page }) => {
  const requests = await mockChat(page, () => [reply("greeting", "Hi, I'm the intake assistant.")]);

  await openChat(page);
  await page.getByLabel("Your message").fill("hello");
  await page.getByLabel("Your message").press("Enter");
  await expect(page.getByText("Hi, I'm the intake assistant.")).toBeVisible();
  await page.getByRole("button", { name: "New conversation" }).click();
  await page.getByRole("button", { name: /What causes asthma/ }).click();

  await expect(page.getByText("Hi, I'm the intake assistant.")).toHaveCount(1);
  expect(requests[1].contextId).toBeUndefined();
});

test("a usage limit is shown as a readable message and the chat stays usable", async ({ page }) => {
  const LIMIT = "You have reached the message limit for now. Please try again in 14 minutes.";
  await mockChat(page, () => [{ type: "reply", text: LIMIT, data: null, contextId: "ctx-1", state: "rejected" }]);

  await openChat(page);
  const box = page.getByRole("textbox");
  await box.fill("my knee hurts");
  await box.press("Enter");

  await expect(page.getByText(LIMIT)).toBeVisible();
  await expect(page.locator('[data-kind="error"]')).toHaveCount(1);
  await expect(box).toBeEnabled();
});

test("after a booking error the time picker can be used again", async ({ page }) => {
  let attempts = 0;
  await mockChat(page, (body) => {
    const action = body.action as { type: string } | undefined;
    if (action?.type === "book") {
      attempts += 1;
      return attempts === 1
        ? [reply("no_answer", "That email address was refused. Please check it.")]
        : [reply("verify_email", "We sent a 6-digit code.", {
            verification: { email: "alex@example.org", start: SLOTS[0].start, clinician_type: "Neurology", minutes: 10 },
          })];
    }
    if (action?.type === "show_slots") return [reply("slots", "Here are open times.", { slots: SLOTS, offer: OFFER })];
    return [reply("report", "## Draft pre-visit report", { urgency: "routine", offer: OFFER })];
  });

  await openChat(page);
  await page.getByRole("button", { name: /headaches on one side/ }).click();
  await page.getByRole("button", { name: /See available times/ }).click();
  const fill = async () => {
    await page.getByRole("button", { name: /9:00.*Dr\. Fred Nurk/ }).click();
    const form = page.getByRole("form", { name: "Confirm booking" });
    await form.getByLabel("Your name").fill("Alex Sample");
    await form.getByLabel("Email for the invitation").fill("alex@example.org");
    await form.getByRole("button", { name: "Email me a code" }).click();
  };
  await fill();
  await expect(page.getByText("That email address was refused. Please check it.")).toBeVisible();

  await fill();
  await expect(page.getByRole("form", { name: "Enter your code" })).toBeVisible();
});

test("each step names its agent, and the team's work stays available after the reply", async ({ page }) => {
  await mockChat(page, () => [
    { type: "progress", text: "Reading your message", agent: "Care Coordinator" },
    { type: "progress", text: "Searched for “one-sided headache”: 4 passages", agent: "Knowledge base" },
    { type: "progress", text: "Weighed the evidence: considering migraine", agent: "Diagnosis agent" },
    reply("question", "Does light or noise make it worse?"),
  ]);

  await openChat(page);
  await page.getByLabel("Your message").fill("headaches on one side");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("Does light or noise make it worse?")).toBeVisible();

  await page.getByText("How the care team worked on this · 3 steps").click();
  const log = page.getByRole("list", { name: "Care team activity" });
  await expect(log).toContainText("Knowledge base · Searched for “one-sided headache”: 4 passages");
  await expect(log).toContainText("Diagnosis agent · Weighed the evidence: considering migraine");
});

test("a citation opens the passage it came from, with a link to read more", async ({ page }) => {
  const source = {
    number: 1,
    question: "What are the symptoms of Asthma?",
    answer_id: "a1",
    excerpt: "Common asthma symptoms include coughing, wheezing and chest tightness.",
  };
  await mockChat(page, () => [
    reply("answer", "Wheezing is a common sign of asthma [1].\n\n**Sources** (MedQuAD):\n1. What are the symptoms of Asthma?", {
      sources: [source],
    }),
  ]);

  await openChat(page);
  await page.getByLabel("Your message").fill("Why do I wheeze?");
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByRole("link", { name: "Source 1" }).click();

  const card = page.locator("details", { hasText: "What are the symptoms of Asthma?" });
  await expect(card.getByText("Common asthma symptoms include")).toBeVisible();
  const more = card.getByRole("link", { name: /Read more on MedlinePlus/ });
  await expect(more).toHaveAttribute("href", /medlineplus.*query=Asthma$/);
  await expect(more).toHaveAttribute("target", "_blank");
  await expect(page.getByText("1. What are the symptoms of Asthma?")).toHaveCount(0); // shown as a card, not a list
});

test("a multiple-choice question sends every option ticked", async ({ page }) => {
  const requests = await mockChat(page, (body) =>
    body.message === "headaches"
      ? [reply("question", "Do you have any of these as well?", {
          quick_replies: ["Nausea", "Light hurts", "Blurred vision"], multi_select: true,
        })]
      : [reply("question", "Thanks. How old are you?")],
  );

  await openChat(page);
  await page.getByLabel("Your message").fill("headaches");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  const choices = page.getByRole("group", { name: "Choose all that apply" });
  await choices.getByRole("checkbox", { name: "Nausea" }).click();
  await choices.getByRole("checkbox", { name: /Blurred vision/ }).click();
  await expect(choices.getByRole("checkbox", { name: /Nausea/ })).toHaveAttribute("aria-checked", "true");
  await choices.getByRole("button", { name: "Send 2 selected" }).click();

  await expect(page.getByText("Thanks. How old are you?")).toBeVisible();
  expect(requests[1].message).toBe("Nausea, Blurred vision");
});

test("'None of these' clears the other choices", async ({ page }) => {
  const requests = await mockChat(page, (body) =>
    body.message === "cough"
      ? [reply("question", "Any of these?", { quick_replies: ["Fever", "Wheezing"], multi_select: true })]
      : [reply("question", "How long have you had the cough?")],
  );

  await openChat(page);
  await page.getByLabel("Your message").fill("cough");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  const choices = page.getByRole("group", { name: "Choose all that apply" });
  await choices.getByRole("checkbox", { name: "Fever" }).click();
  await choices.getByRole("checkbox", { name: "None of these" }).click();
  await choices.getByRole("button", { name: "Send: none of these" }).click();

  await expect(page.getByText("How long have you had the cough?")).toBeVisible();
  expect(requests[1].message).toBe("None of these");
});

test("after an emergency the patient can say it has passed, and a general answer has no self-care badge", async ({ page }) => {
  const requests = await mockChat(page, (body) =>
    body.message === "my face is drooping"
      ? [reply("emergency", "**This may be a medical emergency.**", {
          urgency: "emergency", quick_replies: ["I've called for help", "It's not happening right now"],
        })]
      : body.message === "What causes asthma?"
        ? [reply("answer", "Genes and environment [1].", { urgency: "self_care" })]
        : [reply("question", "Good. Let's carry on.")],
  );

  await openChat(page);
  await page.getByLabel("Your message").fill("What causes asthma?");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByText("Genes and environment")).toBeVisible();
  await expect(page.getByText("Self-care", { exact: true })).toHaveCount(0);

  await page.getByLabel("Your message").fill("my face is drooping");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.getByRole("button", { name: "It's not happening right now" }).click();
  await expect(page.getByText("Good. Let's carry on.")).toBeVisible();
  expect(requests.at(-1)?.message).toBe("It's not happening right now");
});
