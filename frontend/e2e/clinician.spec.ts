import { expect, test } from "@playwright/test";

const REPORT = {
  title: "Draft pre-visit report", notice: "This is not a diagnosis.", summary: "",
  case: { "Main problem": "Headaches on one side", "Severity (0-10)": "7" },
  assessment: "Consistent with migraine [1].", possible_conditions: ["migraine"], urgency: "urgent",
  clinician_type: "Neurology", visit_minutes: 30, visit_reason: "", sufficient_evidence: true,
  sources: [{ number: 1, question: "What is (are) Migraine?", answer_id: "m1" }], created_at: "",
};
const VISITS = [
  { reference: "RC-URGENT", start: "2026-11-23T09:00:00-03:30", end: "", minutes: 30, clinician_type: "Neurology",
    physician: "Dr. Fred Nurk", urgency: "urgent", status: "booked", booking_uid: "u1", created_at: "", report: REPORT },
  { reference: "RC-ROUTINE", start: "2026-11-24T10:00:00-03:30", end: "", minutes: 15, clinician_type: "General physician",
    physician: "Dr. John Doe", urgency: "routine", status: "booked", booking_uid: "u2", created_at: "",
    report: { ...REPORT, urgency: "routine", case: { "Main problem": "Renew a prescription" } } },
];

test("the clinician view needs the access code and lists urgent visits first", async ({ page }) => {
  await page.route("**/api/clinician", async (route) => {
    const { code } = route.request().postDataJSON() as { code: string };
    if (code !== "right-code") {
      await route.fulfill({ status: 401, json: { error: "Wrong access code." } });
    } else {
      await route.fulfill({ status: 200, json: VISITS });
    }
  });

  await page.goto("/clinician");
  await page.getByLabel("Access code").fill("wrong");
  await page.getByRole("button", { name: "Open", exact: true }).click();
  await expect(page.getByText("Wrong access code.")).toBeVisible();

  await page.getByLabel("Access code").fill("right-code");
  await page.getByRole("button", { name: "Open", exact: true }).click();

  await expect(page.getByText("2 upcoming visits")).toBeVisible();
  const cards = page.locator("article");
  await expect(cards.first()).toHaveAttribute("data-urgency", "urgent");
  await expect(cards.first()).toContainText("RC-URGENT");
  await expect(cards.first()).toContainText("Headaches on one side");
  await expect(cards.first()).toContainText("Consistent with migraine");
  await expect(cards.nth(1)).toContainText("Renew a prescription");
});

test("the patient chat links to the clinician view", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Clinician view" }).click();
  await expect(page.getByRole("heading", { name: /Clinician view/ })).toBeVisible();
});
