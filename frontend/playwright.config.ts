import { defineConfig, devices } from "@playwright/test";

/** Browser tests of the web app. The backend is mocked per test, so no services are needed. */
export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: "http://localhost:3100", trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: "npm run dev -- --port 3100",
    url: "http://localhost:3100",
    reuseExistingServer: !process.env.CI,
    env: {
      NEXT_TELEMETRY_DISABLED: "1",
      NEXT_PUBLIC_CLINIC_TIME_ZONE: "America/St_Johns",
      // Cloudflare's published Turnstile test keys: they always pass.
      NEXT_PUBLIC_TURNSTILE_SITE_KEY: "1x00000000000000000000AA",
      TURNSTILE_SECRET_KEY: "1x0000000000000000000000000000000AA",
      SESSION_SECRET: "test-only-session-secret-0123456789abcdef",
      COORDINATOR_URL: "http://127.0.0.1:9", // nothing listens: these tests never reach it
    },
  },
});
