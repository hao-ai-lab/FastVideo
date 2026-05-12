import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for Dreamverse end-to-end tests.
 *
 * Tests assume the Dreamverse Python server is reachable at
 * BACKEND_HOST:BACKEND_PORT (default 127.0.0.1:8009) and the Next.js frontend
 * runs on port 5299. The webServer block boots `pnpm run dev` if no
 * server is already listening so tests work both locally and in CI.
 */
export default defineConfig({
  testDir: './e2e',
  // Tests share a backend, so run sequentially to avoid contention on
  // the single GPU pool slot during real-generation runs. Override per
  // test via test.parallel if a test is safe to run alongside others.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5299',
    headless: true,
    viewport: { width: 1280, height: 720 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: process.env.PLAYWRIGHT_SKIP_WEBSERVER
    ? undefined
    : {
        command: 'pnpm run dev',
        url: 'http://127.0.0.1:5299',
        reuseExistingServer: true,
        timeout: 120_000,
        env: {
          BACKEND_HOST: process.env.BACKEND_HOST || '127.0.0.1',
          BACKEND_PORT: process.env.BACKEND_PORT || '8009',
        },
      },
});
