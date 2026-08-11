import { defineConfig, devices } from '@playwright/test';

import { API_BASE, MOCK_API_PORT } from './e2e/helpers';

/**
 * Playwright config for FastVideo Studio end-to-end tests.
 *
 * The Next.js frontend runs on port 3000 and talks to the in-memory mock API
 * (fastvideo_studio.mock_server) on port 8189. The webServer block boots both
 * the mock backend and `npm run dev` (pointed at the mock via
 * NEXT_PUBLIC_API_BASE_URL) if nothing is already listening, so the suite works
 * both locally and in CI. Set PLAYWRIGHT_SKIP_WEBSERVER to reuse externally
 * managed servers.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:3000',
    headless: true,
    viewport: { width: 1280, height: 720 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: process.env.PLAYWRIGHT_SKIP_WEBSERVER
    ? undefined
    : [
        {
          command: `PYTHONPATH=.. python -m fastvideo_studio.mock_server --port ${MOCK_API_PORT}`,
          // Probe the mock-only sentinel, not /api/models (which a real server
          // also serves), so a real backend on this port is never mistaken for
          // the mock. Always boot a fresh mock in CI.
          url: `http://127.0.0.1:${MOCK_API_PORT}/api/__mock__`,
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
        {
          command: 'npm run dev',
          url: 'http://127.0.0.1:3000',
          reuseExistingServer: true,
          timeout: 120_000,
          env: {
            NEXT_PUBLIC_API_BASE_URL: API_BASE,
          },
        },
      ],
});
