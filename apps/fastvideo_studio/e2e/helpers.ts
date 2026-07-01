import type { APIRequestContext, Page } from '@playwright/test';

/**
 * Base URL of the mock API. The browser reaches it via NEXT_PUBLIC_API_BASE_URL
 * (injected into `npm run dev` by playwright.config.ts). The Playwright
 * `request` fixture uses the app's baseURL (port 3000), so specs that need to
 * read mock state hit this URL directly instead.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8189/api';

export const MOCK_SKIP_MESSAGE =
  'Mock backend not reachable at /api/models ' +
  '(start fastvideo_studio.mock_server on port 8189).';

/** True when the mock server's /api/models responds with a JSON array. */
export async function mockIsUp(request: APIRequestContext): Promise<boolean> {
  try {
    const res = await request.get(`${API_BASE}/models`);
    if (!res.ok()) return false;
    const body = await res.json();
    return Array.isArray(body);
  } catch {
    return false;
  }
}

/**
 * Persist the mock API base URL into the client's stored options before the app
 * boots. `getApiBaseUrl()` reads `fastvideo-default-options.apiServerBaseUrl`
 * from localStorage first; without a value there it falls back to a relative
 * URL once the settings sync writes the key, so any API call made after that
 * sync (e.g. the Create Job modal's model fetch) would 404. Seeding it — the
 * same thing the Settings page's "API Server Base URL" field does — keeps every
 * request pointed at the mock. Must be called before `page.goto`.
 */
export async function seedApiBaseUrl(page: Page): Promise<void> {
  await page.addInitScript((base: string) => {
    try {
      const key = 'fastvideo-default-options';
      const raw = window.localStorage.getItem(key);
      const parsed = raw ? JSON.parse(raw) : {};
      parsed.apiServerBaseUrl = base;
      window.localStorage.setItem(key, JSON.stringify(parsed));
    } catch {
      // Ignore storage errors; the env fallback still points at the mock.
    }
  }, API_BASE);
}
