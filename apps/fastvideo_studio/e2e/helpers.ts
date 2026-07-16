import { test, type APIRequestContext } from '@playwright/test';

/**
 * Port and base URL of the mock API — the single source of truth shared by
 * playwright.config.ts (which boots the mock and points `npm run dev` at it
 * via NEXT_PUBLIC_API_BASE_URL) and the specs (which probe/read mock state
 * directly through the Playwright `request` fixture).
 *
 * The default port is deliberately NOT the real server's default (8189): the
 * e2e specs create and (if autostart is on) launch jobs, so running them
 * against a real backend would mutate the developer's database and fire real
 * GPU runs. API_BASE is derived solely from MOCK_API_PORT (it does not inherit
 * NEXT_PUBLIC_API_BASE_URL) so the specs always talk to the mock.
 */
export const MOCK_API_PORT = process.env.MOCK_API_PORT || '8190';

export const API_BASE = `http://127.0.0.1:${MOCK_API_PORT}/api`;

export const MOCK_SKIP_MESSAGE =
  `Mock backend not reachable at ${API_BASE}/__mock__ ` +
  `(start fastvideo_studio.mock_server on port ${MOCK_API_PORT}).`;

/**
 * True only when the mock server answers on the configured port. Probes the
 * mock-only /api/__mock__ sentinel (which the real server does not serve) so
 * the suite refuses to run its mutating specs against a real backend.
 */
export async function mockIsUp(request: APIRequestContext): Promise<boolean> {
  try {
    const res = await request.get(`${API_BASE}/__mock__`);
    if (!res.ok()) return false;
    const body = await res.json();
    return body?.mock === true;
  } catch {
    return false;
  }
}

/** Self-skip every test in the enclosing describe when the mock is down. */
export function skipWithoutMock(): void {
  test.beforeEach(async ({ request }) => {
    test.skip(!(await mockIsUp(request)), MOCK_SKIP_MESSAGE);
  });
}
