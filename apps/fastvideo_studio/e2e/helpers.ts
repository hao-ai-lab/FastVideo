import { test, type APIRequestContext } from '@playwright/test';

/**
 * Port and base URL of the mock API — the single source of truth shared by
 * playwright.config.ts (which boots the mock and points `npm run dev` at it
 * via NEXT_PUBLIC_API_BASE_URL) and the specs (which probe/read mock state
 * directly through the Playwright `request` fixture).
 */
export const MOCK_API_PORT = process.env.MOCK_API_PORT || '8189';

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  `http://127.0.0.1:${MOCK_API_PORT}/api`;

export const MOCK_SKIP_MESSAGE =
  `Mock backend not reachable at ${API_BASE}/models ` +
  `(start fastvideo_studio.mock_server on port ${MOCK_API_PORT}).`;

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

/** Self-skip every test in the enclosing describe when the mock is down. */
export function skipWithoutMock(): void {
  test.beforeEach(async ({ request }) => {
    test.skip(!(await mockIsUp(request)), MOCK_SKIP_MESSAGE);
  });
}
