import { expect, test } from '@playwright/test';

import { API_BASE, MOCK_SKIP_MESSAGE, mockIsUp, seedApiBaseUrl } from './helpers';

/**
 * Gallery page: the seeded completed inference job surfaces as a media tile
 * (an <article> wrapping a <video>) captioned with its prompt.
 */
test.describe('gallery', () => {
  test.beforeEach(async ({ page, request }) => {
    test.skip(!(await mockIsUp(request)), MOCK_SKIP_MESSAGE);
    await seedApiBaseUrl(page);
  });

  test('shows a media tile for the seeded completed job', async ({
    page,
    request,
  }) => {
    const res = await request.get(`${API_BASE}/jobs?job_type=inference`);
    const jobs = (await res.json()) as Array<{
      status: string;
      output_path: string | null;
      prompt: string;
    }>;
    const completed = jobs.find(
      (j) => j.status === 'completed' && j.output_path,
    );
    expect(completed, 'mock should seed a completed inference job').toBeTruthy();

    await page.goto('/gallery');

    await expect(
      page.getByRole('heading', { level: 1, name: 'Gallery' }),
    ).toBeVisible();

    // The completed job renders as an <article> containing a <video> tile.
    const tile = page
      .locator('article')
      .filter({ has: page.locator('video') });
    await expect(tile.first()).toBeVisible();

    await expect(page.getByText(completed!.prompt)).toBeVisible();
  });
});
