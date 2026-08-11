import { expect, test } from '@playwright/test';

import { API_BASE, skipWithoutMock } from './helpers';

/**
 * Gallery page: the seeded completed inference job surfaces as a media tile
 * with playback controls or an explicit media-error fallback.
 */
test.describe('gallery', () => {
  skipWithoutMock();

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

    const tile = page.locator('article').filter({ hasText: completed!.prompt });
    await expect(tile).toBeVisible();
    await expect(
      tile.locator('video').or(tile.getByText('Preview unavailable')),
    ).toBeVisible();

    const video = tile.locator('video');
    if (await video.isVisible()) {
      await expect(video).toHaveAttribute('controls', '');
    }
  });
});
