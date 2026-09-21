import { expect, test } from '@playwright/test';

import { API_BASE, skipWithoutMock } from './helpers';

/**
 * Gallery page: native video controls display inline with a lightweight poster.
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
    const video = tile.locator('video');
    await expect(video).toBeVisible();
    await expect(video).toHaveAttribute('controls', '');
    await expect(video).toHaveAttribute('preload', 'none');
    await expect(tile.getByRole('button', { name: 'Download video' })).toBeVisible();
    await expect(page.getByRole('dialog')).toHaveCount(0);
    await expect(page.getByText('Watch video', { exact: true })).toHaveCount(0);
  });
});
