import { expect, test } from '@playwright/test';

import { API_BASE, skipWithoutMock } from './helpers';

/**
 * Gallery page: the seeded result loads a poster first and a player on click.
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
    await expect(tile.locator('video')).toHaveCount(0);
    await tile.getByRole('button', { name: /Preview result:/ }).click();

    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(
      dialog.locator('video').or(dialog.getByText('Preview unavailable')),
    ).toBeVisible();
    const video = dialog.locator('video');
    if (await video.isVisible()) {
      await expect(video).toHaveAttribute('controls', '');
    }
    await expect(dialog.getByRole('button', { name: 'Download video' })).toBeVisible();
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(page.locator('video')).toHaveCount(0);
  });
});
