import { expect, test } from '@playwright/test';

import { skipWithoutMock } from './helpers';

/**
 * GPU status page: the mock's fake GPUs render as cards with meters, and the
 * page is reachable from the primary sidebar.
 */
test.describe('gpus', () => {
  skipWithoutMock();

  test('lists the mock GPUs with utilization meters', async ({ page }) => {
    await page.goto('/gpus');

    await expect(
      page.getByRole('heading', { level: 1, name: 'GPUs' }),
    ).toBeVisible();

    await expect(page.getByText('GPU 0')).toBeVisible();
    await expect(page.getByText('GPU 1')).toBeVisible();
    expect(await page.getByRole('meter').count()).toBe(4);
  });

  test('is reachable from the sidebar', async ({ page }) => {
    await page.goto('/inference');
    await page.getByRole('link', { name: 'GPUs' }).click();
    await expect(page).toHaveURL(/\/gpus$/);
    await expect(page.getByText('GPU 0')).toBeVisible();
  });
});
