import { expect, test } from '@playwright/test';

import { skipWithoutMock } from './helpers';

/**
 * Warm models panel: preload a model through the mock (which flips
 * loading -> ready after ~2s, surfaced by the panel's 5s poll), then unload it.
 */
test.describe('generators', () => {
  skipWithoutMock();

  test('preloads and unloads a model', async ({ page }) => {
    await page.goto('/inference');

    const panel = page.getByRole('region', { name: 'Warm models' });
    await expect(panel).toBeVisible();

    await panel
      .getByLabel('Model to preload')
      .selectOption('Wan-AI/Wan2.1-T2V-1.3B-Diffusers');
    await panel.getByRole('button', { name: 'Preload model' }).click();

    const row = panel
      .locator('li')
      .filter({ hasText: 'Wan2.1 T2V 1.3B Diffusers' });
    await expect(row).toBeVisible();
    await expect(row.getByText('ready')).toBeVisible();

    await row.getByRole('button', { name: 'Unload' }).click();
    await expect(row).toBeHidden();
  });
});
