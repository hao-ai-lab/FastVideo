import { expect, test } from '@playwright/test';

import { skipWithoutMock } from './helpers';

/**
 * Warm-model slot: load a model through the mock (which flips
 * loading -> ready after ~2s, surfaced by the panel's 5s poll), then unload it.
 */
test.describe('generators', () => {
  skipWithoutMock();

  test('loads and unloads the resident model', async ({ page }) => {
    await page.goto('/inference');

    const panel = page.getByRole('region', { name: 'Warm models' });
    await expect(panel).toBeVisible();
    await expect(panel.getByText('No model loaded')).toBeVisible();

    await panel
      .getByLabel('Model to load')
      .selectOption('Wan-AI/Wan2.1-T2V-1.3B-Diffusers');
    await panel.getByRole('button', { name: 'Load model' }).click();

    await expect(panel.getByText('Wan2.1 T2V 1.3B Diffusers')).toBeVisible();
    await expect(panel.getByText('ready')).toBeVisible();

    await panel.getByRole('button', { name: 'Unload' }).click();
    await expect(panel.getByText('No model loaded')).toBeVisible();
  });

  test('engine console streams output while open', async ({ page }) => {
    await page.goto('/inference');

    const engineConsole = page.getByRole('region', { name: 'Engine output' });
    await engineConsole.getByRole('button', { name: 'Engine output' }).click();

    await expect(engineConsole.getByText(/\[engine\]/).first()).toBeVisible();
  });
});
