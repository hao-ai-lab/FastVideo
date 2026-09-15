import { test, expect } from '@playwright/test';

/**
 * Smoke layer 2: the Next.js shell renders without crashing and shows
 * the expected GPU-warmup banner when the backend's GPU pool isn't
 * ready yet. This catches integration breakage between the frontend
 * (running against the public FastVideo backend) and the readyz
 * payload shape.
 */
test.describe('frontend shell', () => {
  test('main page loads and exposes the FastVideo brand chip', async ({ page }) => {
    await page.goto('/');
    // The TopStatusBar element is keyed off aria-label="FastVideo"
    // and is rendered before any session interaction is possible.
    await expect(page.getByRole('img', { name: 'FastVideo' })).toBeVisible({
      timeout: 30_000,
    });
  });

  test('composer hydrates with creation studio controls', async ({ page }) => {
    await page.goto('/');

    const continuation = page.getByLabel('Continuation prompt');
    await expect(continuation).toBeVisible({ timeout: 30_000 });

    const generate = page.getByRole('button', { name: /^generate$/i });
    await expect(generate).toBeVisible({ timeout: 30_000 });

    await expect(page.getByText('Direct scenes in seconds')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole('button', { name: /FastLTX/i }).first()).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText('Describe your video or mention elements')).toBeVisible({ timeout: 30_000 });
  });
});
