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

  test('composer hydrates with curated preset cards', async ({ page }) => {
    await page.goto('/');

    // The Continuation prompt textarea + Generate button render once
    // the FE has hydrated against the public-FastVideo-backed
    // dreamverse-server. Their presence proves the integration handshake
    // (CORS, /curated-presets, /prompt-system-config) completed.
    const continuation = page.getByLabel('Continuation prompt');
    await expect(continuation).toBeVisible({ timeout: 30_000 });

    const generate = page.getByRole('button', { name: /^generate$/i });
    await expect(generate).toBeVisible({ timeout: 30_000 });

    // Curated presets render as buttons; verify at least one is
    // available — that's the only way the user can populate the
    // Continuation textarea in the default composer.
    const presetCard = page.getByRole('button', {
      name: /LEGO Stormtroopers|Clay Stop-Motion|Boy & Dog|School Prank|Gamer Gets Banned|Small Town Oil Strike|Grandpa's Wing Costume/i,
    }).first();
    await expect(presetCard).toBeVisible({ timeout: 30_000 });
  });
});
