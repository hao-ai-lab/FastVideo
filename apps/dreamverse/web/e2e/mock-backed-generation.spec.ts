import { test, expect, type WebSocket as PWWebSocket } from '@playwright/test';

test.describe('mock-backed generation smoke', () => {
  test('accepts a custom prompt and enters the streaming state', async ({ page, request }) => {
    const health = await request.get('/healthz');
    const body = health.ok() ? await health.json() : {};
    test.skip(
      body.service !== 'ltx2-streaming-mock-server',
      'Mock-backed smoke requires dreamverse.mock_server on BACKEND_PORT (default 8009).',
    );

    const wsEventTypes: string[] = [];

    page.on('websocket', (ws: PWWebSocket) => {
      ws.on('framereceived', ({ payload }) => {
        if (typeof payload !== 'string') return;
        try {
          const parsed = JSON.parse(payload);
          if (typeof parsed?.type === 'string') wsEventTypes.push(parsed.type);
        } catch {
          // Ignore non-JSON frames; binary media chunks are asserted through UI state.
        }
      });
    });

    await page.goto('/');

    await expect(page.getByRole('img', { name: 'FastVideo' })).toBeVisible();

    const continuation = page.getByLabel('Continuation prompt');
    await expect(continuation).toBeVisible();
    await continuation.fill('A glass whale swims above a neon forest');

    await page.getByRole('button', { name: /^generate$/i }).click();

    await expect(continuation).toBeDisabled({ timeout: 30_000 });
    await expect(continuation).toHaveAttribute('placeholder', /generating video/i);
    await expect(page.getByRole('button', { name: /^leave$/i })).toBeVisible({ timeout: 30_000 });
    await expect(page.locator('video').first()).toHaveCount(1);

    await expect.poll(() => wsEventTypes).toEqual(
      expect.arrayContaining(['ltx2_stream_start', 'ltx2_segment_start', 'media_init']),
    );
  });
});
