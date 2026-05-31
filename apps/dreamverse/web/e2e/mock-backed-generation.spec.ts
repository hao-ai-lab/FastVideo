import { stat } from 'node:fs/promises';

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

  test('streams, plays, and surfaces a downloadable clip', async ({ page, request }) => {
    const health = await request.get('/healthz');
    const body = health.ok() ? await health.json() : {};
    test.skip(
      body.service !== 'ltx2-streaming-mock-server',
      'Mock-backed playback assertions require dreamverse.mock_server on BACKEND_PORT (default 8009).',
    );

    await page.addInitScript(() => {
      (window as unknown as { __sharedFiles: unknown }).__sharedFiles = null;
      (navigator as unknown as { share: (d: { files?: File[] }) => Promise<void> }).share = async (data) => {
        const files = Array.isArray(data?.files) ? data.files : [];
        (window as unknown as { __sharedFiles: unknown }).__sharedFiles = files.map((f) => ({
          name: f.name,
          type: f.type,
          size: f.size,
        }));
      };
    });

    await page.goto('/');

    const continuation = page.getByLabel('Continuation prompt');
    await expect(continuation).toBeVisible();
    await continuation.fill('A glass whale swims above a neon forest');
    await page.getByRole('button', { name: /^generate$/i }).click();

    const liveVideo = page.locator('video:not(.hidden)');
    await expect(liveVideo).toHaveCount(1);

    await test.step('MSE pipeline attaches the live <video>', async () => {
      // Chromium/Firefox attach via blob: URL; Safari uses srcObject with ManagedMediaSource.
      await expect
        .poll(
          async () =>
            liveVideo.evaluate(
              (v: HTMLVideoElement) =>
                v.src.startsWith('blob:') || v.srcObject !== null,
            ),
          { timeout: 30_000 },
        )
        .toBe(true);
    });

    await test.step('SourceBuffer accepts fMP4 and decoder produces frames', async () => {
      await expect
        .poll(
          async () =>
            liveVideo.evaluate((v: HTMLVideoElement) =>
              v.buffered.length > 0 ? v.buffered.end(0) : 0,
            ),
          { timeout: 60_000 },
        )
        .toBeGreaterThan(0);

      // readyState >= 3 = HAVE_FUTURE_DATA.
      await expect
        .poll(
          async () => liveVideo.evaluate((v: HTMLVideoElement) => v.readyState),
          { timeout: 60_000 },
        )
        .toBeGreaterThanOrEqual(3);
    });

    await test.step('<video> playback advances past the first second', async () => {
      await liveVideo.evaluate(async (v: HTMLVideoElement) => {
        if (v.paused) {
          try {
            await v.play();
          } catch {
          }
        }
      });

      await expect
        .poll(
          async () => liveVideo.evaluate((v: HTMLVideoElement) => v.currentTime),
          { timeout: 30_000 },
        )
        .toBeGreaterThan(0);

      await expect
        .poll(
          async () =>
            liveVideo.evaluate((v: HTMLVideoElement) =>
              v.ended ? v.duration : v.currentTime,
            ),
          { timeout: 10_000 },
        )
        .toBeGreaterThanOrEqual(1.0);
    });

    await test.step('the AAC audio track is demuxed and decoded', async () => {
      // The mock emits AAC (mp4a.40.2), matching the real backend
      // (av_streaming.py). Proves the FE decoded the audio track.
      await expect
        .poll(
          async () =>
            liveVideo.evaluate((el: HTMLVideoElement) => {
              const v = el as HTMLVideoElement & {
                audioTracks?: { length: number };
                mozHasAudio?: boolean;
                webkitAudioDecodedByteCount?: number;
              };
              return (
                (v.audioTracks?.length ?? 0) > 0 ||
                v.mozHasAudio === true ||
                (v.webkitAudioDecodedByteCount ?? 0) > 0
              );
            }),
          { timeout: 10_000 },
        )
        .toBe(true);
    });

    await test.step('completed clip surfaces a working download', async () => {
      const downloadButton = page.getByRole('button', { name: /download video|share video/i });
      await expect(downloadButton).toBeVisible({ timeout: 30_000 });

      const buttonLabel = (await downloadButton.getAttribute('aria-label')) ?? '';
      const isShareFlow = /share video/i.test(buttonLabel);

      if (isShareFlow) {
        await downloadButton.click();
        await expect
          .poll(
            async () => page.evaluate(() => (window as { __sharedFiles?: unknown }).__sharedFiles),
            { timeout: 10_000 },
          )
          .not.toBeNull();
        const shared = (await page.evaluate(
          () => (window as { __sharedFiles?: Array<{ name: string; type: string; size: number }> }).__sharedFiles,
        )) ?? [];
        expect(shared).toHaveLength(1);
        expect(shared[0].name).toMatch(/\.(mp4|webm)$/);
        expect(shared[0].size).toBeGreaterThan(0);
      } else {
        const downloadPromise = page.waitForEvent('download');
        await downloadButton.click();
        const download = await downloadPromise;
        expect(download.suggestedFilename()).toMatch(/\.(mp4|webm)$/);

        const savedPath = await download.path();
        expect(savedPath).not.toBeNull();
        const { size } = await stat(savedPath!);
        expect(size).toBeGreaterThan(0);
      }
    });

    await test.step('project history sidebar lists the current session', async () => {
      const sidebar = page.getByRole('complementary', { name: 'Project history' });
      await page.getByRole('button', { name: 'Toggle sidebar' }).click();

      await expect(sidebar).toBeInViewport();

      await expect(sidebar.getByText('Current', { exact: true })).toBeVisible();
      await expect(sidebar.getByText('Active', { exact: true })).toBeVisible();
    });

    const lastError = await liveVideo.evaluate((v: HTMLVideoElement) =>
      v.error ? `${v.error.code}: ${v.error.message ?? ''}` : null,
    );
    expect(lastError).toBeNull();
  });

  test('starts a new project and switches back to the prior session', async ({ page, request }) => {
    const health = await request.get('/healthz');
    const body = health.ok() ? await health.json() : {};
    test.skip(
      body.service !== 'ltx2-streaming-mock-server',
      'Mock-backed project lifecycle assertions require dreamverse.mock_server on BACKEND_PORT (default 8009).',
    );

    await page.goto('/');

    await test.step('complete a generation so there is a project to save', async () => {
      const continuation = page.getByLabel('Continuation prompt');
      await expect(continuation).toBeVisible();
      await continuation.fill('Aurora over a frozen lake');
      await page.getByRole('button', { name: /^generate$/i }).click();
      await expect(
        page.getByRole('button', { name: /download video|share video/i }),
      ).toBeVisible({ timeout: 60_000 });
    });

    const sidebar = page.getByRole('complementary', { name: 'Project history' });

    await test.step('"New project" closes the sidebar and resets the composer', async () => {
      await page.getByRole('button', { name: 'Toggle sidebar' }).click();
      await expect(sidebar).toBeInViewport();
      await sidebar.getByRole('button', { name: /^new project$/i }).click();
      await expect(sidebar).not.toBeInViewport();
      await expect(page.getByRole('button', { name: /^generate$/i })).toBeVisible({ timeout: 30_000 });
      const continuation = page.getByLabel('Continuation prompt');
      await expect(continuation).toBeEnabled();
      await expect(continuation).toHaveValue('');
    });

    await test.step('the prior session appears under timeline', async () => {
      await page.getByRole('button', { name: 'Toggle sidebar' }).click();
      await expect(sidebar).toBeInViewport();
      await expect(sidebar.getByText('Previous', { exact: true })).toBeVisible({ timeout: 30_000 });
      await expect(sidebar.getByText('just now').first()).toBeVisible();
    });

    await test.step('clicking the prior session enters viewing mode', async () => {
      const priorRow = sidebar.locator('div[role="button"]').filter({ hasText: 'just now' }).first();
      await priorRow.click();
      await expect(sidebar).not.toBeInViewport();
      await expect(page.locator('video[autoplay][loop]')).toBeVisible({ timeout: 30_000 });
    });
  });

  test('saved projects persist across a page reload', async ({ page, request }) => {
    const health = await request.get('/healthz');
    const body = health.ok() ? await health.json() : {};
    test.skip(
      body.service !== 'ltx2-streaming-mock-server',
      'Mock-backed persistence assertions require dreamverse.mock_server on BACKEND_PORT (default 8009).',
    );

    await page.goto('/');

    await test.step('complete a generation', async () => {
      const continuation = page.getByLabel('Continuation prompt');
      await expect(continuation).toBeVisible();
      await continuation.fill('Aurora over a frozen lake');
      await page.getByRole('button', { name: /^generate$/i }).click();
      await expect(
        page.getByRole('button', { name: /download video|share video/i }),
      ).toBeVisible({ timeout: 60_000 });
    });

    const sidebar = page.getByRole('complementary', { name: 'Project history' });

    await test.step('persist via "New project" and confirm it lands under Previous', async () => {
      await page.getByRole('button', { name: 'Toggle sidebar' }).click();
      await expect(sidebar).toBeInViewport();
      await sidebar.getByRole('button', { name: /^new project$/i }).click();
      await expect(page.getByRole('button', { name: /^generate$/i })).toBeVisible({ timeout: 30_000 });
      await page.getByRole('button', { name: 'Toggle sidebar' }).click();
      await expect(sidebar).toBeInViewport();
      await expect(sidebar.getByText('Previous', { exact: true })).toBeVisible({ timeout: 30_000 });
      await expect(sidebar.getByText('just now').first()).toBeVisible();
    });

    await test.step('after page reload, the prior project is still in Previous', async () => {
      await page.reload();
      await page.getByRole('button', { name: 'Toggle sidebar' }).click();
      await expect(sidebar).toBeInViewport();
      await expect(sidebar.getByText('Previous', { exact: true })).toBeVisible({ timeout: 30_000 });
      await expect(sidebar.getByText(/^(just now|\d+m ago)$/).first()).toBeVisible();
    });
  });
});
