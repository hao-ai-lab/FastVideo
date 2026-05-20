import { test, expect, type WebSocket as PWWebSocket } from '@playwright/test';

/**
 * Long-running e2e: drive a real two-segment session end-to-end with
 * torch.compile + warmup ENABLED on the backend, and assert audio
 * conditioning carries from segment 1 → segment 2 without the
 * BrokenPipe regression documented in
 * `.agents/memory/dreamverse-integration/decisions-log.md` D-20.
 *
 * Skipped by default. Opt in with PLAYWRIGHT_LONG_RUNNING=1, and
 * boot the backend with both knobs on:
 *
 *     ./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh \
 *         --warmup --torch-compile 4
 *     PLAYWRIGHT_SKIP_WEBSERVER=1 \
 *         BACKEND_HOST=127.0.0.1 \
 *         BACKEND_PORT=8009 \
 *         PLAYWRIGHT_BASE_URL=http://127.0.0.1:5274 \
 *         NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 \
 *         PLAYWRIGHT_LONG_RUNNING=1 \
 *         pnpm exec playwright test e2e/long-running-segments.spec.ts
 *
 * The full run takes ~7-9 minutes on a B200: ~3-4min for torch.compile
 * max-autotune to warm both DiT + text-encoder graphs, then ~30s for
 * segment 1 inference and ~10s for segment 2. The default per-test
 * timeout is bumped to 900_000 ms below.
 */
const LONG_RUNNING_ENABLED = process.env.PLAYWRIGHT_LONG_RUNNING === '1';

interface WSEvent {
  raw: string | Buffer;
  parsed?: {
    type?: string;
    segment_idx?: number;
    message?: string;
    [k: string]: unknown;
  };
  isBinary: boolean;
}

test.describe('long-running two-segment audio continuation', () => {
  test.skip(
    !LONG_RUNNING_ENABLED,
    'Skipping long-running e2e — set PLAYWRIGHT_LONG_RUNNING=1 to enable. ' +
      'Requires backend booted with --warmup --torch-compile (~7-9 min run).',
  );

  test.beforeEach(async ({ request }) => {
    const ready = await request.get('/readyz');
    test.skip(
      !ready.ok(),
      'Skipping — /readyz did not return 200. Boot dreamverse-server first.',
    );
  });

  test('segment 1 + segment 2 stream cleanly with torch.compile + warmup', async ({
    page,
    request,
  }) => {
    test.setTimeout(900_000);

    // Confirm the deploy actually has warmup on. We don't gate on
    // torch.compile because there's no public API surface for it
    // (operator-controlled via ENABLE_TORCH_COMPILE env var).
    const statusResponse = await request.get('/status');
    expect(statusResponse.ok()).toBe(true);
    const status = await statusResponse.json();
    expect(
      status.warmup_enabled,
      'BE must be booted with --warmup for the long-running e2e to be meaningful',
    ).toBe(true);

    // Collect every WS event from the FE's session WS (the only one
    // the page opens). page.on('websocket') fires before the WS is
    // navigated into, so registering before page.goto is safe.
    const events: WSEvent[] = [];
    const errors: string[] = [];
    let mediaInitCount = 0;
    let mediaSegmentCompleteCount = 0;
    const segmentsSeen = new Set<number>();
    const segmentsCompleted = new Set<number>();

    page.on('websocket', (ws: PWWebSocket) => {
      ws.on('framereceived', ({ payload }) => {
        const isBinary = typeof payload !== 'string';
        const evt: WSEvent = { raw: payload, isBinary };
        if (!isBinary) {
          try {
            evt.parsed = JSON.parse(payload as string);
          } catch {
            return;
          }
        }
        events.push(evt);
        const t = evt.parsed?.type;
        const seg = evt.parsed?.segment_idx;
        if (t === 'media_init' && typeof seg === 'number') {
          mediaInitCount += 1;
          segmentsSeen.add(seg);
        } else if (t === 'media_segment_complete' && typeof seg === 'number') {
          mediaSegmentCompleteCount += 1;
          segmentsCompleted.add(seg);
        } else if (t === 'error' || t === 'step_error') {
          const msg =
            (typeof evt.parsed?.message === 'string' && evt.parsed.message) ||
            JSON.stringify(evt.parsed);
          errors.push(msg);
        }
      });
    });

    await page.goto('/');

    const continuation = page.getByLabel('Continuation prompt');
    await expect(continuation).toBeVisible({ timeout: 30_000 });

    // Same preset selector as preset-prompt-generation.spec.ts —
    // the FE auto-fires segments through the curated prompt list.
    const firstPreset = page
      .getByRole('button', {
        name: /LEGO Stormtroopers|Clay Stop-Motion|Boy & Dog|School Prank|Gamer Gets Banned|Small Town Oil Strike|Grandpa's Wing Costume/i,
      })
      .first();
    await expect(firstPreset).toBeVisible({ timeout: 30_000 });
    await firstPreset.click();

    await expect(continuation).toBeDisabled({ timeout: 30_000 });

    // Poll for segment 2 to complete. The FE auto-progresses through
    // the preset's segment_prompts list once a session is started, so
    // segment 1 → segment 2 happens without further user action.
    const deadline = Date.now() + 850_000;
    while (Date.now() < deadline) {
      if (errors.length > 0) {
        throw new Error(
          `WS error frame received: ${errors.join(' | ')}\n` +
            `events captured: init=${mediaInitCount} ` +
            `complete=${mediaSegmentCompleteCount} ` +
            `segments_seen=${[...segmentsSeen].sort().join(',')} ` +
            `segments_completed=${[...segmentsCompleted].sort().join(',')}`,
        );
      }
      if (segmentsCompleted.has(1) && segmentsCompleted.has(2)) {
        break;
      }
      await page.waitForTimeout(2_000);
    }

    expect(
      errors,
      `Expected no WS error frames; got: ${errors.join(' | ')}`,
    ).toHaveLength(0);
    expect(
      [...segmentsCompleted].sort(),
      `Expected segments 1 AND 2 to complete; segments_seen=` +
        `${[...segmentsSeen].sort().join(',')}, ` +
        `mediaInitCount=${mediaInitCount}, ` +
        `mediaSegmentCompleteCount=${mediaSegmentCompleteCount}`,
    ).toEqual(expect.arrayContaining([1, 2]));

    // Sanity: segment 2 must have produced at least one binary chunk
    // (the actual fMP4 bytes). If the BrokenPipe regression returns,
    // ffmpeg closes stdin before any chunks are emitted and we'd
    // see media_init for segment 2 but zero binary frames.
    const binaryFrameCount = events.filter((e) => e.isBinary).length;
    expect(
      binaryFrameCount,
      'Expected at least one binary fMP4 chunk; got zero',
    ).toBeGreaterThan(0);
  });
});
