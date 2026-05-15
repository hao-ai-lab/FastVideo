import { test, expect } from '@playwright/test';

/**
 * Smoke layer 1: the Dreamverse Python server is reachable and the
 * Next.js rewrite proxy at /healthz forwards to it. This runs before
 * any UI interaction so a UI failure has a known-good baseline.
 */
test.describe('backend health', () => {
  test('healthz returns ok via the next.js rewrite', async ({ request }) => {
    const response = await request.get('/healthz');
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.status).toBe('ok');
    expect(body.service).toBe('ltx2-streaming-backend');
  });

  test('readyz reports gpu pool state', async ({ request }) => {
    // /readyz returns 200 once the GPU pool is warm; 503 with a
    // {detail: ...} body otherwise. Either is a valid integration
    // signal — we just want to confirm the route is wired and the
    // payload shape is what the FE expects.
    const response = await request.get('/readyz');
    expect([200, 503]).toContain(response.status());
    const text = await response.text();
    if (response.status() === 503) {
      // Frontend reads .detail to render the "wait for warmup" banner.
      expect(text).toMatch(/detail/);
    }
  });

  test('status endpoint exposes gpu pool snapshot', async ({ request }) => {
    const response = await request.get('/status');
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body).toHaveProperty('total_gpus');
    expect(body).toHaveProperty('gpu_status');
    expect(typeof body.total_gpus).toBe('number');
  });

  test('prompt-system-config exposes the operator-tunable prompts', async ({ request }) => {
    const response = await request.get('/prompt-system-config');
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    // page.tsx reads these to seed the composer when the user opens
    // the prompt-edit drawer; missing keys = silent UI breakage.
    expect(body).toHaveProperty('next_segment_system_prompt');
    expect(body).toHaveProperty('auto_extension_system_prompt');
    expect(body).toHaveProperty('rewrite_window_system_prompt');
  });

  test('curated presets endpoint serves a non-empty list (devtools only)', async ({ request }) => {
    const response = await request.get('/curated-presets');
    test.skip(
      response.status() === 404,
      'Skipping curated-presets check: backend was not booted with ' +
        'FASTVIDEO_ENABLE_DEVTOOLS=1, so the devtools-only route is not ' +
        'mounted. Re-run with that env var to exercise it.',
    );
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(Array.isArray(body.presets)).toBeTruthy();
    expect(body.presets.length).toBeGreaterThan(0);
    for (const preset of body.presets) {
      expect(preset).toHaveProperty('id');
      expect(preset).toHaveProperty('label');
      expect(Array.isArray(preset.segment_prompts)).toBeTruthy();
    }
  });
});
