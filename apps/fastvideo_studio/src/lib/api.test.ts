import { afterEach, expect, it, vi } from 'vitest';
import { createJob } from './api';

afterEach(() => vi.unstubAllGlobals());

it('shows the backend validation error returned when creating a job', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
    JSON.stringify({ detail: 'Unknown attention backend typo' }),
    { status: 400, headers: { 'Content-Type': 'application/json' } },
  )));
  await expect(createJob({ model_id: 'model', prompt: 'cat', attention_backend: 'typo' }))
    .rejects.toThrow('Unknown attention backend typo');
});
