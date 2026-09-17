import { execFileSync } from 'node:child_process';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createJob, updateJob } from './api';
import { STORAGE_KEY, DEFAULT_OPTIONS } from './defaultOptions';
import { createJobRequest, jsonApiRequest, requestToCurl, updateJobRequest } from './apiRequest';

afterEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

/** Capture shell arguments without executing curl or making a request. */
function curlArguments(command: string): string[] {
  const stdout = execFileSync('/bin/sh', ['-c', `curl() { printf '%s\\000' "$@"; }\n${command}`]);
  return stdout.toString().split('\0').slice(0, -1);
}

describe('requestToCurl', () => {
  it('keeps shell syntax in prompts, headers, and URLs literal', () => {
    const baseUrl = "https://example.test/a'b/$(printf injected)/`printf bad`/$HOME";
    const prompt = "A person's cat says \"hello\"; $(printf injected) `printf bad` $HOME\nwith a new line and \\ slashes";
    const request = {
      ...jsonApiRequest('POST', '/jobs?label="$HOME"', { model_id: 'wan/test', prompt }),
      headers: { 'Content-Type': 'application/json', 'X-Label': "it's $HOME" },
    };
    const command = requestToCurl(baseUrl, request);
    const args = curlArguments(command);

    expect(command).toContain('export BACKEND_URL=');
    expect(command).toContain('\n  "model_id": "wan/test",\n');
    expect(args).toEqual([
      '--request', 'POST', `${baseUrl}${request.path}`,
      '--header', 'Content-Type: application/json',
      '--header', "X-Label: it's $HOME",
      '--data-raw', JSON.stringify({ model_id: 'wan/test', prompt }, null, 2),
    ]);
  });

  it('handles JSON omission, trailing URL slashes, and requests without a body', () => {
    const request = jsonApiRequest('POST', '/jobs', { prompt: 'test', omitted: undefined });
    expect(JSON.parse(curlArguments(requestToCurl('http://localhost:8189/api///', request)).at(-1)!))
      .toEqual({ prompt: 'test' });
    expect(curlArguments(requestToCurl('http://localhost:8189/api/', jsonApiRequest('GET', '/jobs'))))
      .toEqual(['--request', 'GET', 'http://localhost:8189/api/jobs']);
  });

  it('shares the create and edit descriptors with the actual API fetch', async () => {
    const baseUrl = 'https://configured.test/api';
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...DEFAULT_OPTIONS, apiServerBaseUrl: `${baseUrl}/` }));
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: 'job-1' }) });
    vi.stubGlobal('fetch', fetchMock);
    const payload = { model_id: 'wan/test', prompt: 'test', num_frames: 60 };

    await createJob(payload);
    await updateJob('job/1', payload);

    for (const [index, request] of [createJobRequest(payload), updateJobRequest('job/1', payload)].entries()) {
      const [url, init] = fetchMock.mock.calls[index];
      const args = curlArguments(requestToCurl(baseUrl, request));
      expect(url).toBe(args[2]);
      expect(init.method).toBe(args[1]);
      expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
      expect(JSON.parse(init.body)).toEqual(JSON.parse(args.at(-1)!));
    }
    expect(fetchMock.mock.calls[1][0]).toBe(`${baseUrl}/jobs/job%2F1`);
  });
});
