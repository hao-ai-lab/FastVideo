import { execFileSync, spawnSync } from 'node:child_process';
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

const hasJq = spawnSync('jq', ['--version']).status === 0;

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

  it.skipIf(!hasJq)('round-trips exported path variables with hostile shell and JSON characters', () => {
    const hostile = "/uploads/person's \"clip\" $(printf injected) `printf bad` $HOME\\name\nlast line\n";
    const payload = {
      model_id: 'wan/test',
      prompt: 'Keep \\(literal jq interpolation) and <img src=x> as plain text',
      image_path: hostile,
      last_image_path: `${hostile}last.png`,
      data_path: ` ${hostile} `,
      validation_dataset_file: '',
      references: [{ source: hostile, media_type: 'image' }, { source: 'https://example.test/a?q="x"', media_type: 'video' }],
      num_frames: 60,
      guidance_scale: 5.25,
      optional: null,
      enabled: false,
    };
    const request = Object.freeze(jsonApiRequest('POST', '/jobs', payload));
    const originalBody = request.body;
    const command = requestToCurl('https://configured.test/api', request);
    for (const name of ['IMAGE_PATH', 'LAST_IMAGE_PATH', 'DATA_PATH', 'VALIDATION_DATASET_PATH', 'REFERENCE_1_PATH', 'REFERENCE_2_PATH']) {
      expect(command).toContain(`export ${name}=`);
      expect(command).toContain(`--arg ${name} "$${name}"`);
    }
    expect(command).toContain("export VALIDATION_DATASET_PATH=''");
    expect(command).toContain('"image_path": $IMAGE_PATH');
    const args = curlArguments(command);
    expect(args[0]).toBe('--request');
    expect(JSON.parse(args.at(-1)!)).toEqual(payload);
    expect(request.body).toBe(originalBody);
  });

  it.skipIf(!hasJq)('applies edits to exported paths without changing other request fields', () => {
    const payload = { model_id: 'wan/test', prompt: '$IMAGE_PATH stays literal', image_path: '/uploads/original.png' };
    const command = requestToCurl('https://configured.test/api', createJobRequest(payload));
    const replacement = '/edited/"quoted"\n$(printf injected)`printf bad`\\image.png';
    const edited = command.replace('\n\n', '\nIMAGE_PATH="$OVERRIDE_IMAGE_PATH"\n\n');
    const stdout = execFileSync('/bin/sh', ['-c', `curl() { printf '%s\\000' "$@"; }\n${edited}`], {
      env: { ...process.env, OVERRIDE_IMAGE_PATH: replacement },
    });
    const args = stdout.toString().split('\0').slice(0, -1);
    expect(JSON.parse(args.at(-1)!)).toEqual({ ...payload, image_path: replacement });
  });

  it('does not add jq or path exports when path fields are omitted or null', () => {
    const request = jsonApiRequest('PATCH', '/jobs/job-1', { image_path: null, references: [], validation_dataset_file: undefined });
    const command = requestToCurl('https://configured.test/api', request);
    expect(command).not.toContain('jq');
    expect(command).not.toContain('export IMAGE_PATH=');
    expect(JSON.parse(curlArguments(command).at(-1)!)).toEqual({ image_path: null, references: [] });
  });

  it('does not invoke curl when the required jq command fails', () => {
    const command = requestToCurl('https://configured.test/api', jsonApiRequest('POST', '/jobs', { image_path: 'file.png' }));
    const result = spawnSync('/bin/sh', ['-c', `jq() { return 127; }\ncurl() { printf 'CURL WAS CALLED'; }\n${command}`]);
    expect(result.status).toBe(127);
    expect(result.stdout.toString()).toBe('');
  });
});
