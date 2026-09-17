// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest } from './api';

/** JSON requests supported by both fetch and the copyable cURL preview. */
export interface ApiRequest {
  path: string;
  method: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: string;
}

export function jsonApiRequest(
  method: ApiRequest['method'],
  path: string,
  data?: unknown,
): ApiRequest {
  return {
    method,
    path,
    ...(data === undefined
      ? {}
      : {
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        }),
  };
}

export function createJobRequest(job: CreateJobRequest): ApiRequest {
  return jsonApiRequest('POST', '/jobs', job);
}

export function updateJobRequest(
  jobId: string,
  updates: Record<string, unknown>,
): ApiRequest {
  return jsonApiRequest('PATCH', `/jobs/${encodeURIComponent(jobId)}`, updates);
}

export function fetchApiRequest(baseUrl: string, request: ApiRequest): Promise<Response> {
  const { path, ...init } = request;
  return fetch(`${baseUrl.replace(/\/+$/, '')}${path}`, init);
}

/** POSIX shell quoting keeps prompts, URLs, and headers literal. */
function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

/** Format the same serialized request used by fetch; no request is executed. */
export function requestToCurl(baseUrl: string, request: ApiRequest): string {
  const lines = [
    `curl --request ${request.method}`,
    `  "\${BACKEND_URL}"${shellQuote(request.path)}`,
    ...Object.entries(request.headers ?? {}).map(
      ([name, value]) => `  --header ${shellQuote(`${name}: ${value}`)}`,
    ),
  ];
  if (request.body !== undefined) {
    // Our descriptors are JSON-only. Reformat the serialized body, so omitted
    // values and number coercion exactly match the request sent by fetch.
    const prettyBody = JSON.stringify(JSON.parse(request.body), null, 2);
    lines.push(`  --data-raw ${shellQuote(prettyBody)}`);
  }
  return `export BACKEND_URL=${shellQuote(baseUrl.replace(/\/+$/, ''))}\n\n${lines.join(' \\\n')}\n`;
}
