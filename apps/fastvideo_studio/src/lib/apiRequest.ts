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

interface PathVariable {
  name: string;
  path: (string | number)[];
  value: string;
}

/** Only expose existing string fields, including an explicitly empty path. */
export function getRequestPathVariables(request: ApiRequest): PathVariable[] {
  if (request.body === undefined) return [];
  const body = JSON.parse(request.body);
  if (!body || typeof body !== 'object' || Array.isArray(body)) return [];
  const fields = [
    ['image_path', 'IMAGE_PATH'],
    ['last_image_path', 'LAST_IMAGE_PATH'],
    ['data_path', 'DATA_PATH'],
    ['validation_dataset_file', 'VALIDATION_DATASET_PATH'],
  ];
  const variables: PathVariable[] = fields.flatMap(([field, name]) =>
    typeof body[field] === 'string' ? [{ name, path: [field], value: body[field] }] : [],
  );
  if (Array.isArray(body.references)) {
    body.references.forEach((reference: { source?: unknown } | null, index: number) => {
      if (reference && typeof reference.source === 'string') {
        variables.push({ name: `REFERENCE_${index + 1}_PATH`, path: ['references', index, 'source'], value: reference.source });
      }
    });
  }
  return variables;
}

/** A jq object expression, with path values supplied separately through --arg. */
function jqTemplate(value: unknown, variables: PathVariable[], path: (string | number)[] = []): string {
  const variable = variables.find((entry) => JSON.stringify(entry.path) === JSON.stringify(path));
  if (variable) return `$${variable.name}`;
  const indent = '  '.repeat(path.length);
  if (Array.isArray(value)) {
    if (!value.length) return '[]';
    return `[\n${value.map((item, index) => `${indent}  ${jqTemplate(item, variables, [...path, index])}`).join(',\n')}\n${indent}]`;
  }
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value);
    if (!entries.length) return '{}';
    return `{\n${entries.map(([key, item]) => `${indent}  ${JSON.stringify(key)}: ${jqTemplate(item, variables, [...path, key])}`).join(',\n')}\n${indent}}`;
  }
  return JSON.stringify(value);
}

/** Format the same serialized request used by fetch; no request is executed. */
export function requestToCurl(baseUrl: string, request: ApiRequest): string {
  const variables = getRequestPathVariables(request);
  const exports = [
    `export BACKEND_URL=${shellQuote(baseUrl.replace(/\/+$/, ''))}`,
    ...variables.map(({ name, value }) => `export ${name}=${shellQuote(value)}`),
  ];
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
    if (variables.length) {
      lines.push('  --data-raw "$REQUEST_BODY"');
    } else {
      const prettyBody = JSON.stringify(JSON.parse(request.body), null, 2);
      lines.push(`  --data-raw ${shellQuote(prettyBody)}`);
    }
  }
  const prepareBody = variables.length
    ? [
        'REQUEST_BODY=$(jq -n',
        ...variables.map(({ name }) => `  --arg ${name} "$${name}"`),
        `  ${shellQuote(jqTemplate(JSON.parse(request.body!), variables))}`,
      ].join(' \\\n') + '\n) && \\\n'
    : '';
  return `${exports.join('\n')}\n\n${prepareBody}${lines.join(' \\\n')}\n`;
}
