// The public JavaScript contract lives in docs/cookbook/design.md.
// @ts-expect-error The shared ES module has no TypeScript declarations.
import * as shared from "../../docs/js/cookbook-config.mjs";

export type Value = null | boolean | number | string | Value[] | { [key: string]: Value };
export interface Schema {
  title?: string;
  description?: string;
  type?: string | string[];
  enum?: Value[];
  anyOf?: Schema[];
  minimum?: number;
  maximum?: number;
  multipleOf?: number;
}
export interface Field {
  path: string;
  schema: Schema;
  value?: Value;
}
export interface Index {
  url: string;
  models: {
    id: string;
    title: string;
    model_id: string;
    deployments: {
      id: string;
      label: string;
      runtime: string;
      workload: string;
    }[];
  }[];
}
export interface Catalog {
  id: string;
  title: string;
  summary?: string;
  workload: string;
  env: Record<string, string>;
  base_config: Record<string, Value>;
  runtime: {
    label: string;
    backend: string;
    interface: string;
    install_url?: string;
    client_guide_url?: string;
  };
  requirements?: string[];
  guide?: { url: string; label?: string };
}
export interface Result {
  config: Record<string, Value>;
  yaml: string;
  command: string;
  clientCommand: string;
  websocketUrl?: string;
}
interface Api {
  sampleRequest(
    catalog: {
      runtime: { interface: string; server_defaults: { host: string; port: number } };
      model: { id: string };
      workload: string;
    },
    config: Record<string, Value>,
  ): { command: string };
  loadIndex(url: string, options?: { signal?: AbortSignal }): Promise<Index>;
  loadDeployment(index: Index, id: string, options?: { signal?: AbortSignal }): Promise<Catalog>;
  getOptions(catalog: Catalog, config?: Catalog["base_config"], edits?: Record<string, Value>): Field[];
  resolveConfig(catalog: Catalog, edits?: Record<string, Value>): Result;
}
export const api = shared as Api;

export function types(schema: Schema): string[] {
  return [
    ...new Set([
      ...(Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : []),
      ...(schema.anyOf?.flatMap(types) ?? []),
    ]),
  ];
}

export function parseField(field: Field, raw: string): Value {
  const kind = types(field.schema);
  if (field.schema.enum || kind.includes("boolean") || kind.includes("object") || kind.includes("array")) {
    return JSON.parse(raw) as Value;
  }
  if (kind.includes("number") || kind.includes("integer")) {
    if (!raw.trim() || !Number.isFinite(Number(raw))) throw new Error("Enter a number.");
    return Number(raw);
  }
  return raw;
}

/** Keep catalog guide links on the current site and its deployment prefix. */
export function guideUrl(path: string): string {
  const url = new URL(
    path,
    new URL(`${import.meta.env.BASE_URL.replace(/\/$/, "")}/assets/cookbook-config/index.json`, window.location.origin),
  );
  if (url.origin === "https://haoailab.com" && url.pathname.startsWith("/FastVideo/")) {
    url.host = window.location.host;
    url.protocol = window.location.protocol;
  }
  return url.href;
}
