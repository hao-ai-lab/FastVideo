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
  base_config: Record<string, Value>;
  runtime: {
    label: string;
    interface: string;
    install_url?: string;
    client_guide_url?: string;
  };
  requirements?: string[];
  guide?: { url: string; label?: string };
}
export interface Result {
  yaml: string;
  command: string;
  clientCommand: string;
  websocketUrl?: string;
}
interface Api {
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

/** Catalog guide links are relative to the original index, which remains published by MkDocs. */
export function guideUrl(path: string): string {
  return new URL(path, "https://haoailab.com/FastVideo/assets/cookbook-config/index.json").href;
}
