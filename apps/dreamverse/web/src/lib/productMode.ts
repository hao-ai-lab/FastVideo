import { parseSimpleQuery } from './simpleMode';

export const PRODUCT_MODE_STREAMING = 'streaming';
export const PRODUCT_MODE_SINGLE5S = 'single5s';

export function normalizeProductMode(value: string | null | undefined): string {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === PRODUCT_MODE_SINGLE5S) {
    return PRODUCT_MODE_SINGLE5S;
  }
  return PRODUCT_MODE_STREAMING;
}

interface ResolveProductModeOptions {
  buildProductMode?: string;
  search?: string;
  allowRuntimeOverride?: boolean;
}

export function resolveProductMode({
  buildProductMode = PRODUCT_MODE_STREAMING,
  search = '',
  allowRuntimeOverride = false,
}: ResolveProductModeOptions = {}): string {
  const normalizedBuildMode = normalizeProductMode(buildProductMode);
  if (!allowRuntimeOverride) {
    return normalizedBuildMode;
  }

  const runtimeSimpleMode = parseSimpleQuery(search);
  if (runtimeSimpleMode === true) {
    return PRODUCT_MODE_SINGLE5S;
  }
  if (runtimeSimpleMode === false) {
    return PRODUCT_MODE_STREAMING;
  }
  return normalizedBuildMode;
}

interface ResolveSimpleModeOptions {
  buildProductMode?: string;
  search?: string;
  allowRuntimeOverride?: boolean;
}

export function resolveSimpleMode({
  buildProductMode = PRODUCT_MODE_STREAMING,
  search = '',
  allowRuntimeOverride = false,
}: ResolveSimpleModeOptions = {}): boolean {
  return resolveProductMode({
    buildProductMode,
    search,
    allowRuntimeOverride,
  }) === PRODUCT_MODE_SINGLE5S;
}
