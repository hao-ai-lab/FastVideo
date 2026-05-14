import { describe, expect, it } from 'vitest';

import {
  PRODUCT_MODE_SINGLE5S,
  PRODUCT_MODE_STREAMING,
  normalizeProductMode,
  resolveProductMode,
  resolveSimpleMode,
} from './productMode';

describe('normalizeProductMode', () => {
  it('keeps supported product modes', () => {
    expect(normalizeProductMode('streaming')).toBe(PRODUCT_MODE_STREAMING);
    expect(normalizeProductMode('single5s')).toBe(PRODUCT_MODE_SINGLE5S);
  });

  it('falls back to streaming for unknown values', () => {
    expect(normalizeProductMode('')).toBe(PRODUCT_MODE_STREAMING);
    expect(normalizeProductMode('unknown')).toBe(PRODUCT_MODE_STREAMING);
  });
});

describe('resolveProductMode', () => {
  it('uses the build mode when runtime overrides are disabled', () => {
    expect(resolveProductMode({
      buildProductMode: PRODUCT_MODE_SINGLE5S,
      search: '?simple=0',
      allowRuntimeOverride: false,
    })).toBe(PRODUCT_MODE_SINGLE5S);
  });

  it('allows the simple query to override the build mode in dev/test', () => {
    expect(resolveProductMode({
      buildProductMode: PRODUCT_MODE_STREAMING,
      search: '?simple=1',
      allowRuntimeOverride: true,
    })).toBe(PRODUCT_MODE_SINGLE5S);
    expect(resolveProductMode({
      buildProductMode: PRODUCT_MODE_SINGLE5S,
      search: '?simple=0',
      allowRuntimeOverride: true,
    })).toBe(PRODUCT_MODE_STREAMING);
  });
});

describe('resolveSimpleMode', () => {
  it('returns whether the effective product mode is single5s', () => {
    expect(resolveSimpleMode({
      buildProductMode: PRODUCT_MODE_STREAMING,
      search: '',
      allowRuntimeOverride: false,
    })).toBe(false);
    expect(resolveSimpleMode({
      buildProductMode: PRODUCT_MODE_SINGLE5S,
      search: '',
      allowRuntimeOverride: false,
    })).toBe(true);
  });
});
