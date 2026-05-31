import { describe, expect, it } from 'vitest';

import {
  parseEditableQuery,
  resolveEditableMode,
} from './editableMode';

describe('parseEditableQuery', () => {
  it('parses true/false values', () => {
    expect(parseEditableQuery('?editable=true')).toBe(true);
    expect(parseEditableQuery('?editable=false')).toBe(false);
  });

  it('returns null for missing or unsupported values', () => {
    expect(parseEditableQuery('')).toBeNull();
    expect(parseEditableQuery('?editable=1')).toBeNull();
    expect(parseEditableQuery('?foo=bar')).toBeNull();
  });
});

describe('resolveEditableMode', () => {
  it('defaults to disabled in dev mode', () => {
    expect(resolveEditableMode({ isDev: true, envValue: '', search: '' })).toBe(false);
  });

  it('enables mode outside dev only when VITE_ENABLE_EDITABLE_MODE=true', () => {
    expect(resolveEditableMode({ isDev: false, envValue: 'true', search: '' })).toBe(true);
    expect(resolveEditableMode({ isDev: false, envValue: 'false', search: '' })).toBe(false);
    expect(resolveEditableMode({ isDev: false, envValue: '', search: '' })).toBe(false);
  });

  it('applies query override in dev mode', () => {
    expect(resolveEditableMode({
      isDev: true,
      envValue: '',
      search: '?editable=false',
    })).toBe(false);

    expect(resolveEditableMode({
      isDev: true,
      envValue: 'false',
      search: '?editable=true',
    })).toBe(true);
  });

  it('ignores query override outside dev mode', () => {
    expect(resolveEditableMode({
      isDev: false,
      envValue: 'false',
      search: '?editable=true',
    })).toBe(false);

    expect(resolveEditableMode({
      isDev: false,
      envValue: 'true',
      search: '?editable=false',
    })).toBe(true);
  });
});
