import { describe, expect, it } from 'vitest';

import {
  parseDevtoolsQuery,
  resolveDevtoolsMode,
} from './devtoolsMode';

describe('parseDevtoolsQuery', () => {
  it('parses truthy query values', () => {
    expect(parseDevtoolsQuery('?devtools')).toBe(true);
    expect(parseDevtoolsQuery('?devtools=1')).toBe(true);
    expect(parseDevtoolsQuery('?devtools=true')).toBe(true);
  });

  it('parses falsy query values', () => {
    expect(parseDevtoolsQuery('?devtools=0')).toBe(false);
    expect(parseDevtoolsQuery('?devtools=false')).toBe(false);
  });

  it('returns null for missing or unsupported values', () => {
    expect(parseDevtoolsQuery('')).toBeNull();
    expect(parseDevtoolsQuery('?devtools=maybe')).toBeNull();
  });
});

describe('resolveDevtoolsMode', () => {
  it('requires a devtools-capable build', () => {
    expect(resolveDevtoolsMode({
      buildEnabled: false,
      search: '?devtools=1',
    })).toBe(false);
  });

  it('activates only for explicit truthy query values', () => {
    expect(resolveDevtoolsMode({
      buildEnabled: true,
      search: '',
    })).toBe(false);
    expect(resolveDevtoolsMode({
      buildEnabled: true,
      search: '?devtools=1',
    })).toBe(true);
  });
});
