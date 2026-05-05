import { describe, expect, it } from 'vitest';

import {
  buildPromptWindowSnapshot,
  buildRewritePromptWindowSnapshot,
  buildRewritePromptWindowSnapshotFromPrompts,
  normalizePromptWindowSnapshot,
} from './promptWindowSnapshot';

describe('normalizePromptWindowSnapshot', () => {
  it('trims prompts and removes empty entries while preserving order', () => {
    expect(normalizePromptWindowSnapshot([
      ' first ',
      '',
      '   ',
      'second',
      ' third  ',
    ])).toEqual([
      'first',
      'second',
      'third',
    ]);
  });

  it('returns an empty array for non-array input', () => {
    expect(normalizePromptWindowSnapshot(null)).toEqual([]);
    expect(normalizePromptWindowSnapshot({})).toEqual([]);
  });
});

describe('buildPromptWindowSnapshot', () => {
  it('builds the payload from currentPromptWindowPrompts only', () => {
    expect(buildPromptWindowSnapshot({
      currentPromptWindowPrompts: [' current 1 ', 'current 2'],
      seedPrompts: ['seed 1', 'seed 2'],
      outboundSessionPrompts: ['outbound 1', 'outbound 2'],
    })).toEqual([
      'current 1',
      'current 2',
    ]);
  });
});

describe('buildRewritePromptWindowSnapshot', () => {
  it('pads short prompt windows to 6 entries', () => {
    const result = buildRewritePromptWindowSnapshot({
      currentPromptWindowPrompts: ['prompt 1', 'prompt 2'],
    });
    expect(result).toHaveLength(6);
    expect(result[0]).toBe('prompt 1');
    expect(result[1]).toBe('prompt 2');
    for (let i = 2; i < 6; i++) {
      expect(result[i]).toBeTruthy();
    }
  });

  it('leaves a 6-element window unchanged', () => {
    const prompts = ['a', 'b', 'c', 'd', 'e', 'f'];
    const result = buildRewritePromptWindowSnapshot({
      currentPromptWindowPrompts: prompts,
    });
    expect(result).toEqual(prompts);
  });

  it('returns empty array when snapshot is empty', () => {
    expect(buildRewritePromptWindowSnapshot({
      currentPromptWindowPrompts: [],
    })).toEqual([]);
  });
});

describe('buildRewritePromptWindowSnapshotFromPrompts', () => {
  it('pads a direct prompt list the same way as store-backed snapshots', () => {
    const result = buildRewritePromptWindowSnapshotFromPrompts([
      ' selected 1 ',
      'selected 2',
    ]);
    expect(result).toHaveLength(6);
    expect(result[0]).toBe('selected 1');
    expect(result[1]).toBe('selected 2');
  });
});
