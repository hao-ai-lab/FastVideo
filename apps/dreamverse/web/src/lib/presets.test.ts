import { describe, expect, it } from 'vitest';

import {
  parseStoryPresets,
  sanitizePresetId,
} from './presets';

describe('sanitizePresetId', () => {
  it('normalizes case, spacing, and symbols', () => {
    expect(sanitizePresetId('  My Preset #1  ')).toBe('my_preset_1');
  });

  it('falls back to custom_editable when empty', () => {
    expect(sanitizePresetId('___')).toBe('custom_editable');
    expect(sanitizePresetId('')).toBe('custom_editable');
  });
});

describe('parseStoryPresets', () => {
  it('keeps only valid entries and trims text fields', () => {
    const parsed = parseStoryPresets([
      {
        id: ' preset_one ',
        label: ' Preset One ',
        segment_prompts: [' first ', ' second '],
      },
      {
        id: '',
        label: 'Missing id',
        segment_prompts: ['a', 'b'],
      },
      {
        id: 'only_one_prompt',
        label: 'Too short',
        segment_prompts: ['a'],
      },
      {
        id: 'blank_prompt',
        label: 'Blank prompt',
        segment_prompts: ['a', '   '],
      },
    ]);

    expect(parsed).toEqual([
      {
        id: 'preset_one',
        label: 'Preset One',
        segment_prompts: ['first', 'second'],
      },
    ]);
  });

  it('returns empty array for invalid raw preset payload', () => {
    expect(parseStoryPresets(null)).toEqual([]);
    expect(parseStoryPresets({})).toEqual([]);
  });

  it('supports wrapped default export payloads', () => {
    const parsed = parseStoryPresets({
      default: [
        {
          id: 'example',
          label: 'Example',
          segment_prompts: ['a', 'b'],
        },
      ],
    });

    expect(parsed).toEqual([
      {
        id: 'example',
        label: 'Example',
        segment_prompts: ['a', 'b'],
      },
    ]);
  });
});
