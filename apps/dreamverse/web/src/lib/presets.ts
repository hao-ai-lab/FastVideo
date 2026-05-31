export const DEFAULT_CUSTOM_PRESET_ID = 'custom_editable';

export function sanitizePresetId(value: string | null | undefined): string {
  const normalized = (value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return normalized || DEFAULT_CUSTOM_PRESET_ID;
}

export interface StoryPreset {
  [key: string]: unknown;
  id: string;
  label: string;
  description?: string;
  segment_prompts: string[];
}

export function parseStoryPresets(rawPresets: any): StoryPreset[] {
  const source: any[] = Array.isArray(rawPresets)
    ? rawPresets
    : Array.isArray(rawPresets?.default)
      ? rawPresets.default
      : [];

  const parsed: StoryPreset[] = [];
  for (const entry of source) {
    const id = (entry?.id || '').trim();
    const label = (entry?.label || '').trim();
    const prompts = entry?.segment_prompts;

    if (
      id &&
      label &&
      Array.isArray(prompts) &&
      prompts.length >= 2 &&
      prompts.every((prompt: any) => typeof prompt === 'string' && prompt.trim())
    ) {
      parsed.push({
        id,
        label,
        description: typeof entry?.description === 'string' ? entry.description.trim() : undefined,
        segment_prompts: prompts.map((prompt: string) => prompt.trim()),
      });
    }
  }

  return parsed;
}
