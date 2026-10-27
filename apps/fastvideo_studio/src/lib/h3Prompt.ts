/**
 * MiniMax-H3 full-reference prompt sections. Serialization follows the worked
 * example in the model's VIDEO_PROMPT_WRITING_GUIDE_ref_en.md: `section_name:`
 * on its own line, content flush left, one blank line between sections.
 */

export const H3_PROMPT_SECTIONS = [
  'subject_definitions',
  'summary',
  'retention_analysis',
  'detailed_description',
  'overall_soundscape',
  'non_diegetic_music',
] as const;

export type H3PromptSection = (typeof H3_PROMPT_SECTIONS)[number];

export type H3PromptFields = Record<H3PromptSection, string>;

export const EMPTY_H3_PROMPT_FIELDS: H3PromptFields = {
  subject_definitions: '',
  summary: '',
  retention_analysis: '',
  detailed_description: '',
  overall_soundscape: '',
  non_diegetic_music: '',
};

export const H3_SECTION_LABELS: Record<H3PromptSection, string> = {
  subject_definitions: 'Subject definitions',
  summary: 'Summary',
  retention_analysis: 'Retention analysis',
  detailed_description: 'Detailed description',
  overall_soundscape: 'Overall soundscape',
  non_diegetic_music: 'Non-diegetic music',
};

/** Per-section guidance, condensed from the guide's rules for each section. */
export const H3_SECTION_HINTS: Record<H3PromptSection, string> = {
  subject_definitions:
    'One line per <Subject N>. A subject may draw on several references, e.g. "<Subject 2> is the dog in <Picture 2>, <Picture 3>, and <Picture 4>."',
  summary:
    'One paragraph, starting with a [task type] prefix: reference generation, keyframe completion, video editing, video continuation, audio reuse, audio reference. Combine with " + ".',
  retention_analysis:
    'Per subject: "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - what is retained."',
  detailed_description:
    'Playback order, using [Shot N] markers, timestamps, (S1) speaker tags and <d>[English] dialogue</d>.',
  overall_soundscape: 'Ambience and physical sounds. N/A if none.',
  non_diegetic_music:
    'Background music audible only to the audience. N/A if none.',
};

/** True when every section is blank. */
export function isEmptyPromptFields(fields: H3PromptFields): boolean {
  return H3_PROMPT_SECTIONS.every((s) => !fields[s].trim());
}

/**
 * Join the sections into the prompt string the model is given. Blank sections
 * become "N/A" rather than being dropped, matching the guide's example.
 */
export function serializeH3Prompt(fields: H3PromptFields): string {
  return H3_PROMPT_SECTIONS.map((section) => {
    const body = fields[section].trim() || 'N/A';
    return `${section}:\n${body}`;
  }).join('\n\n');
}

/**
 * Split a serialized prompt back into sections, so switching between the
 * guided fields and the raw editor does not lose work. Returns null when the
 * text is not in section format (a plain prompt, say).
 */
export function parseH3Prompt(text: string): H3PromptFields | null {
  const fields = { ...EMPTY_H3_PROMPT_FIELDS };
  const headings = new Set<string>(H3_PROMPT_SECTIONS);
  let current: H3PromptSection | null = null;
  let found = false;

  for (const line of text.split('\n')) {
    const heading = line.trim().replace(/:$/, '');
    if (line.trim().endsWith(':') && headings.has(heading)) {
      current = heading as H3PromptSection;
      found = true;
      continue;
    }
    if (current) fields[current] += (fields[current] ? '\n' : '') + line;
  }
  if (!found) return null;
  for (const section of H3_PROMPT_SECTIONS) {
    fields[section] = fields[section].trim();
  }
  return fields;
}
