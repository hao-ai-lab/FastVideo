/**
 * Locate and rewrite a single named field inside a raw MiniMax-H3 prompt,
 * leaving every other character of the prompt untouched. Handles both header
 * styles seen in real prompts: the heading on its own line (six-section
 * reference format) and the heading followed by its text on the same line
 * (`integrated_multimodal_description: [Shot 1] ...`, the base format).
 */

const FIELD_NAMES = [
	"subject_definitions",
	"summary",
	"retention_analysis",
	"detailed_description",
	"integrated_multimodal_description",
	"overall_soundscape",
	"non_diegetic_music",
] as const;

export type H3PromptFieldName = (typeof FIELD_NAMES)[number];

const HEADER_RE = new RegExp(`^(${FIELD_NAMES.join("|")}):[ \\t]*`, "gm");

interface Span {
	start: number;
	end: number;
}

function locate(prompt: string, field: H3PromptFieldName): Span | null {
	const headers = [...prompt.matchAll(HEADER_RE)];
	const i = headers.findIndex((h) => h[1] === field);
	if (i < 0) return null;

	const header = headers[i];
	let start = (header.index ?? 0) + header[0].length;
	// Own-line style: nothing follows the colon, so the body starts on the next line.
	const newline = prompt.slice(start).match(/^\r?\n/);
	if (newline) start += newline[0].length;

	const nextHeader = i + 1 < headers.length ? (headers[i + 1].index ?? prompt.length) : prompt.length;
	const end = start + Math.max(0, prompt.slice(start, nextHeader).trimEnd().length);
	return { start, end: Math.max(start, end) };
}

/** The field's text exactly as written, or null if the prompt has no such field. */
export function readField(prompt: string, field: H3PromptFieldName): string | null {
	const span = locate(prompt, field);
	return span ? prompt.slice(span.start, span.end) : null;
}

/** Replace only the field's text. Returns the prompt unchanged if the field isn't there. */
export function writeField(prompt: string, field: H3PromptFieldName, body: string): string {
	const span = locate(prompt, field);
	return span ? prompt.slice(0, span.start) + body + prompt.slice(span.end) : prompt;
}

export interface ShotTarget {
	/** the field the [Shot N] blocks live in, or null when the whole prompt is the shot text */
	field: "integrated_multimodal_description" | "detailed_description" | null;
	body: string;
}

/**
 * Where a prompt's shots live: the base format's integrated_multimodal_description,
 * else the reference format's detailed_description, else the prompt as a whole.
 */
export function shotTarget(prompt: string): ShotTarget {
	for (const field of ["integrated_multimodal_description", "detailed_description"] as const) {
		const body = readField(prompt, field);
		if (body !== null) return { field, body };
	}
	return { field: null, body: prompt };
}

/** Write shot text back to wherever shotTarget found it. */
export function writeShotTarget(prompt: string, target: ShotTarget, body: string): string {
	return target.field ? writeField(prompt, target.field, body) : body;
}
