/**
 * MiniMax-H3 speech tags: the inline markers the model recognizes inside
 * `<d>[English] dialogue</d>` blocks (and elsewhere in a prompt) to shape
 * delivery -- pauses, breath, emphasis, and non-verbal sounds. `applyH3Tag`
 * is the pure insertion logic; the tag/category tables drive the console UI.
 */

export type H3TagKind = "point" | "wrap";

export interface H3Tag {
	/** stable key for React lists and lookups */
	id: string;
	/** short button label, usually the bare tag name */
	label: string;
	description: string;
	/** short usage example shown as a tooltip */
	example: string;
	kind: H3TagKind;
	/** point: the literal token inserted, e.g. "<pause>". wrap: the opening tag. */
	open: string;
	/** wrap only: the closing tag, e.g. "</i>". */
	close?: string;
	/**
	 * Rough seconds this tag adds to spoken duration, used only for the
	 * dialogue editor's length estimate -- ballpark figures, not measured.
	 * Delivery-only tags (emphasis, whisper, softer) add ~0.
	 */
	seconds?: number;
}

export interface H3TagCategory {
	id: string;
	label: string;
	tags: H3Tag[];
}

export const H3_TAG_CATEGORIES: H3TagCategory[] = [
	{
		id: "pace",
		label: "Pace & breath",
		tags: [
			{
				id: "pause",
				label: "<pause>",
				description: "Short pause",
				example: "Okay, so. <pause> This is just me talking.",
				kind: "point",
				open: "<pause>",
				seconds: 0.45,
			},
			{
				id: "long-pause",
				label: "<long pause>",
				description: "Longer pause",
				example: "I mean... <long pause> I don't even know.",
				kind: "point",
				open: "<long pause>",
				seconds: 1.2,
			},
			{
				id: "breath",
				label: "<breath>",
				description: "Breathing sound",
				example: "And then... <breath> it just happened.",
				kind: "point",
				open: "<breath>",
				seconds: 0.55,
			},
			{
				id: "inhale",
				label: "<inhale>",
				description: "In-breath",
				example: "<inhale> Alright, let's do this.",
				kind: "point",
				open: "<inhale>",
				seconds: 0.5,
			},
			{
				id: "exhale",
				label: "<exhale>",
				description: "Out-breath",
				example: "<exhale> Okay then.",
				kind: "point",
				open: "<exhale>",
				seconds: 0.6,
			},
			{
				id: "catches-breath",
				label: "<catches breath>",
				description: "Out of breath",
				example: "Wait... <catches breath> hold on a sec.",
				kind: "point",
				open: "<catches breath>",
				seconds: 0.8,
			},
			{
				id: "deep-breath",
				label: "<deep breath>",
				description: "Calming down",
				example: "<deep breath> Okay. I can do this.",
				kind: "point",
				open: "<deep breath>",
				seconds: 1.2,
			},
			{
				id: "pant",
				label: "<pant>",
				description: "Panting (single)",
				example: "Run... <pant> run now!",
				kind: "point",
				open: "<pant>",
				seconds: 0.7,
			},
			{
				id: "pants",
				label: "<pants>",
				description: "Panting (continued)",
				example: "Run... <pants> run now!",
				kind: "point",
				open: "<pants>",
				seconds: 0.7,
			},
		],
	},
	{
		id: "delivery",
		label: "Emphasis & delivery",
		tags: [
			{
				id: "emphasis",
				label: "<i>…</i>",
				description: "Emphasize the wrapped words (1-4 words)",
				example: "I was <i>not</i> expecting that.",
				kind: "wrap",
				open: "<i>",
				close: "</i>",
				seconds: 0,
			},
			{
				id: "whisper",
				label: "<whisper>…</whisper>",
				description: "Whisper delivery",
				example: "<whisper>Don't tell anyone this.</whisper>",
				kind: "wrap",
				open: "<whisper>",
				close: "</whisper>",
				seconds: 0,
			},
			{
				id: "softer",
				label: "<softer>",
				description: "Quieter delivery",
				example: "<softer> I don't think I can say it.",
				kind: "point",
				open: "<softer>",
				seconds: 0,
			},
			{
				id: "humming",
				label: "<humming>…</humming>",
				description: "Humming a tune",
				example: "<humming>da-da-da-beautiful-day.</humming>",
				kind: "wrap",
				open: "<humming>",
				close: "</humming>",
				seconds: 2,
			},
		],
	},
	{
		id: "vocal",
		label: "Vocalizations",
		tags: [
			{
				id: "laughs",
				label: "<laughs>",
				description: "Laughing",
				example: "That's... <laughs> that's actually funny.",
				kind: "point",
				open: "<laughs>",
				seconds: 1.1,
			},
			{
				id: "chuckle",
				label: "<chuckle>",
				description: "Laughing (light)",
				example: "That's... <chuckle> that's actually funny.",
				kind: "point",
				open: "<chuckle>",
				seconds: 0.7,
			},
			{
				id: "sighs",
				label: "<sighs>",
				description: "Sigh",
				example: "<sighs> I really tried.",
				kind: "point",
				open: "<sighs>",
				seconds: 0.85,
			},
			{
				id: "gasp",
				label: "<gasp>",
				description: "Sharp intake",
				example: "<gasp> Oh my God.",
				kind: "point",
				open: "<gasp>",
				seconds: 0.5,
			},
			{
				id: "phew",
				label: "<phew>",
				description: "Relief",
				example: "<phew> That was close.",
				kind: "point",
				open: "<phew>",
				seconds: 0.6,
			},
			{
				id: "mhm",
				label: "<mhm>",
				description: "Agreement sound",
				example: "Yeah, <mhm> exactly.",
				kind: "point",
				open: "<mhm>",
				seconds: 0.4,
			},
			{
				id: "uh",
				label: "<uh>",
				description: "Filler / hesitation",
				example: "So, like... <uh> what was I saying?",
				kind: "point",
				open: "<uh>",
				seconds: 0.35,
			},
		],
	},
	{
		id: "body",
		label: "Disfluency & body",
		tags: [
			{
				id: "stutter",
				label: "<stutter>",
				description: "Stutters the words that follow",
				example: "<stutter> I ca can't believe that.",
				kind: "point",
				open: "<stutter>",
				seconds: 0.45,
			},
			{
				id: "coughs",
				label: "<coughs>",
				description: "Cough",
				example: "<coughs> Sorry, one sec.",
				kind: "point",
				open: "<coughs>",
				seconds: 0.6,
			},
			{
				id: "clears-throat",
				label: "<clears throat>",
				description: "Throat clear",
				example: "<clears throat> So anyway...",
				kind: "point",
				open: "<clears throat>",
				seconds: 0.8,
			},
			{
				id: "sniff",
				label: "<sniff>",
				description: "Sniffing",
				example: "<sniff> It's just... really sad.",
				kind: "point",
				open: "<sniff>",
				seconds: 0.45,
			},
			{
				id: "smacks-lips",
				label: "<smacks lips>",
				description:
					"Lip smack; leave empty to trigger at the end of the sentence, or wrap words to place it there",
				example: "<smacks lips></smacks lips> Okay.",
				kind: "wrap",
				open: "<smacks lips>",
				close: "</smacks lips>",
				seconds: 0.3,
			},
		],
	},
];

export const H3_TAGS: H3Tag[] = H3_TAG_CATEGORIES.flatMap((c) => c.tags);

export interface H3TagApplyResult {
	value: string;
	/** selection to restore focus to after the insertion (collapsed unless text was wrapped). */
	start: number;
	end: number;
}

function needsLeadingSpace(before: string): boolean {
	return before.length > 0 && !/\s$/.test(before);
}

function needsTrailingSpace(after: string): boolean {
	return after.length > 0 && !/^\s/.test(after);
}

/**
 * Insert `tag` into `value` at the given selection.
 *
 * Point tags are inserted at the selection start, padded with a space on
 * either side where the surrounding text doesn't already have one, and the
 * cursor collapses to just past the inserted token.
 *
 * Wrap tags wrap the current selection (cursor lands just past the closing
 * tag), or insert an empty `open+close` pair with the cursor left between
 * them when nothing is selected, ready to type into.
 */
export function applyH3Tag(
	value: string,
	selectionStart: number,
	selectionEnd: number,
	tag: H3Tag,
): H3TagApplyResult {
	const start = Math.max(0, Math.min(selectionStart, value.length));
	const end = Math.max(start, Math.min(selectionEnd, value.length));
	const before = value.slice(0, start);
	const selected = value.slice(start, end);
	const after = value.slice(end);

	if (tag.kind === "point") {
		const lead = needsLeadingSpace(before) ? " " : "";
		const trailingContext = selected.length > 0 ? selected : after;
		const trail = needsTrailingSpace(trailingContext) ? " " : "";
		const token = `${lead}${tag.open}${trail}`;
		const cursor = before.length + token.length;
		return {
			value: before + token + selected + after,
			start: cursor,
			end: cursor + selected.length,
		};
	}

	const close = tag.close ?? "";
	if (selected.length > 0) {
		const cursor = before.length + tag.open.length + selected.length + close.length;
		return {
			value: before + tag.open + selected + close + after,
			start: cursor,
			end: cursor,
		};
	}
	const cursor = before.length + tag.open.length;
	return {
		value: before + tag.open + close + after,
		start: cursor,
		end: cursor,
	};
}

/** One-click bundles of tags for common delivery beats (dialogue editor + shot list). */
export interface H3Preset {
	label: string;
	/** literal text inserted at the cursor (a leading space is added when needed). */
	insert?: string;
	/** wrap the selection between these two tokens instead of inserting. */
	wrap?: [string, string];
}

export const H3_PRESETS: H3Preset[] = [
	{ label: "Cracking", insert: "<catches breath> <stutter> " },
	{ label: "Nervous", insert: "<uh> <stutter> " },
	{ label: "Relief", insert: "<phew> <exhale> " },
	{ label: "Cold turn", insert: "<long pause> <softer> " },
	{ label: "Intimate", wrap: ["<whisper>", "</whisper>"] },
	{ label: "Amused", insert: "<chuckle> " },
];

/** Pure counterpart of `applyH3Tag` for presets: same {value, selection} contract. */
export function applyH3Preset(
	value: string,
	selectionStart: number,
	selectionEnd: number,
	preset: H3Preset,
): H3TagApplyResult {
	const start = Math.max(0, Math.min(selectionStart, value.length));
	const end = Math.max(start, Math.min(selectionEnd, value.length));
	const before = value.slice(0, start);
	const mid = value.slice(start, end);
	const after = value.slice(end);

	if (preset.wrap) {
		const [open, close] = preset.wrap;
		const innerStart = before.length + open.length;
		return {
			value: before + open + mid + close + after,
			start: innerStart,
			end: innerStart + mid.length,
		};
	}

	const lead = before.length && !/\s$/.test(before) ? " " : "";
	const chunk = lead + (preset.insert ?? "");
	const caret = before.length + chunk.length;
	return { value: before + chunk + after, start: caret, end: caret };
}
