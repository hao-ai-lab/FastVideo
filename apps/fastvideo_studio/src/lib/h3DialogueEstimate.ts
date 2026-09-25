/**
 * Ballpark spoken-duration estimate for MiniMax-H3 dialogue text, plus the
 * lexer/tree used to render a styled "performance preview" of the tags.
 *
 * Numbers here are heuristic (syllable counting + fixed per-tag seconds from
 * `H3_TAGS[].seconds`), not measured against the model -- good enough to
 * flag "this clip is clearly too long for one generation," calibrated
 * against a real render via `calibrationFactor`.
 */

import { H3_TAGS, type H3Tag } from "@/lib/h3Tags";

/** Bare tag name, e.g. "<long pause>" -> "long pause". */
function bareName(bracketed: string): string {
	return bracketed.replace(/[<>/]/g, "").trim();
}

const WRAP_TAG_NAMES = new Set(H3_TAGS.filter((t) => t.kind === "wrap").map((t) => bareName(t.open)));
const TAG_SECONDS: Record<string, number> = Object.fromEntries(
	H3_TAGS.map((t) => [bareName(t.open), t.seconds ?? 0]),
);
/** Bare tag name -> its registry entry, so callers (the preview renderer) can look up category/description. */
export const TAG_BY_NAME: Record<string, H3Tag> = Object.fromEntries(H3_TAGS.map((t) => [bareName(t.open), t]));

/* -------------------------------------------------------------------- */
/*  Lexer / tree — mirrors the tag registry so preview & estimate always  */
/*  agree with what the console/editor can actually insert.               */
/* -------------------------------------------------------------------- */

export type LexToken =
	| { t: "text"; v: string }
	| { t: "point"; name: string }
	| { t: "open"; name: string }
	| { t: "close"; name: string };

const TAG_RE = /<(\/?)([a-zA-Z ]+?)>/g;

export function lexDialogue(text: string): LexToken[] {
	const tokens: LexToken[] = [];
	let last = 0;
	let m: RegExpExecArray | null;
	TAG_RE.lastIndex = 0;
	while ((m = TAG_RE.exec(text))) {
		if (m.index > last) tokens.push({ t: "text", v: text.slice(last, m.index) });
		const closing = m[1] === "/";
		const name = m[2].trim();
		if (WRAP_TAG_NAMES.has(name)) tokens.push({ t: closing ? "close" : "open", name });
		else if (!closing) tokens.push({ t: "point", name }); // stray/unknown closers are ignored
		last = TAG_RE.lastIndex;
	}
	if (last < text.length) tokens.push({ t: "text", v: text.slice(last) });
	return tokens;
}

export interface DialogueTextNode {
	t: "text";
	v: string;
}
export interface DialoguePointNode {
	t: "point";
	name: string;
}
export interface DialogueSpanNode {
	t: "span";
	name: string;
	children: DialogueNode[];
}
export type DialogueNode = DialogueTextNode | DialoguePointNode | DialogueSpanNode;

export function dialogueTree(text: string): DialogueNode[] {
	const root: { children: DialogueNode[] } = { children: [] };
	const stack: { children: DialogueNode[] }[] = [root];
	for (const tk of lexDialogue(text)) {
		const top = stack[stack.length - 1];
		if (tk.t === "text") top.children.push({ t: "text", v: tk.v });
		else if (tk.t === "point") top.children.push({ t: "point", name: tk.name });
		else if (tk.t === "open") {
			const node: DialogueSpanNode = { t: "span", name: tk.name, children: [] };
			top.children.push(node);
			stack.push(node);
		} else if (tk.t === "close" && stack.length > 1) stack.pop();
	}
	return root.children;
}

/* -------------------------------------------------------------------- */
/*  Duration estimate                                                    */
/* -------------------------------------------------------------------- */

export const SPEECH_RATES = { Measured: 3.2, Natural: 4.0, Quick: 4.8 } as const; // syllables/sec
export type SpeechRate = keyof typeof SPEECH_RATES;

export function syllables(word: string): number {
	let w = word.toLowerCase().replace(/[^a-z]/g, "");
	if (!w) return 0;
	if (w.length <= 3) return 1;
	w = w.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, "");
	w = w.replace(/^y/, "");
	const m = w.match(/[aeiouy]{1,2}/g);
	return m ? m.length : 1;
}

export interface DurationEstimate {
	total: number;
	speech: number;
	punct: number;
	tags: number;
	syl: number;
	words: number;
	low: number;
	high: number;
}

export function estimateDuration(text: string, rate: number, calib: number): DurationEstimate {
	const spoken = text.replace(/<\/?[a-zA-Z ]+?>/g, " ");
	const words = spoken.split(/\s+/).filter(Boolean);
	const syl = words.reduce((n, w) => n + syllables(w), 0);
	const speech = syl / rate;

	const sentence = (spoken.match(/[.!?]/g) || []).length * 0.35;
	const commas = (spoken.match(/[,;:]/g) || []).length * 0.18;
	const ellipsis = (spoken.match(/\.\.\.|…/g) || []).length * 0.5;
	const dashes = (spoken.match(/—/g) || []).length * 0.2;
	const punct = sentence + commas + ellipsis + dashes;

	let tags = 0;
	const re = /<(\/?)([a-zA-Z ]+?)>/g;
	let m: RegExpExecArray | null;
	while ((m = re.exec(text))) {
		if (m[1] === "/") continue;
		tags += TAG_SECONDS[m[2].trim()] ?? 0;
	}

	const total = (speech + punct + tags) * calib;
	return {
		total,
		speech: speech * calib,
		punct: punct * calib,
		tags: tags * calib,
		syl,
		words: words.length,
		low: total * 0.86,
		high: total * 1.16,
	};
}

/** Find the best sentence/pause boundary at which to split text that runs past `ceiling` seconds. */
export function suggestSplit(text: string, rate: number, calib: number, ceiling: number): number {
	const re = /(<\/?[a-zA-Z ]+?>)|([.!?]+)|([,;:]+)|(\s+)|([^\s<.!?,;:]+)/g;
	let cum = 0;
	let boundary = 0;
	let m: RegExpExecArray | null;
	const budget = ceiling - 1.5;
	while ((m = re.exec(text))) {
		if (m[1]) {
			const closing = m[1].startsWith("</");
			const name = bareName(m[1]);
			if (!closing) cum += (TAG_SECONDS[name] ?? 0) * calib;
			if (name === "pause" || name === "long pause") boundary = re.lastIndex;
		} else if (m[2]) {
			cum += 0.35 * calib;
			boundary = re.lastIndex;
		} else if (m[3]) {
			cum += 0.18 * calib;
		} else if (m[5]) {
			cum += (syllables(m[5]) / rate) * calib;
		}
		if (cum > budget && boundary > 0) return boundary;
	}
	return boundary > 0 ? boundary : -1;
}
