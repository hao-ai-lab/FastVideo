/**
 * Turns a plain dialogue script ("NAME: line" turns) into clip-sized shot
 * lists for the shot editor. Mechanical work only -- speakers, dialogue,
 * stage-direction-to-tag mapping, grouping turns into shots, and splitting
 * the whole scene into clips that fit one generation. Camera choices are
 * defaults for the user to refine; a script says nothing about them.
 */

import { SPEECH_RATES, estimateDuration } from "@/lib/h3DialogueEstimate";
import { estimateSpokenSeconds, makeShot, newLine, newTextPart, shotLines, type Shot } from "@/lib/h3Shots";
import { H3_TAGS } from "@/lib/h3Tags";

export interface ScriptTurn {
	speaker: string;
	text: string;
}

// "MARK: ..." -- 1-3 capitalized words, so prose like "note: ..." or scene
// headings without a colon are never mistaken for a speaker.
const SPEAKER_LINE_RE = /^([A-Z][A-Za-z0-9.'_-]*(?: [A-Z][A-Za-z0-9.'_-]*){0,2}):\s*(.*)$/;

function stripMarkdown(line: string): string {
	return line
		.replace(/^\s*(?:[>*-]\s+)+/, "")
		.replace(/\*\*|__/g, "")
		.trim();
}

/**
 * A speaker line starts a turn; non-blank lines directly after it continue
 * that turn. A blank line ends the turn, so scene headings and action lines
 * between turns are ignored rather than glued onto someone's dialogue.
 */
export function parseScript(script: string): ScriptTurn[] {
	const turns: ScriptTurn[] = [];
	let continuing = false;
	for (const raw of script.split(/\r?\n/)) {
		const line = stripMarkdown(raw);
		if (!line) {
			continuing = false;
			continue;
		}
		const m = line.match(SPEAKER_LINE_RE);
		if (m) {
			turns.push({ speaker: m[1], text: m[2].trim() });
			continuing = true;
		} else if (continuing && turns.length > 0) {
			const last = turns[turns.length - 1];
			last.text = `${last.text} ${line}`.trim();
		}
	}
	return turns.filter((t) => t.text);
}

/** Speaker names in order of first appearance. */
export function scriptSpeakers(turns: ScriptTurn[]): string[] {
	const seen: string[] = [];
	for (const t of turns) if (!seen.includes(t.speaker)) seen.push(t.speaker);
	return seen;
}

/* ---------------------------------------------------------------- */
/*  Stage directions -> speech tags                                   */
/* ---------------------------------------------------------------- */

const TAG_BY_ID = Object.fromEntries(H3_TAGS.map((t) => [t.id, t]));

// Order matters: the more specific phrase must be tried before its substring.
const DIRECTION_TAGS: [RegExp, string][] = [
	[/\blong pause\b/, "long-pause"],
	[/\bcatch(?:es)? (?:his |her |their )?breath\b/, "catches-breath"],
	[/\bexhal/, "exhale"],
	[/\binhal/, "inhale"],
	[/\bsigh/, "sighs"],
	[/\blaugh/, "laughs"],
	[/\bchuckl/, "chuckle"],
	[/\bgasp/, "gasp"],
	[/\bcough/, "coughs"],
	[/\bclears? (?:his |her |their )?throat\b/, "clears-throat"],
	[/\bsniff/, "sniff"],
	[/\b(?:pause|beat)\b/, "pause"],
	[/\bstutter|\bstammer/, "stutter"],
	[/\bbreath/, "breath"],
];

export interface ConvertedLine {
	text: string;
	/** parentheticals with no matching tag, kept as stage directions for the shot description */
	directions: string[];
}

/** Rewrites "(EXHALES)"-style parentheticals as speech tags; "(WHISPERING)" wraps the whole line. */
export function convertDirections(text: string): ConvertedLine {
	const directions: string[] = [];
	let wrapWhisper = false;

	let out = text.replace(/\(([^)]*)\)/g, (_m, inner: string) => {
		const key = inner.trim().toLowerCase();
		if (/\bwhisper/.test(key)) {
			wrapWhisper = true;
			return " ";
		}
		for (const [re, id] of DIRECTION_TAGS) {
			const tag = TAG_BY_ID[id];
			if (tag && tag.kind === "point" && re.test(key)) return ` ${tag.open} `;
		}
		if (key) directions.push(key);
		return " ";
	});

	out = out
		.replace(/\s+/g, " ")
		.replace(/\s+([.,!?…])/g, "$1")
		.trim();

	const whisper = TAG_BY_ID.whisper;
	if (wrapWhisper && whisper?.close && out) out = `${whisper.open}${out}${whisper.close}`;
	return { text: out, directions };
}

/* ---------------------------------------------------------------- */
/*  Shots and clips                                                   */
/* ---------------------------------------------------------------- */

export interface ImportSpeaker {
	/** <Subject N> label this speaker is, or null when they aren't a tracked subject */
	subject: string | null;
	/** speaker tag written as "(S1)" */
	tag: string;
}

export interface ImportOptions {
	speakers: Record<string, ImportSpeaker>;
	/** rough length one clip should fit, in seconds */
	targetSeconds: number;
}

export interface ImportClip {
	shots: Shot[];
	/**
	 * True when this clip picks up in the middle of the previous clip's last
	 * shot (a speech too long for one generation). Its first shot continues
	 * that one instead of cutting to something new, so it should be generated
	 * from the previous clip's last frame. False means the boundary is a cut.
	 */
	continuation: boolean;
}

const ESTABLISHING_SECONDS = 1.5;
const REACTION_MAX_WORDS = 4;
// Space left in a clip is only worth starting a forced mid-shot split in if
// it can hold at least this much of the speech.
const MIN_PART_SECONDS = 3;

function spokenSeconds(text: string): number {
	return estimateDuration(text, SPEECH_RATES.Natural, 1).total;
}

function wordCount(text: string): number {
	return (text.replace(/<[^>]*>/g, " ").match(/[\p{L}\p{N}'’-]+/gu) ?? []).length;
}

/**
 * Split one over-long line at sentence boundaries into parts that fit the
 * given budgets (first part, then every later part). An outer wrapping tag
 * like <whisper>...</whisper> is re-applied to each part so none is left
 * with an unmatched tag.
 */
function splitToBudgets(text: string, firstBudget: number, laterBudget: number): string[] {
	const wrap = text.match(/^<([a-z ]+)>([\s\S]*)<\/\1>$/);
	const inner = wrap ? wrap[2] : text;
	const sentences = inner.match(/[^.!?…]+(?:[.!?…]+|$)\s*/g) ?? [inner];
	const parts: string[] = [];
	let current = "";
	let budget = firstBudget;
	for (const sentence of sentences) {
		const candidate = `${current}${sentence}`;
		if (current && spokenSeconds(candidate) > budget) {
			parts.push(current.trim());
			current = sentence;
			budget = laterBudget;
		} else {
			current = candidate;
		}
	}
	if (current.trim()) parts.push(current.trim());
	return wrap ? parts.map((p) => `<${wrap[1]}>${p}</${wrap[1]}>`) : parts;
}

interface Beat {
	/** the first line is the turn the shot is about; later lines are short reactions */
	lines: { speaker: string; text: string }[];
	directions: string[];
}

function beatSeconds(beat: Beat): number {
	return estimateSpokenSeconds(beat.lines.map((l) => l.text));
}

function titleCase(name: string): string {
	return name.toLowerCase().replace(/(^|[\s'-])(\p{L})/gu, (_m, sep: string, ch: string) => sep + ch.toUpperCase());
}

function joinNatural(items: string[]): string {
	return items.length <= 2 ? items.join(" and ") : `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function shotFromBeat(
	beat: Beat,
	opts: ImportOptions,
	totalSeconds: number,
	continued = false,
	truncated = false,
): Shot {
	const speakers = [...new Set(beat.lines.map((l) => l.speaker))];
	const subjectIds = [
		...new Set(speakers.map((s) => opts.speakers[s]?.subject).filter((s): s is string => Boolean(s))),
	];
	const named = speakers.map((s) => opts.speakers[s]?.subject ?? titleCase(s));

	let description: string;
	if (continued) {
		description = `Continuing the same shot on ${joinNatural(named)}, still speaking.`;
	} else {
		description =
			speakers.length > 1
				? `${joinNatural(named)} in frame, exchanging lines.`
				: `Close on ${named[0]}, speaking.`;
	}
	if (beat.directions.length) description += ` Stage direction: ${beat.directions.join("; ")}.`;

	const shot = makeShot({
		type: speakers.length > 1 ? "medium" : "close-up",
		movement: totalSeconds >= 5 ? "dolly" : "static",
		subjectIds,
		description,
		lines: beat.lines.map((l) => newLine({ speaker: opts.speakers[l.speaker]?.tag ?? "", text: l.text })),
	});
	// The guide's marker for speech truncated by the video ending: this clip
	// stops mid-speech and the next one picks it up.
	if (truncated) shot.body.push(newTextPart("<cutoff>"));
	return shot;
}

/** One shot per speaking turn; a short reaction ("Yeah.") rides along in the shot it reacts to. */
function groupBeats(turns: ScriptTurn[]): Beat[] {
	const beats: Beat[] = [];
	for (const turn of turns) {
		const { text, directions } = convertDirections(turn.text);
		if (!text) continue;
		const last = beats[beats.length - 1];
		if (last && wordCount(text) <= REACTION_MAX_WORDS) {
			last.lines.push({ speaker: turn.speaker, text });
			last.directions.push(...directions);
		} else {
			beats.push({ lines: [{ speaker: turn.speaker, text }], directions: [...directions] });
		}
	}
	return beats;
}

/**
 * The guide assigns speaker IDs by who first speaks in *each video*, so a
 * clip that opens on the second character must call them S1. Renumber a
 * clip's default S1/S2/... tags in order of first appearance; custom tags
 * (anything not of the form S<n>) are left alone.
 */
function renumberSpeakerTags(clip: ImportClip): void {
	const lines = clip.shots.flatMap((s) => shotLines(s));
	const seen: string[] = [];
	for (const l of lines) if (l.speaker && !seen.includes(l.speaker)) seen.push(l.speaker);
	if (seen.length === 0 || !seen.every((t) => /^S\d+$/.test(t))) return;
	const renamed = new Map(seen.map((t, i) => [t, `S${i + 1}`]));
	for (const l of lines) if (l.speaker) l.speaker = renamed.get(l.speaker) ?? l.speaker;
}

/**
 * Pack shots into clips that fit `targetSeconds`. Clips break only at cuts
 * (a cut should bring new information), so a turn is never divided just to
 * balance lengths. The one exception is a single turn longer than a whole
 * clip: it is split at sentence breaks, and each later piece starts a
 * `continuation` clip. The first clip opens with an establishing shot.
 */
export function buildClips(turns: ScriptTurn[], opts: ImportOptions): ImportClip[] {
	const beats = groupBeats(turns);

	const allSubjects = [
		...new Set(Object.values(opts.speakers).map((s) => s.subject).filter((s): s is string => Boolean(s))),
	];
	const establishing = makeShot({
		type: "establishing",
		subjectIds: allSubjects,
		description: `Establishing view of the setting${allSubjects.length ? ` with ${joinNatural(allSubjects)} in frame` : ""}.`,
	});

	const clips: ImportClip[] = [];
	let current: ImportClip = { shots: [establishing], continuation: false };
	let seconds = ESTABLISHING_SECONDS;
	const hasDialogue = () => current.shots.some((s) => shotLines(s).length > 0);
	const close = (continuation: boolean) => {
		clips.push(current);
		current = { shots: [], continuation };
		seconds = 0;
	};

	for (const beat of beats) {
		const total = beatSeconds(beat);

		if (total <= opts.targetSeconds) {
			if (hasDialogue() && seconds + total > opts.targetSeconds) close(false);
			current.shots.push(shotFromBeat(beat, opts, total));
			seconds += total;
			continue;
		}

		// Longer than any clip can hold: the split has to fall inside this shot.
		if (current.shots.length > 0 && opts.targetSeconds - seconds < MIN_PART_SECONDS) close(false);
		const [main, ...reactions] = beat.lines;
		const parts = splitToBudgets(main.text, opts.targetSeconds - seconds, opts.targetSeconds);
		parts.forEach((part, k) => {
			if (k > 0) close(true);
			const lines = [{ ...main, text: part }, ...(k === parts.length - 1 ? reactions : [])];
			const pieceBeat: Beat = { lines, directions: k === 0 ? beat.directions : [] };
			current.shots.push(shotFromBeat(pieceBeat, opts, total, k > 0, k < parts.length - 1));
			seconds += beatSeconds(pieceBeat);
		});
	}

	if (current.shots.length > 0 && (clips.length === 0 || hasDialogue())) clips.push(current);
	for (const clip of clips) renumberSpeakerTags(clip);
	return clips;
}
