/**
 * MiniMax-H3 shot list: a structured camera-direction editor that serializes
 * into `detailed_description`'s `[Shot N]` blocks and keeps
 * `retention_analysis`'s "appears in [...]" lines in sync, per the model's
 * reference prompt guide (see h3Prompt.ts / h3References.ts for the six
 * guided sections this plugs into).
 */

import { SPEECH_RATES, estimateDuration } from "@/lib/h3DialogueEstimate";

export type ShotType =
	| "establishing"
	| "wide"
	| "medium"
	| "close-up"
	| "extreme-close-up"
	| "over-the-shoulder"
	| "pov"
	| "insert";

export type CameraMovement =
	| "static"
	| "pan"
	| "tilt"
	| "dolly"
	| "zoom"
	| "handheld"
	| "tracking"
	| "crane";

export interface DialogueLine {
	/** stable key for React lists -- not part of the serialized prompt */
	id: string;
	/** speaker tag without parens, e.g. "S1" -- blank omits the "(S1)" prefix */
	speaker: string;
	/** spoken text only (speech tags allowed); wrapped in <d>[language] ...</d> on serialize */
	text: string;
	/** language marker written inside <d>; recovered on parse so non-English dialogue round-trips */
	language: string;
}

/** A run of prose inside a shot (composition, action, environment, delivery notes). */
export interface TextPart {
	kind: "text";
	/** stable key for React lists -- not part of the serialized prompt */
	id: string;
	text: string;
}

/** One spoken line, written as `(S1) <d>[English] ...</d>`. Its position among the parts is where it plays. */
export interface LinePart {
	kind: "line";
	line: DialogueLine;
}

export type ShotPart = TextPart | LinePart;

export interface Shot {
	/** stable key for React lists -- not part of the serialized prompt */
	id: string;
	type: ShotType;
	movement: CameraMovement;
	/** <Subject N> labels (from subject_definitions) that appear in this shot */
	subjectIds: string[];
	/**
	 * The shot's content in the order it reads: prose and spoken lines
	 * interleaved. This one list is the only record of where each line falls,
	 * so a `<d>` block stays mid-sentence if that is where it was written.
	 */
	body: ShotPart[];
	/** cut-in time in seconds; null = computed from the preceding shots. Ignored for the opening shot. */
	startSeconds: number | null;
	/**
	 * The description carries its own lead-in (text opened from a hand-written
	 * prompt) so type/movement are not written and the wording is kept verbatim.
	 */
	freeLead: boolean;
}

export const SHOT_TYPES: { id: ShotType; label: string }[] = [
	{ id: "establishing", label: "Establishing" },
	{ id: "wide", label: "Wide" },
	{ id: "medium", label: "Medium" },
	{ id: "close-up", label: "Close-up" },
	{ id: "extreme-close-up", label: "Extreme close-up" },
	{ id: "over-the-shoulder", label: "Over-the-shoulder" },
	{ id: "pov", label: "POV" },
	{ id: "insert", label: "Insert" },
];

export const CAMERA_MOVEMENTS: { id: CameraMovement; label: string }[] = [
	{ id: "static", label: "Static" },
	{ id: "pan", label: "Pan" },
	{ id: "tilt", label: "Tilt" },
	{ id: "dolly", label: "Dolly" },
	{ id: "zoom", label: "Zoom" },
	{ id: "handheld", label: "Handheld" },
	{ id: "tracking", label: "Tracking" },
	{ id: "crane", label: "Crane" },
];

const SHOT_TYPE_LABEL = Object.fromEntries(SHOT_TYPES.map((t) => [t.id, t.label])) as Record<ShotType, string>;
const MOVEMENT_LABEL = Object.fromEntries(CAMERA_MOVEMENTS.map((m) => [m.id, m.label])) as Record<
	CameraMovement,
	string
>;
const SHOT_TYPE_BY_LABEL = Object.fromEntries(SHOT_TYPES.map((t) => [t.label.toLowerCase(), t.id])) as Record<
	string,
	ShotType
>;
const MOVEMENT_BY_LABEL = Object.fromEntries(
	CAMERA_MOVEMENTS.map((m) => [m.label.toLowerCase(), m.id]),
) as Record<string, CameraMovement>;

let shotSeq = 0;
export function newShot(): Shot {
	shotSeq += 1;
	return {
		id: `shot-${Date.now()}-${shotSeq}`,
		type: "medium",
		movement: "static",
		subjectIds: [],
		body: [newTextPart()],
		startSeconds: null,
		freeLead: false,
	};
}

let partSeq = 0;
export function newTextPart(text = ""): TextPart {
	partSeq += 1;
	return { kind: "text", id: `text-${Date.now()}-${partSeq}`, text };
}

export function linePart(line: DialogueLine): LinePart {
	return { kind: "line", line };
}

/** Every spoken line of the shot, in playback order. */
export function shotLines(shot: Shot): DialogueLine[] {
	return shot.body.flatMap((p) => (p.kind === "line" ? [p.line] : []));
}

/** The shot's prose with the dialogue left out. */
export function shotProse(shot: Shot): string {
	return shot.body
		.flatMap((p) => (p.kind === "text" ? [p.text.trim()] : []))
		.filter(Boolean)
		.join(" ");
}

/** A body of one prose part followed by the given lines -- the common "description, then dialogue" shape. */
export function makeBody(description = "", lines: DialogueLine[] = []): ShotPart[] {
	return [newTextPart(description), ...lines.map(linePart)];
}

export interface ShotInit extends Partial<Omit<Shot, "id" | "body">> {
	description?: string;
	lines?: DialogueLine[];
	body?: ShotPart[];
}

/** Build a shot from a description and lines (or an explicit body). */
export function makeShot(init: ShotInit = {}): Shot {
	const { description, lines, body, ...rest } = init;
	return { ...newShot(), ...rest, body: body ?? makeBody(description ?? "", lines ?? []) };
}

let lineSeq = 0;
export function newLine(patch: Partial<Omit<DialogueLine, "id">> = {}): DialogueLine {
	lineSeq += 1;
	return { id: `line-${Date.now()}-${lineSeq}`, speaker: "", text: "", language: "English", ...patch };
}

/** Copy shots with brand-new ids so components keyed by id (which own their own text state) remount instead of showing stale edits. */
export function withFreshIds(shots: Shot[]): Shot[] {
	return shots.map((s) => ({
		...s,
		id: newShot().id,
		subjectIds: [...s.subjectIds],
		body: s.body.map((p) =>
			p.kind === "text" ? { ...p, id: newTextPart().id } : { kind: "line" as const, line: { ...p.line, id: newLine().id } },
		),
	}));
}

/** Ordered, de-duplicated <Subject N> labels mentioned in subject_definitions. */
export function parseSubjectLabels(subjectDefinitions: string): string[] {
	const seen: string[] = [];
	const re = /<Subject\s+\d+>/g;
	let m: RegExpExecArray | null;
	while ((m = re.exec(subjectDefinitions))) {
		if (!seen.includes(m[0])) seen.push(m[0]);
	}
	return seen;
}

/* ---------------------------------------------------------------- */
/*  Cut times                                                         */
/* ---------------------------------------------------------------- */

const MIN_SHOT_SECONDS = 2;
const LINE_GAP_SECONDS = 0.5;

/** Ballpark seconds a shot with these spoken lines runs; drives auto cut times. */
export function estimateSpokenSeconds(texts: string[]): number {
	const spoken = texts.filter((t) => t.trim());
	const speech = spoken.reduce((n, t) => n + estimateDuration(t, SPEECH_RATES.Natural, 1).total, 0);
	return Math.max(MIN_SHOT_SECONDS, speech + (spoken.length ? LINE_GAP_SECONDS : 0));
}

export function estimateShotSeconds(shot: Shot): number {
	return estimateSpokenSeconds(shotLines(shot).map((l) => l.text));
}

/**
 * Cut-in time of every shot: the opening shot is 0, later ones use their
 * explicit time or follow the previous shot's estimated length. Times are
 * kept strictly increasing, as the guide requires.
 */
export function resolveShotStarts(shots: Shot[]): number[] {
	const starts: number[] = [];
	shots.forEach((shot, i) => {
		if (i === 0) {
			starts.push(0);
			return;
		}
		const prev = starts[i - 1];
		let start = shot.startSeconds ?? prev + estimateShotSeconds(shots[i - 1]);
		if (start <= prev) start = prev + 0.1;
		starts.push(Math.round(start * 1000) / 1000);
	});
	return starts;
}

/** MM:SS.mmm, the format the guide uses for cut times ("At 00:03.500, ..."). */
export function formatCutTime(seconds: number): string {
	const totalMs = Math.max(0, Math.round(seconds * 1000));
	const pad = (n: number, w: number) => String(n).padStart(w, "0");
	const totalSec = Math.floor(totalMs / 1000);
	return `${pad(Math.floor(totalSec / 60), 2)}:${pad(totalSec % 60, 2)}.${pad(totalMs % 1000, 3)}`;
}

/** Accepts "3.5", "0:03.5", or "00:03.500"; returns seconds, or null if unreadable. */
export function parseCutTime(text: string): number | null {
	const m = text.trim().match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/);
	if (!m) return null;
	return (m[1] ? Number(m[1]) * 60 : 0) + Number(m[2]);
}

/* ---------------------------------------------------------------- */
/*  Serialize                                                         */
/* ---------------------------------------------------------------- */

/** `raw` keeps a line's text exactly as stored (hand-written shots); otherwise it is trimmed. */
function partText(part: ShotPart, raw = false): string {
	if (part.kind === "text") return part.text;
	const { speaker, language, text } = part.line;
	if (!text.trim()) return "";
	return `${speaker.trim() ? `(${speaker.trim()}) ` : ""}<d>[${language.trim() || "English"}] ${raw ? text : text.trim()}</d>`;
}

/** Concatenate pieces, adding one space only where neither side already has whitespace. */
function joinPieces(pieces: string[]): string {
	let out = "";
	for (const piece of pieces) {
		if (!piece) continue;
		if (out && !/\s$/.test(out) && !/^\s/.test(piece)) out += " ";
		out += piece;
	}
	return out;
}

function serializeShot(shot: Shot, n: number, start: number): string {
	// [Shot 1] has no timestamp; later shots mark their cut time.
	const cut = n > 1 ? `At ${formatCutTime(start)}, ` : "";

	if (shot.freeLead) {
		// Text parts keep their exact whitespace, so hand-written wording round-trips byte for byte.
		return `[Shot ${n}] ${cut}${joinPieces(shot.body.map((p) => partText(p, true)))}`.trim();
	}

	const body = shot.body
		.map((p) => partText(p).trim())
		.filter(Boolean)
		.join(" ");
	const label = SHOT_TYPE_LABEL[shot.type];
	const movementPart = shot.movement === "static" ? "" : `, ${MOVEMENT_LABEL[shot.movement].toLowerCase()}`;
	const lead =
		n > 1
			? `the shot cuts to ${/^[aeiou]/i.test(label) ? "an" : "a"} ${label === "POV" ? label : label.toLowerCase()} shot${movementPart}.`
			: `${label} shot${movementPart}.`;
	return [`[Shot ${n}] ${cut}${lead}`, body].filter(Boolean).join(" ");
}

/**
 * Serialize the shot list into detailed_description's [Shot N] format:
 * `[Shot 1] ...` then `[Shot N] At MM:SS.mmm, the shot cuts to ...`. Optional
 * `preamble` (style notes that come before the first shot) is written first.
 */
export function detailedDescriptionFromShots(shots: Shot[], preamble = ""): string {
	const starts = resolveShotStarts(shots);
	return [preamble.trim(), ...shots.map((s, i) => serializeShot(s, i + 1, starts[i]))]
		.filter(Boolean)
		.join("\n");
}

/* ---------------------------------------------------------------- */
/*  Parse                                                             */
/* ---------------------------------------------------------------- */

const CUT_TIME_RE = /^At (\d+):(\d{2})(?:\.(\d{1,3}))?,\s*/i;
// "[the shot cuts to] [a] <type> shot[, <movement>]." -- the lead-in this
// editor writes, recovered on parse so type/movement survive a reopen.
const LEAD_RE =
	/^(?:(?:the (?:shot|camera) (?:cuts|transitions|changes|switches) to |cuts to )?(?:an? )?)([a-zA-Z -]+?) shot(?:,\s*([a-zA-Z]+))?\.\s*/i;
const DIALOGUE_RE = /(?:\(([^)]+)\)\s*)?<d>(?:\[([^\]]*)\]\s*)?([\s\S]*?)<\/d>/g;

/**
 * Split shot text at its `<d>` blocks into an ordered body. `exact` keeps the
 * surrounding text untouched (whitespace included) so it can be rebuilt
 * byte for byte; otherwise prose is trimmed and empty runs dropped.
 */
function splitBody(text: string, exact: boolean): ShotPart[] {
	const parts: ShotPart[] = [];
	const pushText = (raw: string) => {
		const t = exact ? raw : raw.trim();
		if (t) parts.push(newTextPart(t));
	};
	let last = 0;
	for (const m of text.matchAll(DIALOGUE_RE)) {
		const start = m.index ?? 0;
		pushText(text.slice(last, start));
		parts.push(
			linePart(
				newLine({
					speaker: m[1]?.trim() ?? "",
					language: m[2]?.trim() || "English",
					text: exact ? (m[3] ?? "") : (m[3] ?? "").trim(),
				}),
			),
		);
		last = start + m[0].length;
	}
	pushText(text.slice(last));
	return parts;
}

function parseShotBlock(block: string, subjectLabels: string[]): { shot: Shot; time: number | null } {
	const shot = newShot();
	shot.subjectIds = subjectLabels.filter((label) => block.includes(label));
	let rest = block;

	let time: number | null = null;
	const t = rest.match(CUT_TIME_RE);
	if (t) {
		time = Number(t[1]) * 60 + Number(t[2]) + (t[3] ? Number(`0.${t[3]}`) : 0);
		rest = rest.slice(t[0].length);
	}

	const lead = rest.match(LEAD_RE);
	const typeId = lead ? SHOT_TYPE_BY_LABEL[lead[1].trim().toLowerCase()] : undefined;
	if (lead && typeId) {
		shot.type = typeId;
		const moveLabel = lead[2]?.trim().toLowerCase();
		if (moveLabel && MOVEMENT_BY_LABEL[moveLabel]) shot.movement = MOVEMENT_BY_LABEL[moveLabel];
		rest = rest.slice(lead[0].length);
		shot.body = splitBody(rest, false);
	} else {
		// Hand-written wording: keep it exactly as written. Dialogue is still
		// pulled out into editable lines, but each stays where it was written.
		shot.freeLead = true;
		shot.body = splitBody(rest, true);
	}
	if (shot.body.length === 0) shot.body = [newTextPart()];
	return { shot, time };
}

/**
 * Split an existing detailed_description back into shots so opening the
 * builder on a job someone already wrote doesn't discard it. Shots are found
 * by their line-start [Shot N] markers; anything before the first is kept as
 * the preamble. Only wording this editor generated is decomposed into
 * type/movement/dialogue -- anything else is kept verbatim.
 */
export function parseDetailedDescription(
	text: string,
	subjectLabels: string[],
): { preamble: string; shots: Shot[] } {
	const markers = [...text.matchAll(/^\[Shot\s+\d+\][ \t]*/gm)];
	if (markers.length === 0) {
		const body = text.trim();
		if (!body) return { preamble: "", shots: [] };
		const { shot } = parseShotBlock(body, subjectLabels);
		shot.freeLead = true;
		shot.body = splitBody(body, true);
		return { preamble: "", shots: [shot] };
	}

	const preamble = text.slice(0, markers[0].index).trim();
	const parsed = markers.map((m, i) => {
		const from = (m.index ?? 0) + m[0].length;
		const to = i + 1 < markers.length ? (markers[i + 1].index ?? text.length) : text.length;
		return parseShotBlock(text.slice(from, to).trim(), subjectLabels);
	});
	const shots = parsed.map((p) => p.shot);

	// Only pin a cut time when it differs from what the editor would compute,
	// so text this editor wrote keeps flowing as earlier shots change.
	const starts: number[] = [0];
	shots.forEach((shot, i) => {
		if (i === 0) return;
		const auto = starts[i - 1] + estimateShotSeconds(shots[i - 1]);
		const time = parsed[i].time;
		if (time == null) {
			starts.push(auto);
			return;
		}
		starts.push(time);
		if (Math.abs(time - auto) > 0.001) shot.startSeconds = time;
	});
	return { preamble, shots };
}

export function shotsFromDetailedDescription(text: string, subjectLabels: string[]): Shot[] {
	return parseDetailedDescription(text, subjectLabels).shots;
}

const RETENTION_LINE_RE = /^(<[^>]+>)\s*\(appears in [^)]*\):\s*(.*)$/gm;
const DEFAULT_RETENTION_NOTE = "fully_preserved - what is retained.";

/** label -> free-text retention note (everything after the colon), for lines this parser recognizes. */
function parseRetentionNotes(retentionAnalysis: string): Map<string, string> {
	const notes = new Map<string, string>();
	let m: RegExpExecArray | null;
	RETENTION_LINE_RE.lastIndex = 0;
	while ((m = RETENTION_LINE_RE.exec(retentionAnalysis))) notes.set(m[1], m[2].trim());
	return notes;
}

/**
 * Rebuild retention_analysis's shot-tracked lines from the shot list. Each
 * subject's existing retention note (the free text after the colon) is kept
 * untouched -- only the "(appears in [...])" shot list is regenerated, using
 * a sensible default note for a subject that's newly shot-tracked. Anything
 * this shot list doesn't own (untracked subjects, audio-reference lines,
 * free text) passes through unchanged rather than being discarded.
 */
export function retentionAnalysisFromShots(
	shots: Shot[],
	subjectLabels: string[],
	previousRetentionAnalysis: string,
): string {
	const existingNotes = parseRetentionNotes(previousRetentionAnalysis);
	const regenerated = new Set<string>();
	const shotLines: string[] = [];

	for (const label of subjectLabels) {
		const shotNums = shots
			.map((s, i) => (s.subjectIds.includes(label) ? i + 1 : null))
			.filter((n): n is number => n !== null);
		if (shotNums.length === 0) continue;
		regenerated.add(label);
		const note = existingNotes.get(label) ?? DEFAULT_RETENTION_NOTE;
		shotLines.push(`${label} (appears in ${shotNums.map((n) => `[Shot ${n}]`).join(", ")}): ${note}`);
	}

	const passthroughLines = previousRetentionAnalysis
		.split("\n")
		.map((l) => l.trim())
		.filter((line) => {
			if (!line) return false;
			const m = line.match(/^(<[^>]+>)\s*\(appears in/);
			return !(m && regenerated.has(m[1]));
		});

	return [...shotLines, ...passthroughLines].join("\n");
}
