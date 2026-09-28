/**
 * Start a clip from how the one before it ended -- either its last frame (a
 * still) or its last second or so (a short video).
 *
 * A scene is generated as many short clips. To make one pick up where the
 * previous ended, the previous clip's ending is attached as a reference, and
 * the prompt is rewritten the way the prompt guide describes for a picture
 * that anchors the opening frame (`attachLastFrame`) or a video that the shot
 * continues from (`attachLastClip`). The wording is kept together here so it
 * is easy to change if the guide is read differently.
 *
 * The two approaches trade off differently: a still is a hard, exact opening
 * frame, but the model has only a pose to guess the continuing motion from. A
 * short video carries real motion, at the cost of one more reference slot and
 * (per H3's own limits) at most 3 per job instead of 9.
 */
import { readField, writeField } from "@/lib/h3PromptFields";
import { H3_REFERENCE_LIMITS } from "@/lib/h3References";
import {
	detailedDescriptionFromShots,
	newTextPart,
	parseDetailedDescription,
} from "@/lib/h3Shots";

export interface Reference {
	source: string;
	media_type: string;
}

/**
 * A reference that means "the last frame of that job's video", resolved by the
 * server when this job runs -- so it can be attached before that job has
 * finished, and the server's queue knows this job must wait for it.
 */
export const DEFERRED_LAST_FRAME_PREFIX = "job-last-frame:";
/** As above, but for the previous job's trailing video instead of its last frame. */
export const DEFERRED_LAST_CLIP_PREFIX = "job-last-clip:";

export const deferredLastFrameSource = (jobId: string) => `${DEFERRED_LAST_FRAME_PREFIX}${jobId}`;
export const deferredLastClipSource = (jobId: string) => `${DEFERRED_LAST_CLIP_PREFIX}${jobId}`;

/** Saved by the server as `.../last_frames/last_frame_<job id>.png`. */
const LAST_FRAME_RE = /(?:^|\/)last_frame_([^/]+)\.png$/;
/** Saved by the server as `.../last_clips/last_clip_<job id>.mp4`. */
const LAST_CLIP_RE = /(?:^|\/)last_clip_([^/]+)\.mp4$/;

const REQUIRED_SECTIONS = [
	"subject_definitions",
	"summary",
	"retention_analysis",
	"detailed_description",
] as const;

/** The job a last-frame reference came from, or null if `source` isn't one. */
export function lastFrameSourceJobId(source: string): string | null {
	if (source.startsWith(DEFERRED_LAST_FRAME_PREFIX)) return source.slice(DEFERRED_LAST_FRAME_PREFIX.length) || null;
	return source.match(LAST_FRAME_RE)?.[1] ?? null;
}

/** Last-frame references already on a job, with the job each came from. */
export function attachedLastFrames(references: Reference[] | null | undefined): { source: string; jobId: string }[] {
	return (references ?? []).flatMap((r) => {
		const jobId = lastFrameSourceJobId(r.source);
		return jobId ? [{ source: r.source, jobId }] : [];
	});
}

/** The job a last-clip reference came from, or null if `source` isn't one. */
export function lastClipSourceJobId(source: string): string | null {
	if (source.startsWith(DEFERRED_LAST_CLIP_PREFIX)) return source.slice(DEFERRED_LAST_CLIP_PREFIX.length) || null;
	return source.match(LAST_CLIP_RE)?.[1] ?? null;
}

/** Last-clip references already on a job, with the job each came from. */
export function attachedLastClips(references: Reference[] | null | undefined): { source: string; jobId: string }[] {
	return (references ?? []).flatMap((r) => {
		const jobId = lastClipSourceJobId(r.source);
		return jobId ? [{ source: r.source, jobId }] : [];
	});
}

/** Why a last frame can't be attached to this job, or null if it can. */
export function whyCannotAttach(prompt: string, references: Reference[] | null | undefined): string | null {
	const missing = REQUIRED_SECTIONS.filter((s) => readField(prompt, s) === null);
	if (missing.length > 0) {
		return "This prompt isn't in the six-section format, so it can't be rewritten automatically. Attach the frame in Edit instead.";
	}
	const images = (references ?? []).filter((r) => r.media_type === "image").length;
	if (images >= H3_REFERENCE_LIMITS.image) {
		return `A job can have at most ${H3_REFERENCE_LIMITS.image} image references.`;
	}
	return null;
}

/** "[reference generation]" -> "[reference generation + keyframe completion]"; no tag -> "[keyframe completion] ...". */
function withKeyframeCompletion(summary: string): string {
	const tag = summary.match(/^\[([^\]]*)\]\s*/);
	if (!tag) return `[keyframe completion] ${summary}`.trim();
	const types = tag[1].split("+").map((t) => t.trim()).filter(Boolean);
	if (types.includes("keyframe completion")) return summary;
	return `[${[...types, "keyframe completion"].join(" + ")}] ${summary.slice(tag[0].length)}`.trim();
}

const isBlank = (text: string) => !text.trim() || text.trim().toUpperCase() === "N/A";
const appendLine = (body: string, line: string) => (isBlank(body) ? line : `${body}\n${line}`);

export type AttachResult =
	| { ok: true; prompt: string; references: Reference[]; label: string }
	| { ok: false; error: string };

/**
 * Add `framePath` as the next image reference and rewrite the prompt to use it
 * as the opening frame of Shot 1. Fails, changing nothing, if the frame is
 * already attached or the prompt isn't in a form that can be rewritten safely.
 */
export function attachLastFrame(prompt: string, references: Reference[], framePath: string): AttachResult {
	if (references.some((r) => r.source === framePath)) {
		return { ok: false, error: "That frame is already attached to this clip." };
	}
	const blocked = whyCannotAttach(prompt, references);
	if (blocked) return { ok: false, error: blocked };

	// Labels count per media type in reference order, so the new image is the next <Picture N>.
	const label = `<Picture ${references.filter((r) => r.media_type === "image").length + 1}>`;

	const { preamble, shots } = parseDetailedDescription(readField(prompt, "detailed_description") ?? "", []);
	if (shots.length === 0) return { ok: false, error: "This prompt has no shots to start from the frame." };
	const first = shots[0];
	const firstText = first.body.find((p) => p.kind === "text");
	if (firstText && firstText.kind === "text") firstText.text = `The shot begins from ${label}. ${firstText.text}`;
	else first.body.unshift(newTextPart(`The shot begins from ${label}.`));

	let next = prompt;
	next = writeField(
		next,
		"subject_definitions",
		appendLine(
			readField(next, "subject_definitions") ?? "",
			`${label} is the first frame of [Shot 1]: the last frame of the previous clip, showing the scene, positions, and lighting exactly as that clip ended.`,
		),
	);
	next = writeField(
		next,
		"summary",
		`${withKeyframeCompletion(readField(next, "summary") ?? "")} The target video starts from ${label}, the last frame of the previous clip.`,
	);
	next = writeField(
		next,
		"retention_analysis",
		appendLine(
			readField(next, "retention_analysis") ?? "",
			`${label} ([Shot 1] first frame): fully_preserved - the scene, positions, and lighting of the previous clip's last frame are retained.`,
		),
	);
	next = writeField(next, "detailed_description", detailedDescriptionFromShots(shots, preamble));

	return { ok: true, prompt: next, references: [...references, { source: framePath, media_type: "image" }], label };
}

/** Drop the lines `attachLastFrame` added, leaving "N/A" if nothing else was there. */
function removeLine(body: string, startsWith: string): string {
	const kept = body.split("\n").filter((l) => !l.trim().startsWith(startsWith));
	const text = kept.join("\n").trim();
	return text || "N/A";
}

export type DetachResult = { ok: true; prompt: string; references: Reference[] } | { ok: false; error: string };

/**
 * Undo `attachLastFrame`: remove the reference at `source` and take back what was
 * added to the prompt for it. Only the most recently added image can be removed --
 * removing an earlier one would renumber the pictures after it.
 */
export function detachLastFrame(prompt: string, references: Reference[], source: string): DetachResult {
	const images = references.filter((r) => r.media_type === "image");
	const index = images.findIndex((r) => r.source === source);
	if (index < 0) return { ok: false, error: "That frame isn't attached to this clip." };
	if (index !== images.length - 1) {
		return { ok: false, error: "Other images were added after that frame. Remove it in Edit instead." };
	}
	const remaining = references.filter((r) => r.source !== source);

	// A prompt that isn't in the six-section form was never rewritten; only the reference goes.
	if (REQUIRED_SECTIONS.some((s) => readField(prompt, s) === null)) {
		return { ok: true, prompt, references: remaining };
	}

	const label = `<Picture ${index + 1}>`;
	let next = prompt;
	next = writeField(next, "subject_definitions", removeLine(readField(next, "subject_definitions") ?? "", `${label} is the first frame of [Shot 1]`));
	next = writeField(next, "retention_analysis", removeLine(readField(next, "retention_analysis") ?? "", `${label} ([Shot 1] first frame)`));

	const summary = readField(next, "summary") ?? "";
	const sentence = ` The target video starts from ${label}, the last frame of the previous clip.`;
	if (summary.includes(sentence)) {
		let restored = summary.replace(sentence, "");
		const tag = restored.match(/^\[([^\]]*)\]\s*/);
		if (tag) {
			const types = tag[1].split("+").map((t) => t.trim()).filter((t) => t && t !== "keyframe completion");
			restored = types.length > 0 ? `[${types.join(" + ")}] ${restored.slice(tag[0].length)}` : restored.slice(tag[0].length);
		}
		next = writeField(next, "summary", restored.trim());
	}

	const { preamble, shots } = parseDetailedDescription(readField(next, "detailed_description") ?? "", []);
	const lead = `The shot begins from ${label}.`;
	const first = shots[0];
	const part = first?.body.find((p) => p.kind === "text" && p.text.startsWith(lead));
	if (first && part && part.kind === "text") {
		part.text = part.text.slice(lead.length).trimStart();
		if (!part.text) first.body = first.body.filter((p) => p !== part);
		next = writeField(next, "detailed_description", detailedDescriptionFromShots(shots, preamble));
	}
	return { ok: true, prompt: next, references: remaining };
}

/** Why a trailing clip can't be attached to this job, or null if it can. */
export function whyCannotAttachClip(prompt: string, references: Reference[] | null | undefined): string | null {
	const missing = REQUIRED_SECTIONS.filter((s) => readField(prompt, s) === null);
	if (missing.length > 0) {
		return "This prompt isn't in the six-section format, so it can't be rewritten automatically. Attach the clip in Edit instead.";
	}
	const videos = (references ?? []).filter((r) => r.media_type === "video").length;
	if (videos >= H3_REFERENCE_LIMITS.video) {
		return `A job can have at most ${H3_REFERENCE_LIMITS.video} video references.`;
	}
	return null;
}

/** "[reference generation + audio reference]" -> "[video continuation + audio reference]"; drops "keyframe completion". */
function withVideoContinuation(summary: string): string {
	const tag = summary.match(/^\[([^\]]*)\]\s*/);
	const rest = tag ? summary.slice(tag[0].length) : summary;
	const kept = tag
		? tag[1].split("+").map((t) => t.trim()).filter((t) => t && t !== "reference generation" && t !== "keyframe completion")
		: [];
	if (!kept.includes("video continuation")) kept.unshift("video continuation");
	return `[${kept.join(" + ")}] ${rest}`.trim();
}

/**
 * Add `clipPath` as the next video reference and rewrite the prompt so Shot 1
 * continues from it, per the guide's video-reference form: a standalone
 * `<Video N>` declaration (videos aren't bound to a `<Subject>` the way
 * pictures and audio are), a `[video continuation]` task tag, a retention
 * line, and "the shot continues from <Video N>" in the first shot.
 */
export function attachLastClip(prompt: string, references: Reference[], clipPath: string): AttachResult {
	if (references.some((r) => r.source === clipPath)) {
		return { ok: false, error: "That clip is already attached to this job." };
	}
	const blocked = whyCannotAttachClip(prompt, references);
	if (blocked) return { ok: false, error: blocked };

	const label = `<Video ${references.filter((r) => r.media_type === "video").length + 1}>`;

	const { preamble, shots } = parseDetailedDescription(readField(prompt, "detailed_description") ?? "", []);
	if (shots.length === 0) return { ok: false, error: "This prompt has no shots to continue from the clip." };
	const first = shots[0];
	const lead = `The shot continues from ${label}.`;
	const firstText = first.body.find((p) => p.kind === "text");
	if (firstText && firstText.kind === "text") firstText.text = `${lead} ${firstText.text}`;
	else first.body.unshift(newTextPart(lead));

	let next = prompt;
	next = writeField(
		next,
		"subject_definitions",
		appendLine(
			readField(next, "subject_definitions") ?? "",
			`${label} is the source video for [Shot 1]: the last second of the previous clip, providing the continuation starting point and its motion.`,
		),
	);
	next = writeField(next, "summary", withVideoContinuation(readField(next, "summary") ?? ""));
	next = writeField(
		next,
		"retention_analysis",
		appendLine(
			readField(next, "retention_analysis") ?? "",
			`${label} (continuation source): fully_preserved - the motion, scene, positions, and lighting of the previous clip's ending are continued exactly.`,
		),
	);
	next = writeField(next, "detailed_description", detailedDescriptionFromShots(shots, preamble));

	return { ok: true, prompt: next, references: [...references, { source: clipPath, media_type: "video" }], label };
}

/**
 * Undo `attachLastClip`: remove the reference at `source` and take back what
 * was added to the prompt for it. Only the most recently added video can be
 * removed -- removing an earlier one would renumber the videos after it.
 */
export function detachLastClip(prompt: string, references: Reference[], source: string): DetachResult {
	const videos = references.filter((r) => r.media_type === "video");
	const index = videos.findIndex((r) => r.source === source);
	if (index < 0) return { ok: false, error: "That clip isn't attached to this job." };
	if (index !== videos.length - 1) {
		return { ok: false, error: "Other videos were added after that clip. Remove it in Edit instead." };
	}
	const remaining = references.filter((r) => r.source !== source);

	if (REQUIRED_SECTIONS.some((s) => readField(prompt, s) === null)) {
		return { ok: true, prompt, references: remaining };
	}

	const label = `<Video ${index + 1}>`;
	let next = prompt;
	next = writeField(next, "subject_definitions", removeLine(readField(next, "subject_definitions") ?? "", `${label} is the source video for [Shot 1]`));
	next = writeField(next, "retention_analysis", removeLine(readField(next, "retention_analysis") ?? "", `${label} (continuation source)`));

	// "video continuation" replaced "reference generation" as the base tag (attachLastClip drops
	// it outright, unlike "keyframe completion" which is only ever added alongside it) -- so
	// undoing it means putting "reference generation" back, not just deleting the tag.
	const summaryTag = (readField(next, "summary") ?? "").match(/^\[([^\]]*)\]\s*/);
	if (summaryTag && summaryTag[1].split("+").map((t) => t.trim()).includes("video continuation")) {
		const rest = (readField(next, "summary") ?? "").slice(summaryTag[0].length);
		const types = summaryTag[1]
			.split("+")
			.map((t) => t.trim())
			.map((t) => (t === "video continuation" ? "reference generation" : t));
		next = writeField(next, "summary", `[${types.join(" + ")}] ${rest}`.trim());
	}

	const { preamble, shots } = parseDetailedDescription(readField(next, "detailed_description") ?? "", []);
	const lead = `The shot continues from ${label}.`;
	const first = shots[0];
	const part = first?.body.find((p) => p.kind === "text" && p.text.startsWith(lead));
	if (first && part && part.kind === "text") {
		part.text = part.text.slice(lead.length).trimStart();
		if (!part.text) first.body = first.body.filter((p) => p !== part);
		next = writeField(next, "detailed_description", detailedDescriptionFromShots(shots, preamble));
	}
	return { ok: true, prompt: next, references: remaining };
}
