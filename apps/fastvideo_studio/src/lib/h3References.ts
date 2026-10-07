/**
 * MiniMax-H3 Ref2VA reference helpers. Labels mirror the per-type counters in
 * `build_ref2va_presentation`, so what the user sees is what the model is shown.
 */
import type { MediaType } from "@/lib/api";

export interface H3Reference {
	/** stable key for React lists */
	id: string;
	source: string;
	media_type: MediaType;
	fileName: string;
}

/** Per-media-type caps enforced by validate_references (reference.py). */
export const H3_REFERENCE_LIMITS: Record<MediaType, number> = {
	image: 9,
	video: 3,
	audio: 3,
};
export const H3_MAX_REFERENCES = 12;

const LABEL_FOR: Record<MediaType, string> = {
	image: "Picture",
	video: "Video",
	audio: "Audio",
};

/** Label each reference the way the pipeline will, e.g. "<Picture 2>". */
export function labelReferences(refs: H3Reference[]): string[] {
	const counts: Record<MediaType, number> = { image: 0, video: 0, audio: 0 };
	return refs.map((ref) => {
		counts[ref.media_type] += 1;
		return `<${LABEL_FOR[ref.media_type]} ${counts[ref.media_type]}>`;
	});
}

/** Human-readable reason the list is invalid, or null when it is acceptable. */
export function validateReferences(refs: H3Reference[]): string | null {
	if (refs.length === 0) return null;
	if (refs.length > H3_MAX_REFERENCES) {
		return `At most ${H3_MAX_REFERENCES} references (have ${refs.length}).`;
	}
	const counts: Record<MediaType, number> = { image: 0, video: 0, audio: 0 };
	for (const ref of refs) counts[ref.media_type] += 1;
	for (const type of Object.keys(counts) as MediaType[]) {
		if (counts[type] > H3_REFERENCE_LIMITS[type]) {
			return `At most ${H3_REFERENCE_LIMITS[type]} ${type} references (have ${counts[type]}).`;
		}
	}
	if (counts.audio === refs.length) {
		return "Audio references must be paired with at least one image or video.";
	}
	return null;
}

/** Seed the guided prompt fields from the current reference list. */
export function referencePromptSeed(
	refs: H3Reference[],
): Record<string, string> {
	const labels = labelReferences(refs);
	const visual = labels.filter((l) => !l.startsWith("<Audio"));
	const audio = labels.filter((l) => l.startsWith("<Audio"));

	const subjects = visual.map(
		(label, i) =>
			`<Subject ${i + 1}> is the subject from ${label}; describe appearance and distinguishing features.`,
	);
	for (const label of audio) {
		subjects.push(`${label} is the audio reference; describe what it provides.`);
	}

	const retention = visual.map(
		(_, i) =>
			`<Subject ${i + 1}> (appears in [Shot 1]): fully_preserved - what is retained.`,
	);
	for (const label of audio) {
		retention.push(`${label}: reference - how it guides the audio.`);
	}

	return {
		subject_definitions: subjects.join("\n"),
		summary: "[reference generation] Describe the target video and each reference's role.",
		retention_analysis: retention.join("\n"),
		detailed_description:
			"[Shot 1] Describe composition, subjects, environment, lighting, action and camera movement, saying where each reference takes effect.",
		overall_soundscape: "",
		non_diegetic_music: "",
	};
}
