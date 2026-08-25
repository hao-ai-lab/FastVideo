/**
 * MiniMax-H3 Ref2VA reference helpers.
 *
 * Labels mirror `build_ref2va_presentation` in
 * fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_conditioning.py:
 * per-type counters incremented in list order, so the label a user sees here is
 * the one the model is shown.
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

/**
 * Scaffold the six-section prompt from
 * docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md, pre-filled with the labels this
 * reference list will actually receive.
 */
export function referencePromptTemplate(refs: H3Reference[]): string {
	const labels = labelReferences(refs);
	const visual = labels.filter((l) => !l.startsWith("<Audio"));
	const subjects = visual.length
		? visual
				.map(
					(label, i) =>
						`  <Subject ${i + 1}> is the subject from ${label}; describe appearance, distinguishing features, and what must stay consistent.`,
				)
				.join("\n")
		: "  <Subject 1> is ...";
	const audio = labels.filter((l) => l.startsWith("<Audio"));
	const audioLine = audio.length
		? `\n  Audio: reuse ${audio.join(", ")} as described below.`
		: "";

	return [
		"subject_definitions:",
		subjects + audioLine,
		"",
		"summary:",
		"  Full-reference generation. Describe the target video and which reference supplies what.",
		"",
		"retention_analysis:",
		labels
			.map(
				(label) =>
					`  ${label}: state whether it is fully preserved, partially preserved, transferred, or reused, and where it appears.`,
			)
			.join("\n"),
		"",
		"detailed_description:",
		"  Describe composition, subject appearance and position, environment and lighting,",
		"  actions and state changes, camera movement, and sound, in playback order.",
		"  Say explicitly where each referenced item appears or takes effect.",
		"",
		"overall_soundscape:",
		"  Ambience and physical sounds.",
		"",
		"non_diegetic_music:",
		"  Background music audible only to the audience.",
	].join("\n");
}
