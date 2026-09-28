import { describe, expect, it } from "vitest";
import {
	attachLastClip,
	attachLastFrame,
	attachedLastClips,
	attachedLastFrames,
	deferredLastClipSource,
	deferredLastFrameSource,
	detachLastClip,
	detachLastFrame,
	lastClipSourceJobId,
	lastFrameSourceJobId,
	whyCannotAttach,
	whyCannotAttachClip,
} from "@/lib/continuation";
import { parseH3Prompt } from "@/lib/h3Prompt";
import { readField } from "@/lib/h3PromptFields";

const FRAME = "/data/uploads/last_frames/last_frame_job-5.png";
const REFS = [
	{ source: "/refs/veteran.png", media_type: "image" },
	{ source: "/refs/young.png", media_type: "image" },
];

const CLIP = [
	"subject_definitions:",
	"<Subject 1> is the veteran broker from <Picture 1>: a confident man.",
	"<Subject 2> is the young broker from <Picture 2>: eager.",
	"",
	"summary:",
	"[reference generation] The target video shows <Subject 1> and <Subject 2> at a lunch.",
	"",
	"retention_analysis:",
	"<Subject 1> (appears in [Shot 1]): fully_preserved - his suit.",
	"<Subject 2> (appears in [Shot 1]): fully_preserved - his suit.",
	"",
	"detailed_description:",
	"[Shot 1] Close-up shot, dolly. Continuing the same shot on <Subject 1>, still speaking. (S1) <d>[English] And then, well.</d>",
	"[Shot 2] At 00:04.000, the shot cuts to a wide shot. Both men at the table.",
	"",
	"overall_soundscape:",
	"Muted restaurant ambience.",
	"",
	"non_diegetic_music:",
	"N/A",
].join("\n");

function attach(prompt = CLIP, refs = REFS, frame = FRAME) {
	const r = attachLastFrame(prompt, refs, frame);
	if (!r.ok) throw new Error(r.error);
	return r;
}

describe("attachLastFrame", () => {
	it("adds the frame as the next image reference and names it by its picture number", () => {
		const r = attach();
		expect(r.label).toBe("<Picture 3>");
		expect(r.references).toEqual([...REFS, { source: FRAME, media_type: "image" }]);
	});

	it("defines the frame as the first frame of Shot 1, after the existing definitions", () => {
		const defs = readField(attach().prompt, "subject_definitions") as string;
		expect(defs.split("\n")).toEqual([
			"<Subject 1> is the veteran broker from <Picture 1>: a confident man.",
			"<Subject 2> is the young broker from <Picture 2>: eager.",
			"<Picture 3> is the first frame of [Shot 1]: the last frame of the previous clip, showing the scene, positions, and lighting exactly as that clip ended.",
		]);
	});

	it("tags the summary as keyframe completion alongside its existing task type, and names the frame", () => {
		const summary = readField(attach().prompt, "summary") as string;
		expect(summary.startsWith("[reference generation + keyframe completion] The target video shows")).toBe(true);
		expect(summary.endsWith("The target video starts from <Picture 3>, the last frame of the previous clip.")).toBe(true);
	});

	it("adds a first-frame line to the retention analysis", () => {
		const lines = (readField(attach().prompt, "retention_analysis") as string).split("\n");
		expect(lines).toHaveLength(3);
		expect(lines[2]).toBe(
			"<Picture 3> ([Shot 1] first frame): fully_preserved - the scene, positions, and lighting of the previous clip's last frame are retained.",
		);
	});

	it("starts the first shot from the frame and leaves the other shots alone", () => {
		const shots = (readField(attach().prompt, "detailed_description") as string).split("\n");
		expect(shots[0]).toBe(
			"[Shot 1] Close-up shot, dolly. The shot begins from <Picture 3>. Continuing the same shot on <Subject 1>, still speaking. (S1) <d>[English] And then, well.</d>",
		);
		expect(shots[1]).toBe("[Shot 2] At 00:04.000, the shot cuts to a wide shot. Both men at the table.");
	});

	it("leaves every other section untouched", () => {
		const out = attach().prompt;
		expect(readField(out, "overall_soundscape")).toBe("Muted restaurant ambience.");
		expect(readField(out, "non_diegetic_music")).toBe("N/A");
	});

	it("still reads as a six-section prompt", () => {
		expect(parseH3Prompt(attach().prompt)).not.toBeNull();
	});

	it("counts only images when numbering, so a video reference doesn't shift the label", () => {
		const refs = [REFS[0], { source: "/refs/clip.mp4", media_type: "video" }, REFS[1]];
		expect(attach(CLIP, refs).label).toBe("<Picture 3>");
		expect(attach(CLIP, [{ source: "/refs/clip.mp4", media_type: "video" }]).label).toBe("<Picture 1>");
	});

	it("does not mutate what it was given", () => {
		const refs = [...REFS];
		attach(CLIP, refs);
		expect(refs).toEqual(REFS);
	});
});

describe("attachLastFrame: summary wording", () => {
	const summaryOf = (summary: string) =>
		readField(attach(CLIP.replace("[reference generation] The target video shows <Subject 1> and <Subject 2> at a lunch.", summary)).prompt, "summary") as string;

	it("adds the tag when the summary has none", () => {
		expect(summaryOf("Two men at a lunch.").startsWith("[keyframe completion] Two men at a lunch.")).toBe(true);
	});

	it("extends a combined tag", () => {
		expect(summaryOf("[video editing + reference generation] Edit.").startsWith("[video editing + reference generation + keyframe completion] Edit.")).toBe(true);
	});

	it("does not repeat a tag that is already there", () => {
		const s = summaryOf("[keyframe completion + reference generation] Both.");
		expect(s.startsWith("[keyframe completion + reference generation] Both.")).toBe(true);
		expect(s.match(/keyframe completion/g)).toHaveLength(1);
	});
});

describe("attachLastFrame: other prompt shapes", () => {
	it("replaces an N/A subject_definitions instead of appending under it", () => {
		const out = attach(CLIP.replace(/subject_definitions:\n[\s\S]*?\n\nsummary:/, "subject_definitions:\nN/A\n\nsummary:")).prompt;
		expect(readField(out, "subject_definitions")).toBe(
			"<Picture 3> is the first frame of [Shot 1]: the last frame of the previous clip, showing the scene, positions, and lighting exactly as that clip ended.",
		);
	});

	it("keeps hand-written shots verbatim, only prefixing the first", () => {
		const hand = CLIP.replace(
			/\[Shot 1\][^\n]*\n\[Shot 2\][^\n]*/,
			"[Shot 1] A handheld medium shot holds <Subject 1> at frame left.\n[Shot 2] At 00:03.800, the shot cuts to a low tracking shot behind him.",
		);
		const shots = (readField(attach(hand).prompt, "detailed_description") as string).split("\n");
		expect(shots).toEqual([
			"[Shot 1] The shot begins from <Picture 3>. A handheld medium shot holds <Subject 1> at frame left.",
			"[Shot 2] At 00:03.800, the shot cuts to a low tracking shot behind him.",
		]);
	});

	it("adds the sentence to a first shot that is only dialogue", () => {
		const dialogueOnly = CLIP.replace(
			"[Shot 1] Close-up shot, dolly. Continuing the same shot on <Subject 1>, still speaking. (S1) <d>[English] And then, well.</d>",
			"[Shot 1] Wide shot. (S1) <d>[English] Hi.</d>",
		);
		const first = (readField(attach(dialogueOnly).prompt, "detailed_description") as string).split("\n")[0];
		expect(first).toBe("[Shot 1] Wide shot. The shot begins from <Picture 3>. (S1) <d>[English] Hi.</d>");
	});
});

describe("attachLastFrame: refusals", () => {
	it("won't attach the same frame twice", () => {
		const r = attachLastFrame(CLIP, [...REFS, { source: FRAME, media_type: "image" }], FRAME);
		expect(r).toEqual({ ok: false, error: "That frame is already attached to this clip." });
	});

	it("won't rewrite a prompt that isn't in the six-section format", () => {
		const r = attachLastFrame("A toy car drives into a plush dog.", REFS, FRAME);
		expect(r.ok).toBe(false);
		expect(!r.ok && r.error).toMatch(/six-section/);
	});

	it("won't rewrite a base-format prompt either", () => {
		const base = "For the target video, <Picture 1> is used.\n\nintegrated_multimodal_description: [Shot 1] A.\n\noverall_soundscape: Quiet.";
		expect(whyCannotAttach(base, REFS)).toMatch(/six-section/);
	});

	it("stops at nine image references", () => {
		const nine = Array.from({ length: 9 }, (_, i) => ({ source: `/r/${i}.png`, media_type: "image" }));
		const r = attachLastFrame(CLIP, nine, FRAME);
		expect(!r.ok && r.error).toMatch(/at most 9 image/);
	});

	it("reports a prompt with no shots", () => {
		const empty = CLIP.replace(/detailed_description:\n[\s\S]*?\n\noverall_soundscape:/, "detailed_description:\n\n\noverall_soundscape:");
		const r = attachLastFrame(empty, REFS, FRAME);
		expect(r.ok).toBe(false);
	});
});

describe("recognizing an attached last frame", () => {
	it("reads the job id out of the saved path", () => {
		expect(lastFrameSourceJobId(FRAME)).toBe("job-5");
		expect(lastFrameSourceJobId("last_frame_abc-123.png")).toBe("abc-123");
		expect(lastFrameSourceJobId("/refs/veteran.png")).toBeNull();
		expect(lastFrameSourceJobId("/x/not_last_frame_job-5.png")).toBeNull();
	});

	it("finds last-frame references among a job's references", () => {
		expect(attachedLastFrames([...REFS, { source: FRAME, media_type: "image" }])).toEqual([{ source: FRAME, jobId: "job-5" }]);
		expect(attachedLastFrames(REFS)).toEqual([]);
		expect(attachedLastFrames(null)).toEqual([]);
	});

	it("reads the job id out of a reference the server resolves when the job runs", () => {
		expect(deferredLastFrameSource("job-5")).toBe("job-last-frame:job-5");
		expect(lastFrameSourceJobId(deferredLastFrameSource("job-5"))).toBe("job-5");
		expect(lastFrameSourceJobId("job-last-frame:")).toBeNull();
		expect(attachedLastFrames([...REFS, { source: deferredLastFrameSource("job-5"), media_type: "image" }])).toEqual([
			{ source: "job-last-frame:job-5", jobId: "job-5" },
		]);
	});

	it("attaches a deferred frame exactly like a saved one", () => {
		const result = attachLastFrame(CLIP, REFS, deferredLastFrameSource("job-5"));
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.references.at(-1)).toEqual({ source: "job-last-frame:job-5", media_type: "image" });
			expect(result.label).toBe("<Picture 3>");
		}
	});
});

describe("detachLastFrame", () => {
	const roundTrip = (prompt: string, frame = FRAME) => {
		const attached = attach(prompt, REFS, frame);
		const r = detachLastFrame(attached.prompt, attached.references, frame);
		if (!r.ok) throw new Error(r.error);
		return r;
	};

	it("takes back exactly what attaching added", () => {
		const r = roundTrip(CLIP);
		expect(r.references).toEqual(REFS);
		expect(r.prompt).toBe(CLIP);
	});

	it("works for a frame the server resolves when the job runs", () => {
		expect(roundTrip(CLIP, deferredLastFrameSource("job-5")).prompt).toBe(CLIP);
	});

	it.each([
		["a summary with no tag", "Two men at a lunch."],
		["a combined tag", "[video editing + reference generation] Edit."],
	])("restores %s", (_name, summary) => {
		const prompt = CLIP.replace("[reference generation] The target video shows <Subject 1> and <Subject 2> at a lunch.", summary);
		expect(roundTrip(prompt).prompt).toBe(prompt);
	});

	it("restores hand-written shots and an N/A section", () => {
		const hand = CLIP.replace(/\[Shot 1\][^\n]*\n\[Shot 2\][^\n]*/, "[Shot 1] A handheld medium shot.\n[Shot 2] At 00:03.800, the shot cuts to a low tracking shot.").replace(
			/subject_definitions:\n[\s\S]*?\n\nsummary:/,
			"subject_definitions:\nN/A\n\nsummary:",
		);
		expect(roundTrip(hand).prompt).toBe(hand);
	});

	it("removes the sentence from a first shot that is only dialogue", () => {
		const dialogueOnly = CLIP.replace(
			"[Shot 1] Close-up shot, dolly. Continuing the same shot on <Subject 1>, still speaking. (S1) <d>[English] And then, well.</d>",
			"[Shot 1] Wide shot. (S1) <d>[English] Hi.</d>",
		);
		expect(roundTrip(dialogueOnly).prompt).toBe(dialogueOnly);
	});

	it("only removes the reference from a prompt that was never rewritten", () => {
		const r = detachLastFrame("A plain prompt.", [...REFS, { source: FRAME, media_type: "image" }], FRAME);
		expect(r).toEqual({ ok: true, prompt: "A plain prompt.", references: REFS });
	});

	it("refuses when other images were added after the frame", () => {
		const attached = attach();
		const r = detachLastFrame(attached.prompt, [...attached.references, { source: "/refs/extra.png", media_type: "image" }], FRAME);
		expect(r.ok).toBe(false);
	});

	it("refuses a frame that isn't attached", () => {
		expect(detachLastFrame(CLIP, REFS, FRAME)).toEqual({ ok: false, error: "That frame isn't attached to this clip." });
	});
});

const CLIP_VIDEO = "/data/uploads/last_clips/last_clip_job-5.mp4";

function attachClip(prompt = CLIP, refs = REFS, clip = CLIP_VIDEO) {
	const r = attachLastClip(prompt, refs, clip);
	if (!r.ok) throw new Error(r.error);
	return r;
}

describe("attachLastClip", () => {
	it("adds the clip as the next video reference and names it by its video number", () => {
		const r = attachClip();
		expect(r.label).toBe("<Video 1>");
		expect(r.references).toEqual([...REFS, { source: CLIP_VIDEO, media_type: "video" }]);
	});

	it("numbers a second video reference after the first", () => {
		const withOne = attachClip();
		const r = attachLastClip(withOne.prompt, withOne.references, "/refs/another.mp4");
		if (!r.ok) throw new Error(r.error);
		expect(r.label).toBe("<Video 2>");
	});

	it("declares the video standalone, not bound to a <Subject> the way pictures and audio are", () => {
		const defs = readField(attachClip().prompt, "subject_definitions") as string;
		expect(defs).toContain("<Video 1> is the source video for [Shot 1]");
		expect(defs).not.toMatch(/<Subject \d+> is .*<Video 1>/);
	});

	it("adds a retention line with the fully_preserved status", () => {
		const line = (readField(attachClip().prompt, "retention_analysis") as string).split("\n").at(-1);
		expect(line).toBe(
			"<Video 1> (continuation source): fully_preserved - the motion, scene, positions, and lighting of the previous clip's ending are continued exactly.",
		);
	});

	it("prefixes the first shot with a continuation lead, keeping hand-written shots verbatim otherwise", () => {
		const hand = CLIP.replace(
			/\[Shot 1\][^\n]*\n\[Shot 2\][^\n]*/,
			"[Shot 1] A handheld medium shot holds <Subject 1> at frame left.\n[Shot 2] At 00:03.800, the shot cuts to a low tracking shot.",
		);
		const shots = (readField(attachClip(hand).prompt, "detailed_description") as string).split("\n");
		expect(shots).toEqual([
			"[Shot 1] The shot continues from <Video 1>. A handheld medium shot holds <Subject 1> at frame left.",
			"[Shot 2] At 00:03.800, the shot cuts to a low tracking shot.",
		]);
	});

	describe("summary wording", () => {
		const summaryOf = (summary: string) =>
			readField(attachClip(CLIP.replace("[reference generation] The target video shows <Subject 1> and <Subject 2> at a lunch.", summary)).prompt, "summary") as string;

		it("adds the tag when the summary has none", () => {
			expect(summaryOf("Two men at a lunch.")).toBe("[video continuation] Two men at a lunch.");
		});

		it("replaces reference generation, not adds to it", () => {
			expect(summaryOf("[reference generation] Two men.")).toBe("[video continuation] Two men.");
		});

		it("keeps a combined audio reference tag, per the guide's documented combination", () => {
			expect(summaryOf("[reference generation + audio reference] Two men.")).toBe("[video continuation + audio reference] Two men.");
		});

		it("drops keyframe completion when switching from a picture-based continuation", () => {
			expect(summaryOf("[reference generation + keyframe completion] Two men.")).toBe("[video continuation] Two men.");
		});
	});

	describe("refusals", () => {
		it("won't attach the same clip twice", () => {
			const r = attachLastClip(CLIP, [...REFS, { source: CLIP_VIDEO, media_type: "video" }], CLIP_VIDEO);
			expect(r).toEqual({ ok: false, error: "That clip is already attached to this job." });
		});

		it("refuses a prompt that is not in the six-section format", () => {
			const r = attachLastClip("A plain prompt.", REFS, CLIP_VIDEO);
			expect(r).toEqual({ ok: false, error: "This prompt isn't in the six-section format, so it can't be rewritten automatically. Attach the clip in Edit instead." });
		});

		it("refuses past H3's video reference limit", () => {
			const many = Array.from({ length: 3 }, (_, i) => ({ source: `/v${i}.mp4`, media_type: "video" }));
			const r = attachLastClip(CLIP, [...REFS, ...many], CLIP_VIDEO);
			expect(r).toEqual({ ok: false, error: "A job can have at most 3 video references." });
		});

		it("refuses a prompt with no shots", () => {
			const noShots = CLIP.replace(/detailed_description:\n[\s\S]*?\n\noverall_soundscape:/, "detailed_description:\n\noverall_soundscape:");
			const r = attachLastClip(noShots, REFS, CLIP_VIDEO);
			expect(r).toEqual({ ok: false, error: "This prompt has no shots to continue from the clip." });
		});
	});
});

describe("recognizing an attached last clip", () => {
	it("reads the job id out of the saved path", () => {
		expect(lastClipSourceJobId(CLIP_VIDEO)).toBe("job-5");
		expect(lastClipSourceJobId("last_clip_abc-123.mp4")).toBe("abc-123");
		expect(lastClipSourceJobId("/refs/veteran.png")).toBeNull();
	});

	it("reads the job id out of a reference the server resolves when the job runs", () => {
		expect(deferredLastClipSource("job-5")).toBe("job-last-clip:job-5");
		expect(lastClipSourceJobId(deferredLastClipSource("job-5"))).toBe("job-5");
	});

	it("finds last-clip references among a job's references, separately from last-frame ones", () => {
		const frame = { source: "/data/uploads/last_frames/last_frame_job-2.png", media_type: "image" };
		expect(attachedLastClips([...REFS, frame, { source: CLIP_VIDEO, media_type: "video" }])).toEqual([
			{ source: CLIP_VIDEO, jobId: "job-5" },
		]);
		expect(attachedLastClips([frame])).toEqual([]);
		expect(attachedLastClips(null)).toEqual([]);
	});
});

describe("whyCannotAttachClip", () => {
	it("blocks a prompt not in the six-section format", () => {
		expect(whyCannotAttachClip("A plain prompt.", REFS)).toMatch(/six-section/);
	});

	it("allows a normal prompt under the video limit", () => {
		expect(whyCannotAttachClip(CLIP, REFS)).toBeNull();
	});
});

describe("detachLastClip", () => {
	const roundTrip = (prompt: string, clip = CLIP_VIDEO) => {
		const attached = attachClip(prompt, REFS, clip);
		const r = detachLastClip(attached.prompt, attached.references, clip);
		if (!r.ok) throw new Error(r.error);
		return r;
	};

	it("takes back exactly what attaching added", () => {
		const r = roundTrip(CLIP);
		expect(r.references).toEqual(REFS);
		expect(r.prompt).toBe(CLIP);
	});

	it("works for a clip the server resolves when the job runs", () => {
		expect(roundTrip(CLIP, deferredLastClipSource("job-5")).prompt).toBe(CLIP);
	});

	it("round-trips a starting tag combined with audio reference", () => {
		// A realistic starting point: audio references already attached (see continuation.test.ts's
		// own audio-reference wiring), before any video continuation is attached.
		const prompt = CLIP.replace(
			"[reference generation] The target video shows <Subject 1> and <Subject 2> at a lunch.",
			"[reference generation + audio reference] Two men at a lunch.",
		);
		expect(roundTrip(prompt).prompt).toBe(prompt);
	});

	// Unlike attachLastFrame's "keyframe completion" (only ever added alongside the existing base
	// tag), "video continuation" *replaces* the base tag -- so once attached, nothing remembers
	// what was there before, and detaching lands on the default "reference generation", not back
	// on "no tag".
	it("restores a summary with no tag to the default reference-generation tag, not back to no tag", () => {
		const prompt = CLIP.replace("[reference generation] The target video shows <Subject 1> and <Subject 2> at a lunch.", "Two men at a lunch.");
		expect(roundTrip(prompt).prompt).toBe(CLIP.replace("The target video shows <Subject 1> and <Subject 2> at a lunch.", "Two men at a lunch."));
	});

	it("restores hand-written shots and an N/A section", () => {
		const hand = CLIP.replace(/\[Shot 1\][^\n]*\n\[Shot 2\][^\n]*/, "[Shot 1] A handheld medium shot.\n[Shot 2] At 00:03.800, the shot cuts to a low tracking shot.").replace(
			/subject_definitions:\n[\s\S]*?\n\nsummary:/,
			"subject_definitions:\nN/A\n\nsummary:",
		);
		expect(roundTrip(hand).prompt).toBe(hand);
	});

	it("only removes the reference from a prompt that was never rewritten", () => {
		const r = detachLastClip("A plain prompt.", [...REFS, { source: CLIP_VIDEO, media_type: "video" }], CLIP_VIDEO);
		expect(r).toEqual({ ok: true, prompt: "A plain prompt.", references: REFS });
	});

	it("refuses when other videos were added after the clip", () => {
		const attached = attachClip();
		const r = detachLastClip(attached.prompt, [...attached.references, { source: "/refs/extra.mp4", media_type: "video" }], CLIP_VIDEO);
		expect(r.ok).toBe(false);
	});

	it("refuses a clip that isn't attached", () => {
		expect(detachLastClip(CLIP, REFS, CLIP_VIDEO)).toEqual({ ok: false, error: "That clip isn't attached to this job." });
	});

	it("does not remove a last-frame attachment, and vice versa", () => {
		const withFrame = attach(CLIP, REFS, FRAME);
		const withBoth = attachClip(withFrame.prompt, withFrame.references, CLIP_VIDEO);
		const r = detachLastClip(withBoth.prompt, withBoth.references, CLIP_VIDEO);
		if (!r.ok) throw new Error(r.error);
		expect(attachedLastFrames(r.references)).toEqual([{ source: FRAME, jobId: "job-5" }]);
		expect(attachedLastClips(r.references)).toEqual([]);
	});
});
