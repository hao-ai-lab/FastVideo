import { describe, expect, it } from "vitest";
import { jobToFormFields, referenceFileName, type JobLike } from "@/lib/jobToFields";

const job: JobLike = {
	id: "abc",
	model_id: "MiniMaxAI/MiniMax-H3",
	prompt: "subject_definitions:\n<Subject 1> is a dog.\n\nsummary:\n[reference generation] x",
	workload_type: "i2v",
	references: [
		{ source: "/uploads/9f/wukong_source.mp4", media_type: "video" },
		{ source: "/uploads/2a/MonkeyKing_0.jpg", media_type: "image" },
	],
	num_frames: 141,
	height: 768,
	width: 1344,
	guidance_scale: 1.0,
	guidance_rescale: 0.1,
	seed: 0,
	num_gpus: 4,
	use_fsdp_inference: true,
};

describe("jobToFormFields", () => {
	it("carries the settings that differ from form defaults", () => {
		const f = jobToFormFields(job);
		expect(f.numFrames).toBe(141);
		expect(f.height).toBe(768);
		expect(f.width).toBe(1344);
		expect(f.guidanceScale).toBe(1.0);
		expect(f.guidanceRescale).toBe(0.1);
		expect(f.seed).toBe(0);          // 0 must survive, not fall back to 1024
		expect(f.numGpus).toBe(4);
		expect(f.useFsdpInference).toBe(true);
	});

	it("does not let falsy-but-valid values fall through to defaults", () => {
		const f = jobToFormFields({ ...job, seed: 0, vsa_sparsity: 0, guidance_rescale: 0 });
		expect(f.seed).toBe(0);
		expect(f.vsaSparsity).toBe(0);
		expect(f.guidanceRescale).toBe(0);
	});

	it("rebuilds the reference list with readable names", () => {
		const f = jobToFormFields(job);
		expect(f.references).toHaveLength(2);
		expect(f.references[0].fileName).toBe("wukong_source.mp4");
		expect(f.references[1].media_type).toBe("image");
		expect(new Set(f.references.map((r) => r.id)).size).toBe(2);
	});

	it("splits a six-section prompt back into fields", () => {
		const f = jobToFormFields(job);
		expect(f.promptFields?.subject_definitions).toBe("<Subject 1> is a dog.");
		expect(f.promptFields?.summary).toBe("[reference generation] x");
	});

	it("returns null promptFields for a plain prompt", () => {
		const f = jobToFormFields({ ...job, prompt: "a toy car" });
		expect(f.promptFields).toBeNull();
		expect(f.prompt).toBe("a toy car");
	});

	it("handles a job with no references", () => {
		const f = jobToFormFields({ ...job, references: null });
		expect(f.references).toEqual([]);
	});
});

describe("referenceFileName", () => {
	it("takes the basename", () => {
		expect(referenceFileName("/a/b/c.mp4")).toBe("c.mp4");
		expect(referenceFileName("c.mp4")).toBe("c.mp4");
	});
});
