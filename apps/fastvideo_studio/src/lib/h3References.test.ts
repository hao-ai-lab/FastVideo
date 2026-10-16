import { describe, expect, it } from "vitest";
import {
	labelReferences,
	moveReference,
	referencePromptSeed,
	validateReferences,
	type H3Reference,
} from "@/lib/h3References";

const ref = (media_type: H3Reference["media_type"], n: number): H3Reference => ({
	id: `${media_type}-${n}`,
	source: `/tmp/${media_type}${n}`,
	media_type,
	fileName: `${media_type}${n}`,
});

describe("labelReferences", () => {
	it("numbers each media type independently, in list order", () => {
		expect(
			labelReferences([ref("image", 1), ref("video", 1), ref("image", 2)]),
		).toEqual(["<Picture 1>", "<Video 1>", "<Picture 2>"]);
	});
});

describe("validateReferences", () => {
	it("accepts an empty list and a normal mix", () => {
		expect(validateReferences([])).toBeNull();
		expect(validateReferences([ref("image", 1), ref("audio", 1)])).toBeNull();
	});

	it("rejects audio-only lists", () => {
		expect(validateReferences([ref("audio", 1)])).toMatch(/paired/);
	});

	it("enforces the per-type caps", () => {
		const videos = [1, 2, 3, 4].map((n) => ref("video", n));
		expect(validateReferences(videos)).toMatch(/At most 3 video/);
	});

	it("enforces the overall cap", () => {
		const many = Array.from({ length: 13 }, (_, i) => ref("image", i));
		expect(validateReferences(many)).toMatch(/At most 12/);
	});
});

describe("referencePromptSeed", () => {
	it("cites the real labels and never indents content", () => {
		const seed = referencePromptSeed([ref("image", 1), ref("video", 1)]);
		expect(seed.subject_definitions).toContain("<Picture 1>");
		expect(seed.subject_definitions).toContain("<Video 1>");
		for (const value of Object.values(seed)) {
			expect(value).not.toMatch(/^ {2}\S/m);
		}
	});

	it("uses the guide's retention_analysis form", () => {
		const seed = referencePromptSeed([ref("image", 1)]);
		expect(seed.retention_analysis).toMatch(/fully_preserved - /);
	});

	it("starts the summary with a bracketed task type", () => {
		const seed = referencePromptSeed([ref("image", 1)]);
		expect(seed.summary).toMatch(/^\[[a-z +]+\]/);
	});

	it("describes audio references separately", () => {
		const seed = referencePromptSeed([ref("image", 1), ref("audio", 1)]);
		expect(seed.subject_definitions).toContain("<Audio 1>");
		expect(seed.retention_analysis).toContain("<Audio 1>: reference -");
	});
});

describe("moveReference", () => {
	const refs = [ref("video", 1), ref("image", 1), ref("audio", 1)];

	it("moves a reference and renumbers labels by the new order", () => {
		const moved = moveReference(refs, 1, -1);
		expect(moved.map((r) => r.id)).toEqual(["image-1", "video-1", "audio-1"]);
		expect(labelReferences(moved)).toEqual(["<Picture 1>", "<Video 1>", "<Audio 1>"]);
		expect(refs.map((r) => r.id)).toEqual(["video-1", "image-1", "audio-1"]);
	});

	it("ignores moves past either end", () => {
		expect(moveReference(refs, 0, -1)).toBe(refs);
		expect(moveReference(refs, 2, 1)).toBe(refs);
	});
});

describe("validateReferences with a model capability", () => {
	const required = {
		media_types: ["image", "video", "audio"] as H3Reference["media_type"][],
		limits: { image: 2, video: 1, audio: 1 },
		max_total: 3,
		requires_visual: true,
		required: true,
	};

	it("requires a reference when the model does", () => {
		expect(validateReferences([], required)).toMatch(/at least one image or video/);
	});

	it("uses the model's limits", () => {
		expect(validateReferences([ref("video", 1), ref("video", 2)], required)).toMatch(
			/At most 1 video/,
		);
		expect(
			validateReferences([ref("image", 1), ref("image", 2), ref("video", 1), ref("audio", 1)], required),
		).toMatch(/At most 3 references/);
	});
});
