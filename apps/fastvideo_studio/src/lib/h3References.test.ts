import { describe, expect, it } from "vitest";
import {
	labelReferences,
	referencePromptTemplate,
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

describe("referencePromptTemplate", () => {
	it("includes all six guide sections and cites the real labels", () => {
		const out = referencePromptTemplate([ref("image", 1), ref("video", 1)]);
		for (const section of [
			"subject_definitions:",
			"summary:",
			"retention_analysis:",
			"detailed_description:",
			"overall_soundscape:",
			"non_diegetic_music:",
		]) {
			expect(out).toContain(section);
		}
		expect(out).toContain("<Picture 1>");
		expect(out).toContain("<Video 1>");
	});

	it("mentions audio references when present", () => {
		const out = referencePromptTemplate([ref("image", 1), ref("audio", 1)]);
		expect(out).toContain("<Audio 1>");
	});
});
