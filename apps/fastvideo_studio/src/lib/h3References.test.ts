import { describe, expect, it } from "vitest";
import {
	labelReferences,
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
