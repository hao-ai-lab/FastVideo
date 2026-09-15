import { describe, expect, it } from "vitest";

import {
	CREATION_MODELS,
	buildMentionOptions,
	formatDurationLabel,
	formatResolutionLabel,
	isReferenceMediaFile,
	modeRequiresReference,
	modeUsesDualFrames,
} from "@/lib/creationConfig";

describe("creationConfig", () => {
	it("formats resolution labels", () => {
		expect(formatResolutionLabel("480p")).toBe("480P");
		expect(formatResolutionLabel("720p")).toBe("720P");
		expect(formatResolutionLabel("4k")).toBe("4K");
	});

	it("formats duration labels", () => {
		expect(formatDurationLabel(5)).toBe("5s");
	});

	it("excludes H3 from lobby models", () => {
		expect(CREATION_MODELS.map((model) => model.id)).toEqual(["fast-ltx23", "fast-ltx2"]);
	});

	it("builds mention options from presets", () => {
		expect(
			buildMentionOptions([
				{ id: "preset-a", label: "Preset A", description: "A short preset" },
				{ label: "Missing id" },
			]),
		).toEqual([
			{
				id: "preset-a",
				label: "Preset A",
				kind: "preset",
				description: "A short preset",
			},
			{
				id: "Missing id",
				label: "Missing id",
				kind: "preset",
				description: undefined,
			},
		]);
	});

	it("derives mode-specific reference requirements", () => {
		expect(modeRequiresReference("ref2av")).toBe(true);
		expect(modeRequiresReference("t2v")).toBe(false);
		expect(modeUsesDualFrames("fl2av")).toBe(true);
		expect(modeUsesDualFrames("t2v")).toBe(false);
	});

	it("accepts image and video reference files", () => {
		expect(isReferenceMediaFile(new File(["x"], "a.png", { type: "image/png" }))).toBe(true);
		expect(isReferenceMediaFile(new File(["x"], "a.mp4", { type: "video/mp4" }))).toBe(true);
		expect(isReferenceMediaFile(new File(["x"], "a.txt", { type: "text/plain" }))).toBe(false);
	});
});
