import { describe, expect, it } from "vitest";

import {
	DEFAULT_GENERATION_MODE,
	GENERATION_MODES,
	getGenerationMode,
	isGenerationMode,
	toGenerationMode,
} from "./generationMode";

describe("generation modes", () => {
	it("exposes stable wire IDs in the expected product order", () => {
		expect(GENERATION_MODES.map((mode) => mode.id)).toEqual([
			"t2va",
			"fl2va",
			"ref2va",
		]);
		expect(DEFAULT_GENERATION_MODE).toBe("t2va");
	});

	it("validates and resolves generation mode values", () => {
		expect(isGenerationMode("ref2va")).toBe(true);
		expect(isGenerationMode("unknown")).toBe(false);
		expect(getGenerationMode("fl2va").label).toBe("FL2VA");
	});

	it("maps creation studio mode IDs to upstream wire values", () => {
		expect(toGenerationMode("t2v")).toBe("t2va");
		expect(toGenerationMode("fl2av")).toBe("fl2va");
		expect(toGenerationMode("ref2av")).toBe("ref2va");
	});
});
