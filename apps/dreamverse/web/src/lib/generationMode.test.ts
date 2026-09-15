import { describe, expect, it } from "vitest";

import {
	DEFAULT_GENERATION_MODE,
	GENERATION_MODES,
	getGenerationMode,
	isGenerationMode,
	buildGenerationInitFields,
	validateGenerationInputs,
	type GenerationAsset,
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
});

const image: GenerationAsset = { asset_id: "img", kind: "image", name: "frame.png", mime_type: "image/png", size: 12, url: "/assets/img" };
const audio: GenerationAsset = { asset_id: "sound", kind: "audio", name: "sound.wav", mime_type: "audio/wav", size: 12, url: "/assets/sound" };

describe("generation input contract", () => {
	it("keeps text-only init valid and rejects accidental references", () => {
		expect(buildGenerationInitFields("t2va", [], [])).toEqual({ generation_mode: "t2va", conditioning_assets: [] });
		expect(() => buildGenerationInitFields("t2va", [{ asset_id: "img", role: "reference" }], [image])).toThrow("text only");
	});
	it("requires first frame but permits first-only or both endpoints", () => {
		expect(validateGenerationInputs("fl2va", [], [])).toMatch(/first frame/);
		const first = { asset_id: "img", role: "first_frame" } as const;
		expect(validateGenerationInputs("fl2va", [first], [image])).toBeNull();
		expect(validateGenerationInputs("fl2va", [first, { asset_id: "img", role: "last_frame" }], [image])).toBeNull();
		expect(validateGenerationInputs("fl2va", [first, first], [image])).toMatch(/one first frame/);
		expect(validateGenerationInputs("fl2va", [{ asset_id: "sound", role: "first_frame" }], [audio])).toMatch(/images only/);
	});
	it("requires a visual reference and preserves multimodal ordering without file bodies", () => {
		expect(validateGenerationInputs("ref2va", [{ asset_id: "sound", role: "reference" }], [audio])).toMatch(/image or video/);
		const items = [{ asset_id: "sound", role: "reference" }, { asset_id: "img", role: "reference" }] as const;
		expect(buildGenerationInitFields("ref2va", items, [image, audio])).toEqual({ generation_mode: "ref2va", conditioning_assets: items });
	});
	it("rejects per-kind limits, total limits, and stale uploads", () => {
		const images = Array.from({ length: 10 }, (_, index) => ({ ...image, asset_id: `img-${index}` }));
		expect(validateGenerationInputs("ref2va", images.map((item) => ({ asset_id: item.asset_id, role: "reference" })), images)).toMatch(/at most 9 image/);
		const refs = Array.from({ length: 13 }, () => ({ asset_id: "img", role: "reference" as const }));
		expect(validateGenerationInputs("ref2va", refs, [image])).toMatch(/at most 12/);
		expect(validateGenerationInputs("fl2va", [{ asset_id: "img", role: "first_frame" }], [{ ...image, missing: true }])).toMatch(/no longer on the server/);
	});
});
