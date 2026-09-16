import { describe, expect, it } from "vitest";

import { parseEchoedCreationConfig, validateCreationInputs } from "@/lib/creationPayload";

describe("creationPayload", () => {
	it("requires a reference asset for omni reference mode", () => {
		expect(
			validateCreationInputs({
				modeId: "ref2av",
				referenceFile: null,
			}),
		).toMatch(/reference asset/i);
	});

	it("requires both frames for first and last frame mode", () => {
		expect(
			validateCreationInputs({
				modeId: "fl2av",
				firstFrameFile: new File(["a"], "first.png", { type: "image/png" }),
				lastFrameFile: null,
			}),
		).toMatch(/both first and last/i);
	});

	it("accepts text to video without references", () => {
		expect(
			validateCreationInputs({
				modeId: "t2v",
			}),
		).toBeNull();
	});

	it("parses echoed creation config from server payloads", () => {
		expect(
			parseEchoedCreationConfig({
				type: "gpu_assigned",
				creation_config: {
					model_id: "fast-ltx2",
					generation_mode: "ref2va",
					aspect_ratio: "9:16",
					resolution: "480p",
					duration_sec: 10,
				},
			}),
		).toEqual({
			modelId: "fast-ltx2",
			modeId: "ref2av",
			aspectRatio: "9:16",
			resolution: "480p",
			durationSec: 10,
		});
	});

	it("ignores invalid echoed creation config", () => {
		expect(parseEchoedCreationConfig({ creation_config: { model_id: "unknown" } })).toBeNull();
	});

	it("rejects unsupported reference mime types", () => {
		expect(
			validateCreationInputs({
				modeId: "t2v",
				referenceFile: new File(["a"], "clip.mp4", { type: "video/mp4" }),
			}),
		).toMatch(/PNG, JPEG, or WebP/i);
	});
});
