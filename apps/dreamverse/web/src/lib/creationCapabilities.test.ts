import { describe, expect, it } from "vitest";

import {
	DEFAULT_LOBBY_CAPABILITIES_BUNDLE,
	clampLobbySelectionToCapabilities,
	parseLobbyCapabilitiesBundle,
	resolveModelCapabilities,
	validateLobbyCreationSelection,
} from "@/lib/creationCapabilities";

describe("creationCapabilities", () => {
	it("parses backend capability payloads with per-model caps", () => {
		const bundle = parseLobbyCapabilitiesBundle({
			model_ids: ["fast-ltx2", "fast-h3"],
			models: {
				"fast-ltx2": {
					generation_modes: ["t2va"],
					resolutions: ["480p", "720p"],
					duration_sec: [5, 10],
				},
				"fast-h3": {
					generation_modes: ["t2va", "ref2va"],
					aspect_ratios: ["16:9"],
					resolutions: ["720p"],
				},
			},
		});
		expect(bundle.model_ids).toEqual(["fast-ltx2", "fast-h3"]);
		expect(bundle.models["fast-h3"]?.aspect_ratios).toEqual(["16:9"]);
	});

	it("includes fast-h3 in default lobby models", () => {
		expect(DEFAULT_LOBBY_CAPABILITIES_BUNDLE.model_ids).toContain("fast-h3");
	});

	it("clamps unsupported lobby selections to model-specific defaults", () => {
		expect(
			clampLobbySelectionToCapabilities({
				capabilities: resolveModelCapabilities(DEFAULT_LOBBY_CAPABILITIES_BUNDLE, "fast-h3"),
				modelId: "fast-h3",
				modeId: "fl2av",
				aspectRatio: "9:16",
				resolution: "4k",
				durationSec: 99,
			}),
		).toEqual({
			modelId: "fast-h3",
			modeId: "t2v",
			aspectRatio: "16:9",
			resolution: "720p",
			durationSec: 5,
		});
	});

	it("rejects unsupported generation modes with a clear message", () => {
		expect(
			validateLobbyCreationSelection({
				capabilities: resolveModelCapabilities(DEFAULT_LOBBY_CAPABILITIES_BUNDLE, "fast-ltx23"),
				modelId: "fast-ltx23",
				modeId: "fl2av",
				aspectRatio: "16:9",
				resolution: "720p",
				durationSec: 5,
			}),
		).toMatch(/FL2VA/i);
	});

	it("rejects unsupported resolutions for ltx models", () => {
		expect(
			validateLobbyCreationSelection({
				capabilities: resolveModelCapabilities(DEFAULT_LOBBY_CAPABILITIES_BUNDLE, "fast-ltx23"),
				modelId: "fast-ltx23",
				modeId: "t2v",
				aspectRatio: "16:9",
				resolution: "4k",
				durationSec: 5,
			}),
		).toMatch(/resolution/i);
	});
});
