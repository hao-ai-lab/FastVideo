import type {
	AspectRatioId,
	CreationModeId,
	CreationModelId,
	ResolutionId,
} from "@/lib/creationConfig";
import { fromGenerationMode, toGenerationMode, type GenerationMode } from "@/lib/generationMode";

const ALL_MODEL_IDS: CreationModelId[] = ["fast-ltx23", "fast-ltx2", "fast-h3"];
const ALL_GENERATION_MODES: GenerationMode[] = ["t2va", "fl2va", "ref2va"];
const ALL_ASPECT_RATIOS: AspectRatioId[] = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const ALL_RESOLUTIONS: ResolutionId[] = ["480p", "720p", "1080p", "4k"];

export interface ModelCreationCapabilities {
	generation_modes: GenerationMode[];
	aspect_ratios: AspectRatioId[];
	resolutions: ResolutionId[];
	duration_sec: number[];
	unsupported_generation_modes: Record<string, string>;
	reference_assets: {
		mime_types: string[];
		max_bytes: number;
	};
}

export interface LobbyCreationCapabilities extends ModelCreationCapabilities {
	model_ids: CreationModelId[];
}

export interface LobbyCapabilitiesBundle {
	model_ids: CreationModelId[];
	models: Partial<Record<CreationModelId, ModelCreationCapabilities>>;
	generation_modes: GenerationMode[];
	aspect_ratios: AspectRatioId[];
	resolutions: ResolutionId[];
	duration_sec: number[];
	unsupported_generation_modes: Record<string, string>;
	reference_assets: {
		mime_types: string[];
		max_bytes: number;
	};
}

const DEFAULT_LTX_MODEL_CAPABILITIES: ModelCreationCapabilities = {
	generation_modes: ["t2va", "ref2va"],
	aspect_ratios: ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
	resolutions: ["480p", "720p", "1080p"],
	duration_sec: [5, 10, 15],
	unsupported_generation_modes: {
		fl2va: "First/last frame mode (FL2VA) is not supported yet.",
	},
	reference_assets: {
		mime_types: ["image/png", "image/jpeg", "image/webp"],
		max_bytes: 15 * 1024 * 1024,
	},
};

const DEFAULT_H3_MODEL_CAPABILITIES: ModelCreationCapabilities = {
	generation_modes: ["t2va", "ref2va"],
	aspect_ratios: ["16:9"],
	resolutions: ["720p"],
	duration_sec: [5, 10, 15],
	unsupported_generation_modes: {
		fl2va: "First/last frame mode (FL2VA) is not supported yet.",
	},
	reference_assets: DEFAULT_LTX_MODEL_CAPABILITIES.reference_assets,
};

export const DEFAULT_LOBBY_CAPABILITIES_BUNDLE: LobbyCapabilitiesBundle = {
	model_ids: ALL_MODEL_IDS,
	models: {
		"fast-ltx2": DEFAULT_LTX_MODEL_CAPABILITIES,
		"fast-ltx23": DEFAULT_LTX_MODEL_CAPABILITIES,
		"fast-h3": DEFAULT_H3_MODEL_CAPABILITIES,
	},
	generation_modes: ["t2va", "ref2va"],
	aspect_ratios: ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
	resolutions: ["480p", "720p", "1080p"],
	duration_sec: [5, 10, 15],
	unsupported_generation_modes: DEFAULT_LTX_MODEL_CAPABILITIES.unsupported_generation_modes,
	reference_assets: DEFAULT_LTX_MODEL_CAPABILITIES.reference_assets,
};

function pickStrings<T extends string>(value: unknown, allowed: readonly T[], fallback: readonly T[]): T[] {
	if (!Array.isArray(value)) return [...fallback];
	return value.filter((item): item is T => typeof item === "string" && allowed.includes(item as T));
}

function parseReferenceAssets(
	value: unknown,
	fallback: ModelCreationCapabilities["reference_assets"],
): ModelCreationCapabilities["reference_assets"] {
	if (!value || typeof value !== "object") return fallback;
	const data = value as Record<string, unknown>;
	return {
		mime_types: Array.isArray(data.mime_types)
			? (data.mime_types as string[])
			: fallback.mime_types,
		max_bytes: typeof data.max_bytes === "number" ? data.max_bytes : fallback.max_bytes,
	};
}

function parseModelCreationCapabilities(
	value: unknown,
	fallback: ModelCreationCapabilities,
): ModelCreationCapabilities {
	if (!value || typeof value !== "object") return fallback;
	const data = value as Record<string, unknown>;
	return {
		generation_modes: pickStrings(data.generation_modes, ALL_GENERATION_MODES, fallback.generation_modes),
		aspect_ratios: pickStrings(data.aspect_ratios, ALL_ASPECT_RATIOS, fallback.aspect_ratios),
		resolutions: pickStrings(data.resolutions, ALL_RESOLUTIONS, fallback.resolutions),
		duration_sec: Array.isArray(data.duration_sec)
			? data.duration_sec.filter((item): item is number => typeof item === "number")
			: fallback.duration_sec,
		unsupported_generation_modes:
			typeof data.unsupported_generation_modes === "object" && data.unsupported_generation_modes
				? (data.unsupported_generation_modes as Record<string, string>)
				: fallback.unsupported_generation_modes,
		reference_assets: parseReferenceAssets(data.reference_assets, fallback.reference_assets),
	};
}

export function parseLobbyCapabilitiesBundle(payload: unknown): LobbyCapabilitiesBundle {
	if (!payload || typeof payload !== "object") {
		return DEFAULT_LOBBY_CAPABILITIES_BUNDLE;
	}
	const data = payload as Record<string, unknown>;
	const modelIds = pickStrings(data.model_ids, ALL_MODEL_IDS, DEFAULT_LOBBY_CAPABILITIES_BUNDLE.model_ids);
	const rawModels = typeof data.models === "object" && data.models ? (data.models as Record<string, unknown>) : {};
	const models: Partial<Record<CreationModelId, ModelCreationCapabilities>> = {};
	for (const modelId of modelIds) {
		const fallback =
			DEFAULT_LOBBY_CAPABILITIES_BUNDLE.models[modelId] ??
			(modelId === "fast-h3" ? DEFAULT_H3_MODEL_CAPABILITIES : DEFAULT_LTX_MODEL_CAPABILITIES);
		models[modelId] = parseModelCreationCapabilities(rawModels[modelId], fallback);
	}
	const unionFallback = parseModelCreationCapabilities(payload, DEFAULT_LTX_MODEL_CAPABILITIES);
	return {
		model_ids: modelIds,
		models,
		generation_modes: unionFallback.generation_modes,
		aspect_ratios: unionFallback.aspect_ratios,
		resolutions: unionFallback.resolutions,
		duration_sec: unionFallback.duration_sec,
		unsupported_generation_modes: unionFallback.unsupported_generation_modes,
		reference_assets: unionFallback.reference_assets,
	};
}

export function resolveModelCapabilities(
	bundle: LobbyCapabilitiesBundle,
	modelId: CreationModelId,
): LobbyCreationCapabilities {
	const modelCaps =
		bundle.models[modelId] ??
		(modelId === "fast-h3" ? DEFAULT_H3_MODEL_CAPABILITIES : DEFAULT_LTX_MODEL_CAPABILITIES);
	return {
		model_ids: bundle.model_ids,
		...modelCaps,
	};
}

export function supportedCreationModes(capabilities: LobbyCreationCapabilities) {
	return capabilities.generation_modes.map((wireMode) => ({
		wireMode,
		modeId: fromGenerationMode(wireMode),
	}));
}

export function isSupportedCreationMode(modeId: CreationModeId, capabilities: LobbyCreationCapabilities): boolean {
	return capabilities.generation_modes.includes(toGenerationMode(modeId));
}

export function isSupportedResolution(resolution: ResolutionId, capabilities: LobbyCreationCapabilities): boolean {
	return capabilities.resolutions.includes(resolution);
}

export function isSupportedReferenceImage(file: File, capabilities: LobbyCreationCapabilities): boolean {
	return capabilities.reference_assets.mime_types.includes(file.type);
}

export function unsupportedModeNotice(modeId: CreationModeId, capabilities: LobbyCreationCapabilities): string | null {
	const wireMode = toGenerationMode(modeId);
	return capabilities.unsupported_generation_modes[wireMode] ?? null;
}

export function clampLobbySelectionToCapabilities(input: {
	capabilities: LobbyCreationCapabilities;
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
}): {
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
} {
	const { capabilities } = input;
	const modelId = capabilities.model_ids.includes(input.modelId)
		? input.modelId
		: (capabilities.model_ids[0] ?? "fast-ltx23");
	const supportedModes = supportedCreationModes(capabilities);
	const modeId = isSupportedCreationMode(input.modeId, capabilities)
		? input.modeId
		: (supportedModes[0]?.modeId ?? "t2v");
	const aspectRatio = capabilities.aspect_ratios.includes(input.aspectRatio)
		? input.aspectRatio
		: (capabilities.aspect_ratios[0] ?? "16:9");
	const resolution = isSupportedResolution(input.resolution, capabilities)
		? input.resolution
		: (capabilities.resolutions[0] ?? "720p");
	const durationSec = capabilities.duration_sec.includes(input.durationSec)
		? input.durationSec
		: (capabilities.duration_sec[0] ?? 5);
	return { modelId, modeId, aspectRatio, resolution, durationSec };
}

export function validateLobbyCreationSelection(input: {
	capabilities: LobbyCreationCapabilities;
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
	referenceFile?: File | null;
	firstFrameFile?: File | null;
	lastFrameFile?: File | null;
}): string | null {
	const unsupportedMode = unsupportedModeNotice(input.modeId, input.capabilities);
	if (unsupportedMode) return unsupportedMode;
	if (!input.capabilities.model_ids.includes(input.modelId)) {
		return "Selected model is not supported yet.";
	}
	if (!isSupportedCreationMode(input.modeId, input.capabilities)) {
		return "Selected mode is not supported yet.";
	}
	if (!input.capabilities.aspect_ratios.includes(input.aspectRatio)) {
		return "Selected aspect ratio is not supported for this model yet.";
	}
	if (!isSupportedResolution(input.resolution, input.capabilities)) {
		return "Selected resolution is not supported for this model yet.";
	}
	if (!input.capabilities.duration_sec.includes(input.durationSec)) {
		return "Selected duration is not supported yet.";
	}
	if (input.modeId === "ref2av" && !input.referenceFile) {
		return "Upload a reference image to use reference-guided mode.";
	}
	if (input.referenceFile && !isSupportedReferenceImage(input.referenceFile, input.capabilities)) {
		return "Reference assets must be PNG, JPEG, or WebP images.";
	}
	if (input.firstFrameFile && !isSupportedReferenceImage(input.firstFrameFile, input.capabilities)) {
		return "First frame must be a PNG, JPEG, or WebP image.";
	}
	if (input.lastFrameFile && !isSupportedReferenceImage(input.lastFrameFile, input.capabilities)) {
		return "Last frame must be a PNG, JPEG, or WebP image.";
	}
	return null;
}
