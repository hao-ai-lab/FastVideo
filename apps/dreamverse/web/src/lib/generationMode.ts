export const GENERATION_MODES = [
	{
		id: "t2va",
		label: "T2VA",
		name: "Text to video + audio",
		description: "Start with a text prompt; no reference asset is required.",
	},
	{
		id: "fl2va",
		label: "FL2VA",
		name: "First/last frames to video + audio",
		description: "Start from a first frame image. Add an optional last frame to guide the ending.",
	},
	{
		id: "ref2va",
		label: "Ref2VA",
		name: "References to video + audio",
		description: "Guide the result with ordered image, video, or audio references.",
	},
] as const;

export type GenerationMode = (typeof GENERATION_MODES)[number]["id"];

export const DEFAULT_GENERATION_MODE: GenerationMode = "t2va";

export function isGenerationMode(value: unknown): value is GenerationMode {
	return GENERATION_MODES.some((mode) => mode.id === value);
}

export function getGenerationMode(value: GenerationMode) {
	return GENERATION_MODES.find((mode) => mode.id === value) ?? GENERATION_MODES[0];
}

export type AssetKind = "image" | "video" | "audio";
export type ConditioningRole = "first_frame" | "last_frame" | "reference";

/** Runtime-owned uploads; project metadata keeps references, never file contents. */
export interface GenerationAsset {
	asset_id: string;
	kind: AssetKind;
	name: string;
	mime_type: string;
	size: number;
	url: string;
	missing?: boolean;
}

export interface ConditioningAsset {
	asset_id: string;
	role: ConditioningRole;
}

export interface GenerationCapabilities {
	model_id: string;
	modes: GenerationMode[];
	mock?: boolean;
}

export interface GenerationInitFields {
	generation_mode: GenerationMode;
	conditioning_assets: ConditioningAsset[];
}

export const REFERENCE_LIMITS = { image: 9, video: 3, audio: 3, total: 12 } as const;

export function validateGenerationInputs(
	mode: GenerationMode,
	conditioning: readonly ConditioningAsset[],
	assets: readonly GenerationAsset[],
): string | null {
	if (mode === "t2va") {
		return conditioning.length ? "T2VA uses text only. Remove the selected references." : null;
	}
	const resolved = conditioning.map((item) => assets.find((asset) => asset.asset_id === item.asset_id));
	if (resolved.some((asset) => !asset || asset.missing)) {
		return "A selected asset is no longer on the server. Upload it again and select the new copy.";
	}
	if (mode === "fl2va") {
		if (!conditioning.some((item) => item.role === "first_frame")) return "Choose a first frame image to generate.";
		if (conditioning.some((item) => item.role === "reference") || resolved.some((asset) => asset?.kind !== "image")) {
			return "FL2VA accepts first and last frame images only.";
		}
		if (conditioning.filter((item) => item.role === "first_frame").length !== 1
			|| conditioning.filter((item) => item.role === "last_frame").length > 1) {
			return "Choose one first frame and at most one last frame.";
		}
		return null;
	}
	if (conditioning.some((item) => item.role !== "reference")) return "Ref2VA accepts ordered reference assets only.";
	if (!resolved.some((asset) => asset?.kind === "image" || asset?.kind === "video")) {
		return "Add at least one image or video reference. Audio alone is not enough.";
	}
	if (conditioning.length > REFERENCE_LIMITS.total) return "Use at most 12 reference assets in total.";
	for (const kind of ["image", "video", "audio"] as const) {
		if (resolved.filter((asset) => asset?.kind === kind).length > REFERENCE_LIMITS[kind]) {
			return `Use at most ${REFERENCE_LIMITS[kind]} ${kind} references.`;
		}
	}
	return null;
}

/** Shared by both session_init_v2 and project_init_v1. */
export function buildGenerationInitFields(
	mode: GenerationMode,
	conditioning: readonly ConditioningAsset[],
	assets: readonly GenerationAsset[],
): GenerationInitFields {
	const problem = validateGenerationInputs(mode, conditioning, assets);
	if (problem) throw new Error(problem);
	return {
		generation_mode: mode,
		conditioning_assets: conditioning.map(({ asset_id, role }) => ({ asset_id, role })),
	};
}
