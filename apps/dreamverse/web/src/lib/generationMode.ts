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
		description: "Provide first and last frame images to control the transition.",
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
