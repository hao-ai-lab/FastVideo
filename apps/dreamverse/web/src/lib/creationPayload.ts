import type {
	AspectRatioId,
	CreationModeId,
	CreationModelId,
	ResolutionId,
} from "@/lib/creationConfig";
import { fromGenerationMode, type GenerationMode } from "@/lib/generationMode";

const LOBBY_MODEL_IDS = new Set<CreationModelId>(["fast-ltx2", "fast-ltx23", "fast-h3"]);
const ASPECT_RATIO_IDS = new Set<AspectRatioId>(["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]);
const RESOLUTION_IDS = new Set<ResolutionId>(["480p", "720p", "1080p", "4k"]);
const DURATION_SEC_VALUES = new Set([5, 10, 15]);

export interface EchoedSessionCreationConfig {
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
}

const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

export interface InitialImagePayload {
	name: string;
	mime_type: string;
	data_url: string;
}

export interface CreationInitPayload {
	model_id: string;
	aspect_ratio: string;
	resolution: string;
	duration_sec: number;
	initial_image: InitialImagePayload | null;
	last_frame_image: InitialImagePayload | null;
}

function readFileAsDataUrl(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => {
			if (typeof reader.result === "string") {
				resolve(reader.result);
				return;
			}
			reject(new Error("Failed to read reference image."));
		};
		reader.onerror = () => reject(new Error("Failed to read reference image."));
		reader.readAsDataURL(file);
	});
}

export async function fileToInitialImagePayload(file: File): Promise<InitialImagePayload> {
	if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
		throw new Error("Reference assets must be PNG, JPEG, or WebP images.");
	}
	if (file.size > MAX_IMAGE_BYTES) {
		throw new Error("Reference image must be 15 MB or smaller.");
	}
	return {
		name: file.name,
		mime_type: file.type,
		data_url: await readFileAsDataUrl(file),
	};
}

export async function resolveCreationImages(input: {
	modeId: CreationModeId;
	referenceFile?: File | null;
	firstFrameFile?: File | null;
	lastFrameFile?: File | null;
}): Promise<Pick<CreationInitPayload, "initial_image" | "last_frame_image">> {
	if (input.modeId === "fl2av") {
		const firstFrame = input.firstFrameFile ? await fileToInitialImagePayload(input.firstFrameFile) : null;
		const lastFrame = input.lastFrameFile ? await fileToInitialImagePayload(input.lastFrameFile) : null;
		return {
			initial_image: firstFrame,
			last_frame_image: lastFrame,
		};
	}

	const reference = input.referenceFile ? await fileToInitialImagePayload(input.referenceFile) : null;
	return {
		initial_image: reference,
		last_frame_image: null,
	};
}

export function validateCreationInputs(input: {
	modeId: CreationModeId;
	referenceFile?: File | null;
	firstFrameFile?: File | null;
	lastFrameFile?: File | null;
}): string | null {
	if (input.modeId === "ref2av" && !input.referenceFile) {
		return "Upload a reference asset to use Omni reference mode.";
	}
	if (input.modeId === "fl2av") {
		if (!input.firstFrameFile || !input.lastFrameFile) {
			return "Upload both first and last frame assets.";
		}
	}
	if (input.referenceFile && !SUPPORTED_IMAGE_TYPES.has(input.referenceFile.type)) {
		return "Reference assets must be PNG, JPEG, or WebP images.";
	}
	if (input.firstFrameFile && !SUPPORTED_IMAGE_TYPES.has(input.firstFrameFile.type)) {
		return "First frame must be a PNG, JPEG, or WebP image.";
	}
	if (input.lastFrameFile && !SUPPORTED_IMAGE_TYPES.has(input.lastFrameFile.type)) {
		return "Last frame must be a PNG, JPEG, or WebP image.";
	}
	return null;
}

export function parseEchoedCreationConfig(data: unknown): EchoedSessionCreationConfig | null {
	if (!data || typeof data !== "object") {
		return null;
	}
	const creationConfig = (data as Record<string, unknown>).creation_config;
	if (!creationConfig || typeof creationConfig !== "object") {
		return null;
	}
	const config = creationConfig as Record<string, unknown>;
	const modelId = typeof config.model_id === "string" && LOBBY_MODEL_IDS.has(config.model_id as CreationModelId)
		? (config.model_id as CreationModelId)
		: null;
	const generationMode = typeof config.generation_mode === "string" ? config.generation_mode as GenerationMode : null;
	const modeId = generationMode === "t2va" || generationMode === "fl2va" || generationMode === "ref2va"
		? fromGenerationMode(generationMode)
		: null;
	const aspectRatio = typeof config.aspect_ratio === "string" && ASPECT_RATIO_IDS.has(config.aspect_ratio as AspectRatioId)
		? (config.aspect_ratio as AspectRatioId)
		: null;
	const resolution = typeof config.resolution === "string" && RESOLUTION_IDS.has(config.resolution as ResolutionId)
		? (config.resolution as ResolutionId)
		: null;
	const durationSec = typeof config.duration_sec === "number" && DURATION_SEC_VALUES.has(config.duration_sec)
		? config.duration_sec
		: null;
	if (modelId === null || modeId === null || aspectRatio === null || resolution === null || durationSec === null) {
		return null;
	}
	return {
		modelId,
		modeId,
		aspectRatio,
		resolution,
		durationSec,
	};
}

export async function buildCreationInitPayload(input: {
	modelId: string;
	modeId: CreationModeId;
	aspectRatio: string;
	resolution: string;
	durationSec: number;
	referenceFile?: File | null;
	firstFrameFile?: File | null;
	lastFrameFile?: File | null;
}): Promise<CreationInitPayload> {
	const images = await resolveCreationImages(input);
	return {
		model_id: input.modelId,
		aspect_ratio: input.aspectRatio,
		resolution: input.resolution,
		duration_sec: input.durationSec,
		...images,
	};
}
