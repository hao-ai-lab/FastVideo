/**
 * What a model accepts, as reported by GET /api/models. The create-job form
 * decides which inputs to show from these fields, never from the model name.
 */
import type { MediaType } from "@/lib/api";

export interface ReferenceCapability {
	media_types: MediaType[];
	limits: Record<MediaType, number>;
	max_total: number;
	/** At least one image or video must be present when any reference is. */
	requires_visual: boolean;
	/** The model only runs with references (FastH3-Omni). */
	required: boolean;
}

export interface ModelCapabilities {
	references: ReferenceCapability | null;
	start_frame: boolean;
	end_frame: boolean;
	max_frames: number | null;
	/** GPU counts the model can split across, or null for any. */
	num_gpus: number[] | null;
	fixed_fps?: number;
}

/** A FastH3 Ref2VA PDD export's fastvideo_inference.json. */
export interface FastH3Contract {
	pdd_steps: number;
	pdd_step_indices: number[];
	num_inference_steps: number;
	transformer_forwards: number;
	guidance_scale: number;
	attention_backend: string;
	vsa_sparsity: number;
	vsa_tile_size: number;
	vsa_ref_keep_rate: number;
	video_scheduler_shift: number;
	audio_scheduler_shift: number;
	base_model_revision: string;
}

export interface ModelRecipe {
	kind: string;
	source: string;
	export: string;
	export_revision: string | null;
	base: string;
	base_revision: string | null;
	contract_sha256: string;
	contract: FastH3Contract;
	/** Job fields the server sets from the recipe, whatever the form sends. */
	locked_fields: string[];
	/** Set once a run has composed the model. */
	model_dir?: string;
}

export interface ModelHardware {
	/** null when the server could not read its GPUs. */
	supported: boolean | null;
	reason: string;
}

export interface RecipeDefaults {
	height: number;
	width: number;
	num_frames: number;
	num_gpus: number;
	text_encoder_cpu_offload: boolean;
	vae_cpu_offload: boolean;
}

/** Older servers report only id and label: every input stays as it was. */
export const DEFAULT_CAPABILITIES: ModelCapabilities = {
	references: null,
	start_frame: true,
	end_frame: false,
	max_frames: null,
	num_gpus: null,
};

export function capabilitiesOf(
	model: { capabilities?: ModelCapabilities | null } | undefined,
): ModelCapabilities {
	return model?.capabilities ?? DEFAULT_CAPABILITIES;
}

export function shortRevision(revision: string | null | undefined): string {
	if (!revision) return "";
	return /^[0-9a-f]{40}$/.test(revision) ? revision.slice(0, 12) : revision;
}

/** "source@revision", with commit hashes shortened. */
export function formatSource(source: string, revision: string | null): string {
	return revision ? `${source}@${shortRevision(revision)}` : source;
}

/** One-line summary for job cards, e.g. "8 forwards · VSA 0.9 · tile 128". */
export function recipeChip(recipe: ModelRecipe): string {
	const c = recipe.contract;
	return `${c.transformer_forwards} forwards · VSA ${c.vsa_sparsity} · tile ${c.vsa_tile_size}`;
}

/** Rows for the read-only recipe panel. */
export function recipeRows(
	recipe: ModelRecipe,
	numGpus: number,
): { label: string; value: string }[] {
	const c = recipe.contract;
	const rows = [
		{
			label: "Source",
			value: `${recipe.source} (sha256 ${recipe.contract_sha256.slice(0, 12)})`,
		},
		{
			label: "Export",
			value: formatSource(recipe.export, recipe.export_revision),
		},
		{
			label: "Base model",
			value: formatSource(recipe.base, recipe.base_revision),
		},
		{
			// PDD-N counts fused blocks (one forward each), unlike a DMD ladder
			// whose step count includes the terminal grid point.
			label: "Sampling",
			value: `${c.num_inference_steps} fused blocks of a ${c.pdd_steps}-interval grid = ${c.transformer_forwards} transformer forwards`,
		},
		{
			label: "Guidance",
			value: `CFG ${c.guidance_scale.toFixed(1)} (no classifier-free guidance; negative prompt unused)`,
		},
		{
			label: "Attention",
			value: `${c.attention_backend} · sparsity ${c.vsa_sparsity} · ${c.vsa_tile_size}-token tiles · reference keep rate ${c.vsa_ref_keep_rate}`,
		},
		{
			label: "Scheduler shifts",
			value: `video ${c.video_scheduler_shift} · audio ${c.audio_scheduler_shift}`,
		},
		{
			label: "Parallelism",
			value:
				numGpus > 1
					? `${numGpus} GPUs: sequence parallel, DiT sharded (FSDP)`
					: "1 GPU",
		},
	];
	if (recipe.model_dir) {
		rows.push({ label: "Composed model", value: recipe.model_dir });
	}
	return rows;
}

export const ERROR_CATEGORY_LABELS: Record<string, string> = {
	weights: "Weights",
	parameters: "Parameters",
	kernel: "Kernel / hardware",
	memory: "GPU memory",
	runtime: "Runtime",
};
