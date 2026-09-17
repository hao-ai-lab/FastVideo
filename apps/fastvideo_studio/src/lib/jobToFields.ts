/**
 * Map a persisted job back onto the create-job form's fields. Every value the
 * form seeds from defaults must be covered here, or edit mode silently shows
 * defaults for whatever is missing.
 */
import type { H3Reference } from "@/lib/h3References";
import { parseH3Prompt, type H3PromptFields } from "@/lib/h3Prompt";

export interface JobLike {
	id: string;
	model_id: string;
	name?: string;
	prompt: string;
	workload_type?: string;
	job_type?: string;
	image_path?: string;
	last_image_path?: string;
	references?: { source: string; media_type: string }[] | null;
	negative_prompt?: string;
	num_inference_steps?: number;
	num_frames?: number;
	height?: number;
	width?: number;
	guidance_scale?: number;
	guidance_rescale?: number;
	fps?: number;
	seed?: number;
	num_gpus?: number;
	dit_cpu_offload?: boolean;
	dit_layerwise_offload?: boolean;
	text_encoder_cpu_offload?: boolean;
	vae_cpu_offload?: boolean;
	image_encoder_cpu_offload?: boolean;
	use_fsdp_inference?: boolean;
	enable_torch_compile?: boolean;
	vsa_sparsity?: number;
	tp_size?: number;
	sp_size?: number;
	data_path?: string;
	validation_dataset_file?: string;
	max_train_steps?: number;
	train_batch_size?: number;
	learning_rate?: number;
	num_latent_t?: number;
	lora_rank?: number;
	dmd_use_vsa?: boolean;
	dmd_vsa_sparsity?: number;
	dmd_denoising_steps?: string;
	real_score_guidance_scale?: number;
	generator_update_interval?: number;
	real_score_model_path?: string;
	fake_score_model_path?: string;
}

export interface JobFormFields {
	modelId: string;
	name: string;
	workloadType: string;
	jobType: string;
	prompt: string;
	negativePrompt: string;
	imagePath: string;
	lastImagePath: string;
	references: H3Reference[];
	promptFields: H3PromptFields | null;
	numInferenceSteps: number;
	numFrames: number;
	height: number;
	width: number;
	guidanceScale: number;
	guidanceRescale: number;
	fps: number;
	seed: number;
	numGpus: number;
	ditCpuOffload: boolean;
	ditLayerwiseOffload: boolean;
	textEncoderCpuOffload: boolean;
	vaeCpuOffload: boolean;
	imageEncoderCpuOffload: boolean;
	useFsdpInference: boolean;
	enableTorchCompile: boolean;
	vsaSparsity: number;
	tpSize: number;
	spSize: number;
	dataPath: string;
	validationDatasetFile: string;
	maxTrainSteps: number;
	trainBatchSize: number;
	learningRate: number;
	numLatentT: number;
	loraRank: number;
	dmdUseVsa: boolean;
	dmdVsaSparsity: number;
	dmdDenoisingSteps: string;
	realScoreGuidanceScale: number;
	generatorUpdateInterval: number;
	realScoreModelPath: string;
	fakeScoreModelPath: string;
}

/** Uploads keep their original basename, so this is the display name. */
export function referenceFileName(source: string): string {
	return source.split("/").filter(Boolean).pop() ?? source;
}

export function jobToFormFields(job: JobLike): JobFormFields {
	const refs: H3Reference[] = (job.references ?? []).map((r, i) => ({
		id: `${job.id}-${i}`,
		source: r.source,
		media_type: r.media_type as H3Reference["media_type"],
		fileName: referenceFileName(r.source),
	}));

	return {
		modelId: job.model_id,
		name: job.name ?? "",
		workloadType: job.workload_type ?? "t2v",
		jobType: job.job_type ?? "inference",
		prompt: job.prompt ?? "",
		negativePrompt: job.negative_prompt ?? "",
		imagePath: job.image_path ?? "",
		lastImagePath: job.last_image_path ?? "",
		references: refs,
		// null when the prompt is not in six-section form; the caller then keeps
		// the raw editor rather than silently dropping content into fields.
		promptFields: parseH3Prompt(job.prompt ?? ""),
		numInferenceSteps: job.num_inference_steps ?? 50,
		numFrames: job.num_frames ?? 81,
		height: job.height ?? 480,
		width: job.width ?? 832,
		guidanceScale: job.guidance_scale ?? 5.0,
		guidanceRescale: job.guidance_rescale ?? 0.0,
		fps: job.fps ?? 24,
		seed: job.seed ?? 1024,
		numGpus: job.num_gpus ?? 1,
		ditCpuOffload: job.dit_cpu_offload ?? false,
		ditLayerwiseOffload: job.dit_layerwise_offload ?? false,
		textEncoderCpuOffload: job.text_encoder_cpu_offload ?? false,
		vaeCpuOffload: job.vae_cpu_offload ?? false,
		imageEncoderCpuOffload: job.image_encoder_cpu_offload ?? false,
		useFsdpInference: job.use_fsdp_inference ?? false,
		enableTorchCompile: job.enable_torch_compile ?? false,
		vsaSparsity: job.vsa_sparsity ?? 0,
		tpSize: job.tp_size ?? -1,
		spSize: job.sp_size ?? -1,
		dataPath: job.data_path ?? "",
		validationDatasetFile: job.validation_dataset_file ?? "",
		maxTrainSteps: job.max_train_steps ?? 1000,
		trainBatchSize: job.train_batch_size ?? 1,
		learningRate: job.learning_rate ?? 5e-5,
		numLatentT: job.num_latent_t ?? 20,
		loraRank: job.lora_rank ?? 32,
		dmdUseVsa: job.dmd_use_vsa ?? false,
		dmdVsaSparsity: job.dmd_vsa_sparsity ?? 0.8,
		dmdDenoisingSteps: job.dmd_denoising_steps ?? "1000,757,522",
		realScoreGuidanceScale: job.real_score_guidance_scale ?? 3.5,
		generatorUpdateInterval: job.generator_update_interval ?? 5,
		realScoreModelPath: job.real_score_model_path ?? "",
		fakeScoreModelPath: job.fake_score_model_path ?? "",
	};
}
