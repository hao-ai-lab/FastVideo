// SPDX-License-Identifier: Apache-2.0

import type { ModelRecipe } from "./modelCapabilities";

export type JobType = "inference" | "finetuning" | "distillation";

export interface Job {
	id: string;
	model_id: string;
	name?: string;
	prompt: string;
	job_type?: JobType;
	workload_type?: string;
	status: string;
	created_at: number;
	started_at: number | null;
	finished_at: number | null;
	error: string | null;
	/** weights / parameters / kernel / memory / runtime, for failed jobs. */
	error_category?: string | null;
	recipe?: ModelRecipe | null;
	references?: { source: string; media_type: string }[];
	output_path: string | null;
	log_file_path: string | null;
	num_inference_steps: number;
	num_frames: number;
	height: number;
	width: number;
	guidance_scale: number;
	seed: number;
	num_gpus: number;
	progress: number;
	progress_msg: string;
	phase: string;
}
