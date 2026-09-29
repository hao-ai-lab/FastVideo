// SPDX-License-Identifier: Apache-2.0

export type JobType = "inference" | "finetuning" | "distillation";

/**
 * One razor-cut section of a clip's kept (already start/end-trimmed) range,
 * and its own color grade. `end_seconds` is seconds into that kept range, not
 * the original file's timeline; only the last section in a list may leave it
 * null, meaning "to the end". Sections are contiguous, in order, no gaps.
 */
export interface GradeSegment {
	end_seconds: number | null;
	brightness: number;
	contrast: number;
	saturation: number;
}

export interface Job {
	id: string;
	model_id: string;
	name?: string;
	prompt: string;
	job_type?: JobType;
	workload_type?: string;
	status: string;
	created_at: number;
	/** when the job joined the queue; null unless it is queued */
	queued_at?: number | null;
	started_at: number | null;
	finished_at: number | null;
	error: string | null;
	output_path: string | null;
	log_file_path: string | null;
	/** the range/sections last applied via the trim API; neutral/full-range for a never-edited job */
	edit_start_seconds?: number;
	edit_end_seconds?: number | null;
	edit_segments?: GradeSegment[];
	num_inference_steps: number;
	num_frames: number;
	/** frames per second; the API sends it, older cached objects may not have it */
	fps?: number;
	/** reference media (image/video/audio) attached to the job, as saved on the server */
	references?: { source: string; media_type: string }[] | null;
	height: number;
	width: number;
	guidance_scale: number;
	seed: number;
	num_gpus: number;
	progress: number;
	progress_msg: string;
	phase: string;
}
