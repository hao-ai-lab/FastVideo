// SPDX-License-Identifier: Apache-2.0

import {
	STORAGE_KEY,
	loadDefaultOptions,
	type DefaultOptions,
} from "./defaultOptions";
import type { Job, JobType } from "./types";

const DEFAULT_API_BASE_URL = "http://localhost:8189/api";

// getApiBaseUrl runs on every request and in render paths (media URL
// builders), so cache the configured URL keyed on the raw stored options
// string instead of re-parsing the options JSON each call.
let cachedRaw: string | null = null;
let cachedConfigured = "";

export function getApiBaseUrl(): string {
	// 1) Check user-configured value from the persisted options (Settings
	// page). The settings sync persists the full options object, so the key
	// holds an empty string until the user sets a URL — treat that as unset
	// and fall through to the env/default below.
	if (typeof window !== "undefined") {
		try {
			const raw = window.localStorage.getItem(STORAGE_KEY);
			if (raw !== cachedRaw) {
				cachedRaw = raw;
				cachedConfigured = stripTrailingSlash(
					loadDefaultOptions().apiServerBaseUrl.trim(),
				);
			}
			if (cachedConfigured) return cachedConfigured;
		} catch {
			// Ignore storage errors and fall back to env/default
		}
	}

	// 2) Fall back to env var if set, then hardcoded default
	const fromEnv = process.env.NEXT_PUBLIC_API_BASE_URL;
	const apiUrl =
		typeof fromEnv === "string" && fromEnv ? fromEnv : DEFAULT_API_BASE_URL;
	return stripTrailingSlash(apiUrl);
}

// The base URL is used as a prefix in `${base}/jobs` template strings, so a
// trailing slash (a natural thing to paste into Settings) would produce `//`
// and 404 every request.
function stripTrailingSlash(url: string): string {
	return url.replace(/\/+$/, "");
}

/** Full URL for streaming a job's video/image output (for &lt;video&gt; or &lt;img&gt; src). */
/**
 * `version` busts the browser's cache after editing a job's video in place
 * (Trim/Edit video): the URL is otherwise identical before and after an edit,
 * even though the file on disk changed, so without this the same cached
 * response keeps showing. Omit it where staleness doesn't matter.
 */
export function getJobVideoUrl(jobId: string, version?: number): string {
	const base = `${getApiBaseUrl()}/jobs/${jobId}/video`;
	return version ? `${base}?v=${version}` : base;
}

export interface CreateJobRequest {
	model_id: string;
	/** Optional label; the card and output filename fall back to the prompt. */
	name?: string;
	prompt: string;
	workload_type?: string;
	job_type?: JobType;
	image_path?: string;
	data_path?: string;
	max_train_steps?: number;
	train_batch_size?: number;
	learning_rate?: number;
	num_latent_t?: number;
	validation_dataset_file?: string;
	lora_rank?: number;
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
	dit_cpu_offload?: boolean | null;
	text_encoder_cpu_offload?: boolean | null;
	vae_cpu_offload?: boolean | null;
	image_encoder_cpu_offload?: boolean | null;
	use_fsdp_inference?: boolean | null;
	enable_torch_compile?: boolean | null;
	vsa_sparsity?: number;
	tp_size?: number;
	sp_size?: number;
	// DMD distillation extras (sent by CreateJobModal)
	dmd_use_vsa?: boolean;
	dmd_vsa_sparsity?: number;
	dmd_denoising_steps?: string;
	real_score_guidance_scale?: number;
	generator_update_interval?: number;
	real_score_model_path?: string;
	fake_score_model_path?: string;
}

export interface Model {
	id: string;
	label: string;
}

export interface GpuInfo {
	index: number;
	name: string;
	utilization: number;
	memory_used_mib: number;
	memory_total_mib: number;
	temperature_c: number | null;
	power_watts: number | null;
	power_limit_watts: number | null;
}

export interface GpuSnapshot {
	available: boolean;
	gpus: GpuInfo[];
	error: string | null;
}

/**
 * Server-persisted settings: everything in DefaultOptions except
 * apiServerBaseUrl, which is purely local (per-browser).
 */
export type Settings = Omit<DefaultOptions, "apiServerBaseUrl">;

// MARK: - API Functions

export async function getSettings(): Promise<Settings> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/settings`);
	if (!response.ok) {
		throw new Error("Failed to fetch settings");
	}
	return response.json();
}

export async function updateSettings(
	updates: Partial<Settings>,
): Promise<Settings> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/settings`, {
		method: "PUT",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(updates),
	});
	if (!response.ok) {
		throw new Error("Failed to update settings");
	}
	return response.json();
}

export type MediaType = "image" | "video" | "audio";

/**
 * Upload an image, video or audio file for a MiniMax-H3 Ref2VA reference.
 * The server derives media_type from the extension and returns it, so callers
 * do not have to duplicate that mapping.
 */
/** Create a new pending job with the same configuration as an existing one. */
export async function duplicateJob(jobId: string): Promise<{ id: string }> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${jobId}/duplicate`, {
		method: "POST",
	});
	if (!response.ok) {
		const err = await response
			.json()
			.catch(() => ({ detail: "Duplicate failed" }));
		throw new Error(err.detail || "Duplicate failed");
	}
	return response.json();
}

/** Edit a pending job's configuration. Started jobs are rejected by the API. */
export async function updateJob(
	jobId: string,
	updates: Record<string, unknown>,
): Promise<unknown> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${jobId}`, {
		method: "PATCH",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(updates),
	});
	if (!response.ok) {
		const err = await response.json().catch(() => ({ detail: "Update failed" }));
		throw new Error(err.detail || "Update failed");
	}
	return response.json();
}

/**
 * Ask the server to save the last frame of a finished clip and return its path,
 * ready to attach to another job as an image reference.
 */
export async function extractLastFrame(jobId: string): Promise<{ path: string; media_type: MediaType }> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${jobId}/last-frame`, { method: "POST" });
	if (!response.ok) {
		const err = await response.json().catch(() => ({ detail: "Could not get the last frame" }));
		throw new Error(err.detail || "Could not get the last frame");
	}
	return response.json();
}

export interface EditVideoParams {
	startSeconds: number;
	/** omitted (or undefined) keeps to the end of the video */
	endSeconds?: number;
	/** added directly to 0-255 pixel values; 0 = unchanged */
	brightness?: number;
	/** 1 = unchanged, 0 = flat gray, >1 = more contrast */
	contrast?: number;
	/** 1 = unchanged, 0 = grayscale, >1 = more saturated */
	saturation?: number;
}

/**
 * Cut a completed job's video to a range and/or adjust its color, in place.
 * The untouched original is kept server-side, so this always renders fresh
 * from it with the *given* parameters -- pass every value you want kept, not
 * just the one you changed, or the others reset to their neutral defaults.
 * Can be undone with restoreJobVideo.
 */
export async function trimJob(jobId: string, params: EditVideoParams): Promise<Job> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${jobId}/trim`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			start_seconds: params.startSeconds,
			end_seconds: params.endSeconds ?? null,
			brightness: params.brightness ?? 0,
			contrast: params.contrast ?? 1,
			saturation: params.saturation ?? 1,
		}),
	});
	if (!response.ok) {
		const err = await response.json().catch(() => ({ detail: "Could not edit the clip" }));
		throw new Error(err.detail || "Could not edit the clip");
	}
	return response.json();
}

/** Undo every trim and color adjustment on a job's video, back to what it originally generated. */
export async function restoreJobVideo(jobId: string): Promise<Job> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${jobId}/restore-video`, { method: "POST" });
	if (!response.ok) {
		const err = await response.json().catch(() => ({ detail: "Could not restore the original video" }));
		throw new Error(err.detail || "Could not restore the original video");
	}
	return response.json();
}

/**
 * Ask the server to save the last second or so of a finished clip and return its
 * path, ready to attach to another job as a video reference -- so the next
 * clip continues with real motion instead of guessing it from a single frame.
 */
export async function extractLastClip(jobId: string): Promise<{ path: string; media_type: MediaType }> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${jobId}/last-clip`, { method: "POST" });
	if (!response.ok) {
		const err = await response.json().catch(() => ({ detail: "Could not get the end of the clip" }));
		throw new Error(err.detail || "Could not get the end of the clip");
	}
	return response.json();
}

export async function uploadMedia(
	file: File,
): Promise<{ path: string; media_type: MediaType }> {
	const baseApiUrl = getApiBaseUrl();
	const formData = new FormData();
	formData.append("file", file);
	const response = await fetch(`${baseApiUrl}/upload-media`, {
		method: "POST",
		body: formData,
	});
	if (!response.ok) {
		const err = await response
			.json()
			.catch(() => ({ detail: "Upload failed" }));
		throw new Error(err.detail || "Upload failed");
	}
	return response.json();
}

export async function uploadImage(file: File): Promise<{ path: string }> {
	const baseApiUrl = getApiBaseUrl();
	const formData = new FormData();
	formData.append("file", file);
	const response = await fetch(`${baseApiUrl}/upload-image`, {
		method: "POST",
		body: formData,
	});
	if (!response.ok) {
		const err = await response
			.json()
			.catch(() => ({ detail: "Upload failed" }));
		throw new Error(err.detail || "Upload failed");
	}
	return response.json();
}

export async function uploadRawDataset(
	files: File[],
): Promise<{ path: string; upload_id: string; file_names: string[] }> {
	const baseApiUrl = getApiBaseUrl();
	const formData = new FormData();
	for (const f of files) {
		formData.append("files", f);
	}
	const response = await fetch(`${baseApiUrl}/upload-raw-dataset`, {
		method: "POST",
		body: formData,
	});
	if (!response.ok) {
		const err = await response
			.json()
			.catch(() => ({ detail: "Upload failed" }));
		throw new Error(err.detail || "Upload failed");
	}
	return response.json();
}

export async function getModels(workloadType?: string): Promise<Model[]> {
	const baseApiUrl = getApiBaseUrl();
	const url = workloadType
		? `${baseApiUrl}/models?workload_type=${encodeURIComponent(workloadType)}`
		: `${baseApiUrl}/models`;
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error("Failed to fetch models");
	}
	return response.json();
}

export async function getGpus(): Promise<GpuSnapshot> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/gpus`);
	if (!response.ok) {
		throw new Error("Failed to fetch GPU status");
	}
	return response.json();
}

export async function getJobsList(jobType?: JobType): Promise<Job[]> {
	const baseApiUrl = getApiBaseUrl();
	const url = jobType
		? `${baseApiUrl}/jobs?job_type=${encodeURIComponent(jobType)}`
		: `${baseApiUrl}/jobs`;
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error("Failed to fetch jobs");
	}
	return response.json();
}

export async function createJob(job: CreateJobRequest): Promise<Job> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(job),
	});
	if (!response.ok) {
		throw new Error("Failed to create job");
	}
	return response.json();
}

export async function startJob(id: string): Promise<Job> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${id}/start`, {
		method: "POST",
	});
	if (!response.ok) {
		const error = await response
			.json()
			.catch(() => ({ detail: "Failed to start job" }));
		throw new Error(error.detail || "Failed to start job");
	}
	return response.json();
}

/**
 * Queue jobs, in the order given, to run one after another as the server has room.
 * A job that starts from another job's last frame waits for that job to finish.
 * All-or-nothing: if any job can't be queued, none are.
 */
export async function queueJobs(ids: string[]): Promise<Job[]> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/queue`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ job_ids: ids }),
	});
	if (!response.ok) {
		const error = await response
			.json()
			.catch(() => ({ detail: "Failed to queue jobs" }));
		throw new Error(error.detail || "Failed to queue jobs");
	}
	return response.json();
}

/** Take a queued job out of the queue; it goes back to pending. */
export async function dequeueJob(id: string): Promise<Job> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${id}/dequeue`, {
		method: "POST",
	});
	if (!response.ok) {
		const error = await response
			.json()
			.catch(() => ({ detail: "Failed to remove job from the queue" }));
		throw new Error(error.detail || "Failed to remove job from the queue");
	}
	return response.json();
}

export interface MergedScene {
	/** file name on the server; pass to getMergedVideoUrl / downloadMergedVideo */
	filename: string;
	clips: number;
	/** length of the merged video */
	seconds: number;
}

/**
 * Join the finished clips of a scene, in the order given, into one video. Fails,
 * making nothing, if any clip hasn't finished.
 */
export async function mergeScene(jobIds: string[], name: string): Promise<MergedScene> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/scenes/merge`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ job_ids: jobIds, name }),
	});
	if (!response.ok) {
		const error = await response.json().catch(() => ({ detail: "Failed to merge the scene" }));
		throw new Error(error.detail || "Failed to merge the scene");
	}
	return response.json();
}

export function getMergedVideoUrl(filename: string): string {
	return `${getApiBaseUrl()}/merged/${encodeURIComponent(filename)}`;
}

export async function downloadMergedVideo(filename: string): Promise<Blob> {
	const response = await fetch(getMergedVideoUrl(filename));
	if (!response.ok) {
		throw new Error("Failed to download the merged video");
	}
	return response.blob();
}

export async function stopJob(id: string): Promise<Job> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${id}/stop`, {
		method: "POST",
	});
	if (!response.ok) {
		const error = await response
			.json()
			.catch(() => ({ detail: "Failed to stop job" }));
		throw new Error(error.detail || "Failed to stop job");
	}
	return response.json();
}

export async function deleteJob(id: string): Promise<void> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${id}`, {
		method: "DELETE",
	});
	if (!response.ok) {
		const error = await response
			.json()
			.catch(() => ({ detail: "Failed to delete job" }));
		throw new Error(error.detail || "Failed to delete job");
	}
}

export interface JobLogs {
	lines: string[];
	total: number;
	progress: number;
	progress_msg: string;
	phase: string;
}

export async function getJobLogs(
	id: string,
	after: number = 0,
): Promise<JobLogs> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(
		`${baseApiUrl}/jobs/${id}/logs?after=${after}`,
	);
	if (!response.ok) {
		throw new Error("Failed to fetch job logs");
	}
	return response.json();
}

export async function downloadJobLog(id: string): Promise<Blob> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${id}/download_log`);
	if (!response.ok) {
		throw new Error("Failed to download job log");
	}
	return response.blob();
}

export async function downloadJobVideo(id: string): Promise<Blob> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/jobs/${id}/video`);
	if (!response.ok) {
		throw new Error("Failed to download video");
	}
	return response.blob();
}

// --- Datasets ---

export interface Dataset {
	id: string;
	name: string;
	created_at: number;
	file_count?: number;
	size_bytes?: number;
}

export interface CreateDatasetRequest {
	name: string;
	upload_path: string;
	file_names: string[];
	/** Optional: file_name -> caption from videos2caption.json */
	captions?: Record<string, string>;
}

export interface DatasetFilesResponse {
	file_names: string[];
	captions: Record<string, string>;
}

export async function getDatasets(): Promise<Dataset[]> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/datasets`);
	if (!response.ok) {
		throw new Error("Failed to fetch datasets");
	}
	return response.json();
}

export async function createDataset(
	data: CreateDatasetRequest,
): Promise<Dataset> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/datasets`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(data),
	});
	if (!response.ok) {
		const err = await response
			.json()
			.catch(() => ({ detail: "Failed to create dataset" }));
		throw new Error(err.detail || "Failed to create dataset");
	}
	return response.json();
}

export async function deleteDataset(id: string): Promise<void> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/datasets/${id}`, {
		method: "DELETE",
	});
	if (!response.ok) {
		const err = await response
			.json()
			.catch(() => ({ detail: "Failed to delete dataset" }));
		throw new Error(err.detail || "Failed to delete dataset");
	}
}

export async function getDatasetFiles(
	id: string,
): Promise<DatasetFilesResponse> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/datasets/${id}/files`);
	if (!response.ok) {
		throw new Error("Failed to fetch dataset files");
	}
	return response.json();
}

export async function updateDatasetCaption(
	id: string,
	file_name: string,
	caption: string,
): Promise<void> {
	const baseApiUrl = getApiBaseUrl();
	const response = await fetch(`${baseApiUrl}/datasets/${id}/captions`, {
		method: "PUT",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ file_name, caption }),
	});
	if (!response.ok) {
		throw new Error("Failed to update caption");
	}
}

export function getDatasetMediaUrl(
	datasetId: string,
	fileName: string,
): string {
	const baseApiUrl = getApiBaseUrl();
	return `${baseApiUrl}/datasets/${datasetId}/media/${encodeURIComponent(fileName)}`;
}
