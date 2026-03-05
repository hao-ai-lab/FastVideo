// SPDX-License-Identifier: Apache-2.0

import { Job, type JobType } from "./types";

function getApiBaseUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  if (!apiUrl) {
    throw new Error(
      "Please set NEXT_PUBLIC_API_BASE_URL in your .env.local file or as an environment variable. " +
      "Example: NEXT_PUBLIC_API_BASE_URL=http://localhost:8189/api"
    );
  }

  return apiUrl;
}

export interface CreateJobRequest {
    model_id: string;
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
}

export interface Model {
    id: string;
    label: string;
    type: string;
}

export interface Settings {
    defaultModelId: string;
    defaultModelIdT2v: string;
    defaultModelIdI2v: string;
    defaultModelIdT2i: string;
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
    textEncoderCpuOffload: boolean;
    vaeCpuOffload: boolean;
    imageEncoderCpuOffload: boolean;
    useFsdpInference: boolean;
    enableTorchCompile: boolean;
    vsaSparsity: number;
    tpSize: number;
    spSize: number;
    autoStartJob: boolean;
    datasetUploadPath: string;
}

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
    updates: Partial<Settings>
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

export async function uploadImage(file: File): Promise<{ path: string }> {
    const baseApiUrl = getApiBaseUrl();
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${baseApiUrl}/upload-image`, {
        method: "POST",
        body: formData,
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(err.detail || "Upload failed");
    }
    return response.json();
}

export async function uploadRawDataset(
    files: File[]
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
        const err = await response.json().catch(() => ({ detail: "Upload failed" }));
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

export async function getJobDetails(id: string): Promise<Job> {
    const baseApiUrl = getApiBaseUrl();
    const response = await fetch(`${baseApiUrl}/jobs/${id}`);
    if (!response.ok) {
        throw new Error("Failed to fetch job");
    }
    return response.json();
}

export async function startJob(id: string): Promise<Job> {
    const baseApiUrl = getApiBaseUrl();
    const response = await fetch(`${baseApiUrl}/jobs/${id}/start`, {
        method: "POST",
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Failed to start job" }));
        throw new Error(error.detail || "Failed to start job");
    }
    return response.json();
}

export async function stopJob(id: string): Promise<Job> {
    const baseApiUrl = getApiBaseUrl();
    const response = await fetch(`${baseApiUrl}/jobs/${id}/stop`, {
        method: "POST",
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Failed to stop job" }));
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
        const error = await response.json().catch(() => ({ detail: "Failed to delete job" }));
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

export async function getJobLogs(id: string, after: number = 0): Promise<JobLogs> {
    const baseApiUrl = getApiBaseUrl();
    const response = await fetch(`${baseApiUrl}/jobs/${id}/logs?after=${after}`);
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

export async function getDataset(id: string): Promise<Dataset> {
    const baseApiUrl = getApiBaseUrl();
    const response = await fetch(`${baseApiUrl}/datasets/${id}`);
    if (!response.ok) {
        throw new Error("Failed to fetch dataset");
    }
    return response.json();
}

export async function createDataset(
    data: CreateDatasetRequest
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
        const err = await response.json().catch(() => ({ detail: "Failed to create dataset" }));
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
        const err = await response.json().catch(() => ({ detail: "Failed to delete dataset" }));
        throw new Error(err.detail || "Failed to delete dataset");
    }
}

export async function getDatasetFiles(id: string): Promise<DatasetFilesResponse> {
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
    caption: string
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

export function getDatasetMediaUrl(datasetId: string, fileName: string): string {
    const baseApiUrl = getApiBaseUrl();
    return `${baseApiUrl}/datasets/${datasetId}/media/${encodeURIComponent(fileName)}`;
}

