// SPDX-License-Identifier: Apache-2.0

import { Job } from "./types";

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
    num_inference_steps?: number;
    num_frames?: number;
    height?: number;
    width?: number;
    guidance_scale?: number;
    seed?: number;
    num_gpus?: number;
    dit_cpu_offload?: boolean | null;
    text_encoder_cpu_offload?: boolean | null;
    use_fsdp_inference?: boolean | null;
}

export interface Model {
    id: string;
    label: string;
    type: string;
}

// MARK: - API Functions

export async function getModels(): Promise<Model[]> {
    const baseApiUrl = getApiBaseUrl();
    const response = await fetch(`${baseApiUrl}/models`);
    if (!response.ok) {
        throw new Error("Failed to fetch models");
    }
    return response.json();
}

export async function getJobsList(): Promise<Job[]> {
    const baseApiUrl = getApiBaseUrl();
    const response = await fetch(`${baseApiUrl}/jobs`);
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
