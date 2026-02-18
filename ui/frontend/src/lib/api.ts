// SPDX-License-Identifier: Apache-2.0

import { Job } from "./types";

const API_BASE_URL = 
  (typeof process !== "undefined" && process.env.API_BASE_URL) ||
  "http://localhost:8189/api";

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
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) {
    throw new Error("Failed to fetch models");
  }
  return response.json();
}

export async function getJobsList(): Promise<Job[]> {
  const response = await fetch(`${API_BASE_URL}/jobs`);
  if (!response.ok) {
    throw new Error("Failed to fetch jobs");
  }
  return response.json();
}

export async function createJob(job: CreateJobRequest): Promise<Job> {
    const response = await fetch(`${API_BASE_URL}/jobs`, {
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
    const response = await fetch(`${API_BASE_URL}/jobs/${id}`);
    if (!response.ok) {
        throw new Error("Failed to fetch job");
    }
    return response.json();
}

export async function startJob(id: string): Promise<Job> {
    const response = await fetch(`${API_BASE_URL}/jobs/${id}/start`, {
        method: "POST",
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Failed to start job" }));
        throw new Error(error.detail || "Failed to start job");
    }
    return response.json();
}

export async function stopJob(id: string): Promise<Job> {
    const response = await fetch(`${API_BASE_URL}/jobs/${id}/stop`, {
        method: "POST",
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Failed to stop job" }));
        throw new Error(error.detail || "Failed to stop job");
    }
    return response.json();
}

export async function deleteJob(id: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/jobs/${id}`, {
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
    const response = await fetch(`${API_BASE_URL}/jobs/${id}/logs?after=${after}`);
    if (!response.ok) {
        throw new Error("Failed to fetch job logs");
    }
    return response.json();
}

export async function downloadJobLog(id: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/jobs/${id}/download_log`);
    if (!response.ok) {
        throw new Error("Failed to download job log");
    }
    return response.blob();
}
