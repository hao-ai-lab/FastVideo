// SPDX-License-Identifier: Apache-2.0

import { Job } from "./types";

const API_BASE_URL = "http://localhost:8189/api";

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
}

// MARK: - API Functions

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
