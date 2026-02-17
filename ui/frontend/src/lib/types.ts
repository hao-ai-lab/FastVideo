// SPDX-License-Identifier: Apache-2.0

export interface Job {
    id: string;
    model_id: string;
    prompt: string;
    status: string;
    created_at: number;
    started_at: number | null;
    finished_at: number | null;
    error: string | null;
    output_path: string | null;
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
