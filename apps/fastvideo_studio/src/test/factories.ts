import type { Job } from '@/lib/types';

/**
 * Base Job with every required field filled with neutral values. Tests
 * layer their scenario-specific fields on top (usually via a small local
 * wrapper), so adding a required field to Job only touches this factory.
 */
export function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 'job-1',
    model_id: 'model-x',
    prompt: 'a prompt',
    job_type: 'inference',
    workload_type: 't2v',
    status: 'pending',
    created_at: 0,
    started_at: null,
    finished_at: null,
    error: null,
    output_path: null,
    log_file_path: null,
    num_inference_steps: 0,
    num_frames: 0,
    height: 0,
    width: 0,
    guidance_scale: 0,
    seed: 0,
    num_gpus: 0,
    progress: 0,
    progress_msg: '',
    phase: '',
    ...overrides,
  };
}
