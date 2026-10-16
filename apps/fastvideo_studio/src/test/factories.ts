import type { Model } from '@/lib/api';
import type { ModelRecipe } from '@/lib/modelCapabilities';
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

/** A FastH3-Omni PDD-8 recipe, as GET /api/models reports it. */
export const OMNI_RECIPE: ModelRecipe = {
  kind: 'fasth3_omni_pdd',
  source: 'fastvideo_inference.json',
  export: '/models/fasth3_omni_export',
  export_revision: null,
  base: 'MiniMaxAI/MiniMax-H3',
  base_revision: '9bfb6693f2cf6de171db46d1aa586f67d773a1da',
  contract_sha256: 'abcdef0123456789',
  contract: {
    pdd_steps: 32,
    pdd_step_indices: [0, 4, 8, 12, 16, 20, 24, 28, 32],
    num_inference_steps: 8,
    transformer_forwards: 8,
    guidance_scale: 1.0,
    attention_backend: 'VIDEO_SPARSE_ATTN_H3',
    vsa_sparsity: 0.9,
    vsa_tile_size: 128,
    vsa_ref_keep_rate: 0.1,
    video_scheduler_shift: 12,
    audio_scheduler_shift: 3,
    base_model_revision: 'hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da',
  },
  locked_fields: [
    'workload_type', 'image_path', 'last_image_path', 'num_inference_steps', 'guidance_scale',
    'guidance_rescale', 'negative_prompt', 'vsa_sparsity', 'fps', 'enable_torch_compile',
    'use_fsdp_inference', 'tp_size', 'sp_size', 'dit_cpu_offload', 'dit_layerwise_offload',
    'image_encoder_cpu_offload',
  ],
};

/** A FastH3-Omni export configured on the server, on hardware that cannot run tile 128. */
export function makeOmniModel(overrides: Partial<Model> = {}): Model {
  return {
    id: '/models/fasth3_omni_export',
    label: 'FastH3-Omni · fasth3_omni_export',
    workload_types: ['i2v'],
    config_source: 'server',
    capabilities: {
      references: {
        media_types: ['image', 'video', 'audio'],
        limits: { image: 9, video: 3, audio: 3 },
        max_total: 12,
        requires_visual: true,
        required: true,
      },
      start_frame: false,
      end_frame: false,
      max_frames: 361,
      num_gpus: [1, 2, 4, 7, 8],
      fixed_fps: 24,
    },
    recipe: OMNI_RECIPE,
    recipe_defaults: {
      height: 480,
      width: 832,
      num_frames: 124,
      num_gpus: 1,
      text_encoder_cpu_offload: true,
      vae_cpu_offload: true,
    },
    hardware: { supported: false, reason: 'found NVIDIA RTX 6000 Ada Generation (sm_89).' },
    error: null,
    ...overrides,
  };
}
