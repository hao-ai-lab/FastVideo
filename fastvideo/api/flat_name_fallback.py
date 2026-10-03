# SPDX-License-Identifier: Apache-2.0
"""Flat ``FastVideoArgs`` and ``TrainingArgs`` attribute names on a ``ResolvedGeneratorConfig``.

Runtime code that still reads ``fastvideo_args.<flat name>`` receives a ``ResolvedGeneratorConfig``. An attribute
that is not a field of the typed tree reaches :func:`read_flat_name`, which looks the name up in ``FLAT_TO_TYPED``
and returns the value that ``FastVideoArgs`` or ``TrainingArgs`` held for the same input:

- ``typed``: the value at the typed path. A ``None`` value stands for ``none_value``, the ``FastVideoArgs`` default
  (``cli_none_value`` on the training and preprocessing roots, whose runs used the argparse defaults). ``convert``
  names the flat format, such as an enum or a comma-separated string.
- ``pipeline_config``: the materialized ``PipelineConfig``.
- ``property``: a read-only property of the resolved config.
- ``derived``: a pure function of a typed value.
- ``runtime_state``: mutable state behind ``RUNTIME_STATE_NAMES``, shared by every config that ``with_override``
  derives from the same resolution.
- ``removed``: a name without a typed field; ``fallback`` says which input still supplies the flat value.

``FLAT_TO_TYPED`` and ``PIPELINE_CONFIG_TYPED_HOMES`` are generated from ``FastVideoArgs``, ``TrainingArgs``, and the
typed schema, together with ``flat_to_typed.json``. This module goes away with the flat names.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from fastvideo.api.resolution import _lookup, _Struct, _to_config, _to_plain

# Names of runtime state that code assigns or mutates in place after resolution.
RUNTIME_STATE_NAMES = frozenset({"model_paths", "model_loaded", "ray_placement_group", "ray_runtime_env"})
# Root config classes whose runs took their values from the FastVideoArgs and TrainingArgs command line.
_COMMAND_LINE_ROOTS = frozenset({"TrainingRunConfig", "PreprocessRunConfig"})
# Generic refine keys under pipeline.experimental or pipeline.preset_overrides, and the typed path each one sets.
GENERIC_REFINE_PATHS = {
    "refine_enabled": "pipeline.ltx2.refine.enabled",
    "refine_upsampler_path": "pipeline.components.upsampler_weights",
    "refine_transformer_path": "pipeline.ltx2.refine.transformer_path",
    "refine_lora_path": "pipeline.ltx2.refine.lora_path",
    "refine_num_inference_steps": "pipeline.ltx2.refine.num_inference_steps",
    "refine_guidance_scale": "pipeline.ltx2.refine.guidance_scale",
    "refine_add_noise": "pipeline.ltx2.refine.add_noise",
    "refine_noise_path": "pipeline.ltx2.refine.noise_path",
    "refine_audio_noise_path": "pipeline.ltx2.refine.audio_noise_path",
}

# BEGIN GENERATED FLAT_TO_TYPED (gen_flat_to_typed.py)
FLAT_TO_TYPED: dict[str, dict[str, Any]] = {
    'VSA_cache_tile_buf': {
        'kind': 'typed',
        'path': 'training.vsa.cache_tile_buf',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'VSA_decay_interval_steps': {
        'kind': 'typed',
        'path': 'training.vsa.decay_interval_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'VSA_decay_rate': {
        'kind': 'typed',
        'path': 'training.vsa.decay_rate',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'VSA_sparsity': {
        'kind': 'typed',
        'none_value': 0.0,
        'path': 'engine.attention.vsa_sparsity',
        'source': 'FastVideoArgs'
    },
    'VSA_tile_size': {
        'kind': 'typed',
        'none_value': 256,
        'path': 'engine.attention.vsa_tile_size',
        'source': 'FastVideoArgs'
    },
    'attention_backend': {
        'kind': 'typed',
        'path': 'engine.attention.backend',
        'source': 'FastVideoArgs'
    },
    'betas': {
        'convert': 'comma_string',
        'kind': 'typed',
        'path': 'training.optimizer.betas',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'boundary_ratio': {
        'kind': 'typed',
        'none_value': 0.875,
        'path': 'pipeline.boundary_ratio',
        'source': 'FastVideoArgs'
    },
    'checkpoints_total_limit': {
        'kind': 'typed',
        'path': 'training.checkpoint.checkpoints_total_limit',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'context_noise': {
        'kind': 'typed',
        'path': 'training.self_forcing.context_noise',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'data_path': {
        'kind': 'typed',
        'path': 'training.data.data_path',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'dataloader_num_workers': {
        'kind': 'typed',
        'path': 'training.data.dataloader_num_workers',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'dfake_gen_update_ratio': {
        'kind': 'typed',
        'path': 'training.self_forcing.dfake_gen_update_ratio',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'disable_autocast': {
        'kind': 'typed',
        'none_value': False,
        'path': 'engine.disable_autocast',
        'source': 'FastVideoArgs'
    },
    'dist_timeout': {
        'kind': 'typed',
        'path': 'engine.parallelism.dist_timeout',
        'source': 'FastVideoArgs'
    },
    'distill_cfg': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'distributed_executor_backend': {
        'kind': 'typed',
        'none_value': 'mp',
        'path': 'engine.execution_backend',
        'source': 'FastVideoArgs'
    },
    'dit_cpu_offload': {
        'kind': 'typed',
        'none_value': True,
        'path': 'engine.offload.dit',
        'source': 'FastVideoArgs'
    },
    'dit_layerwise_offload': {
        'kind': 'typed',
        'none_value': True,
        'path': 'engine.offload.dit_layerwise',
        'source': 'FastVideoArgs'
    },
    'ema_decay': {
        'kind': 'typed',
        'path': 'training.ema.decay',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'ema_start_step': {
        'kind': 'typed',
        'path': 'training.ema.start_step',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'enable_gradient_checkpointing_type': {
        'kind': 'typed',
        'path': 'training.model.enable_gradient_checkpointing_type',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'enable_gradient_masking': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'enable_stage_verification': {
        'kind': 'typed',
        'none_value': True,
        'path': 'engine.enable_stage_verification',
        'source': 'FastVideoArgs'
    },
    'enable_torch_compile': {
        'kind': 'typed',
        'none_value': False,
        'path': 'engine.compile.enabled',
        'source': 'FastVideoArgs'
    },
    'enable_torch_compile_audio_vae': {
        'kind': 'typed',
        'none_value': False,
        'path': 'engine.compile.audio_vae_enabled',
        'source': 'FastVideoArgs'
    },
    'enable_torch_compile_text_encoder': {
        'kind': 'typed',
        'none_value': False,
        'path': 'engine.compile.text_encoder_enabled',
        'source': 'FastVideoArgs'
    },
    'enable_torch_compile_vae': {
        'kind': 'typed',
        'none_value': False,
        'path': 'engine.compile.vae_enabled',
        'source': 'FastVideoArgs'
    },
    'fake_score_betas': {
        'convert': 'comma_string',
        'kind': 'typed',
        'path': 'training.distillation.fake_score_betas',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'fake_score_learning_rate': {
        'kind': 'typed',
        'path': 'training.distillation.fake_score_learning_rate',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'fake_score_lr_scheduler': {
        'kind': 'typed',
        'path': 'training.distillation.fake_score_lr_scheduler',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'fake_score_model_path': {
        'kind': 'typed',
        'path': 'training.distillation.fake_score_model_path',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'fsdp_sharding_startegy': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'generator_update_interval': {
        'kind': 'typed',
        'path': 'training.distillation.generator_update_interval',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'gradient_accumulation_steps': {
        'kind': 'typed',
        'path': 'training.loop.gradient_accumulation_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'gradient_mask_last_n_frames': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'group_frame': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'group_resolution': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'h3_sequential_load': {
        'kind': 'typed',
        'path': 'pipeline.minimax_h3.sequential_load',
        'source': 'FastVideoArgs'
    },
    'hsdp_replicate_dim': {
        'kind': 'typed',
        'none_value': 1,
        'path': 'engine.parallelism.hsdp_replicate_dim',
        'source': 'FastVideoArgs'
    },
    'hsdp_shard_dim': {
        'kind': 'typed',
        'none_value': -1,
        'path': 'engine.parallelism.hsdp_shard_dim',
        'source': 'FastVideoArgs'
    },
    'hunyuan_teacher_disable_cfg': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'image_encoder_cpu_offload': {
        'kind': 'typed',
        'none_value': True,
        'path': 'engine.offload.image_encoder',
        'source': 'FastVideoArgs'
    },
    'independent_first_frame': {
        'kind': 'typed',
        'path': 'training.self_forcing.independent_first_frame',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'inference_mode': {
        'kind': 'property',
        'note': 'not (mode in (FINETUNING, DISTILLATION))',
        'path': 'inference_mode',
        'source': 'FastVideoArgs'
    },
    'inference_torch_compile': {
        'kind': 'typed',
        'none_value': False,
        'path': 'engine.compile.regional',
        'source': 'FastVideoArgs'
    },
    'init_weights_from_safetensors': {
        'cli_none_value': None,
        'kind': 'typed',
        'none_value': '',
        'path': 'pipeline.components.transformer_weights',
        'source': 'FastVideoArgs'
    },
    'init_weights_from_safetensors_2': {
        'cli_none_value': None,
        'kind': 'typed',
        'none_value': '',
        'path': 'pipeline.components.transformer_2_weights',
        'source': 'FastVideoArgs'
    },
    'last_step_only': {
        'kind': 'typed',
        'path': 'training.self_forcing.last_step_only',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lazy_module_load': {
        'kind': 'typed',
        'path': 'engine.offload.lazy_module_load',
        'source': 'FastVideoArgs'
    },
    'learning_rate': {
        'kind': 'typed',
        'path': 'training.optimizer.learning_rate',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'linear_quadratic_threshold': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'linear_range': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'log_validation': {
        'kind': 'typed',
        'path': 'training.validation.enabled',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'log_visualization': {
        'kind': 'typed',
        'path': 'training.validation.log_visualization',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'logit_mean': {
        'kind': 'typed',
        'path': 'training.model.logit_mean',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'logit_std': {
        'kind': 'typed',
        'path': 'training.model.logit_std',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lora_alpha': {
        'kind': 'typed',
        'path': 'training.lora.alpha',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lora_nickname': {
        'kind': 'typed',
        'none_value': 'default',
        'path': 'pipeline.components.lora_nickname',
        'source': 'FastVideoArgs'
    },
    'lora_path': {
        'kind': 'typed',
        'path': 'pipeline.components.lora_path',
        'source': 'FastVideoArgs'
    },
    'lora_rank': {
        'kind': 'typed',
        'path': 'training.lora.rank',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lora_strength': {
        'kind': 'typed',
        'none_value': 1.0,
        'path': 'pipeline.components.lora_strength',
        'source': 'FastVideoArgs'
    },
    'lora_target_modules': {
        'kind': 'typed',
        'path': 'pipeline.components.lora_target_modules',
        'source': 'FastVideoArgs'
    },
    'lora_training': {
        'kind': 'typed',
        'path': 'training.lora.enabled',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lr_num_cycles': {
        'kind': 'typed',
        'path': 'training.optimizer.lr_num_cycles',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lr_power': {
        'kind': 'typed',
        'path': 'training.optimizer.lr_power',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lr_scheduler': {
        'kind': 'typed',
        'path': 'training.optimizer.lr_scheduler',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'lr_warmup_steps': {
        'kind': 'typed',
        'path': 'training.optimizer.lr_warmup_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'ltx2_audio_latent_path': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.audio_latent_path',
        'source': 'FastVideoArgs'
    },
    'ltx2_first_frame_conditioning_p': {
        'kind': 'typed',
        'path': 'training.model.ltx2_first_frame_conditioning_p',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'ltx2_initial_latent_path': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.initial_latent_path',
        'source': 'FastVideoArgs'
    },
    'ltx2_legacy_native_noise_order': {
        'kind': 'typed',
        'none_value': False,
        'path': 'pipeline.ltx2.legacy_native_noise_order',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_add_noise': {
        'kind': 'typed',
        'none_value': True,
        'path': 'pipeline.ltx2.refine.add_noise',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_audio_noise_path': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.refine.audio_noise_path',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_enabled': {
        'kind': 'typed',
        'none_value': False,
        'path': 'pipeline.ltx2.refine.enabled',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_guidance_scale': {
        'kind': 'typed',
        'none_value': 1.0,
        'path': 'pipeline.ltx2.refine.guidance_scale',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_lora_path': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.refine.lora_path',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_noise_path': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.refine.noise_path',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_num_inference_steps': {
        'kind': 'typed',
        'none_value': 3,
        'path': 'pipeline.ltx2.refine.num_inference_steps',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_transformer_path': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.refine.transformer_path',
        'source': 'FastVideoArgs'
    },
    'ltx2_refine_upsampler_path': {
        'kind': 'typed',
        'path': 'pipeline.components.upsampler_weights',
        'source': 'FastVideoArgs'
    },
    'ltx2_use_distilled_sigmas': {
        'kind': 'typed',
        'none_value': True,
        'path': 'pipeline.ltx2.use_distilled_sigmas',
        'source': 'FastVideoArgs'
    },
    'ltx2_vae_spatial_tile_overlap_in_pixels': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.vae_spatial_tile_overlap_in_pixels',
        'source': 'FastVideoArgs'
    },
    'ltx2_vae_spatial_tile_size_in_pixels': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.vae_spatial_tile_size_in_pixels',
        'source': 'FastVideoArgs'
    },
    'ltx2_vae_temporal_tile_overlap_in_frames': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.vae_temporal_tile_overlap_in_frames',
        'source': 'FastVideoArgs'
    },
    'ltx2_vae_temporal_tile_size_in_frames': {
        'kind': 'typed',
        'path': 'pipeline.ltx2.vae_temporal_tile_size_in_frames',
        'source': 'FastVideoArgs'
    },
    'ltx2_vae_tiling': {
        'kind': 'typed',
        'path': 'pipeline.vae_tiling',
        'source': 'FastVideoArgs'
    },
    'master_port': {
        'kind': 'typed',
        'path': 'engine.parallelism.master_port',
        'source': 'FastVideoArgs'
    },
    'master_weight_type': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'max_grad_norm': {
        'kind': 'typed',
        'path': 'training.optimizer.max_grad_norm',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'max_timestep_ratio': {
        'kind': 'typed',
        'path': 'training.distillation.max_timestep_ratio',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'max_train_steps': {
        'kind': 'typed',
        'path': 'training.loop.max_train_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'min_lr_ratio': {
        'kind': 'typed',
        'path': 'training.optimizer.min_lr_ratio',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'min_timestep_ratio': {
        'kind': 'typed',
        'path': 'training.distillation.min_timestep_ratio',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'mixed_precision': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'moba_config': {
        'kind': 'typed',
        'none_value': {},
        'path': 'engine.attention.moba_config',
        'source': 'FastVideoArgs'
    },
    'moba_config_path': {
        'kind': 'typed',
        'path': 'engine.attention.moba_config_path',
        'source': 'FastVideoArgs'
    },
    'mode': {
        'convert': 'execution_mode',
        'kind': 'typed',
        'path': 'mode',
        'source': 'FastVideoArgs'
    },
    'mode_scale': {
        'kind': 'typed',
        'path': 'training.model.mode_scale',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'model_loaded': {
        'default': {
            'transformer': True,
            'upsampler': True,
            'vae': True
        },
        'kind': 'runtime_state',
        'path': 'ComposedPipelineBase.model_loaded (pipeline object)',
        'source': 'FastVideoArgs'
    },
    'model_path': {
        'kind': 'typed',
        'path': 'model_path',
        'source': 'FastVideoArgs'
    },
    'model_paths': {
        'default': {},
        'kind': 'runtime_state',
        'path': 'ComposedPipelineBase.model_paths (pipeline object)',
        'source': 'FastVideoArgs'
    },
    'multi_phased_distill_schedule': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'not_apply_cfg_solver': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'num_euler_timesteps': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'num_frame_per_block': {
        'kind': 'typed',
        'path': 'training.self_forcing.num_frame_per_block',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'num_frames': {
        'kind': 'typed',
        'path': 'training.data.num_frames',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'num_gpus': {
        'kind': 'typed',
        'none_value': 1,
        'path': 'engine.num_gpus',
        'source': 'FastVideoArgs'
    },
    'num_height': {
        'kind': 'typed',
        'path': 'training.data.num_height',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'num_latent_t': {
        'kind': 'typed',
        'path': 'training.data.num_latent_t',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'num_train_epochs': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'num_width': {
        'kind': 'typed',
        'path': 'training.data.num_width',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'output_dir': {
        'kind': 'typed',
        'path': 'training.checkpoint.output_dir',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'output_type': {
        'kind': 'typed',
        'none_value': 'pil',
        'path': 'pipeline.output_type',
        'source': 'FastVideoArgs'
    },
    'override_pipeline_cls_name': {
        'kind': 'typed',
        'path': 'pipeline.components.override_pipeline_cls_name',
        'source': 'FastVideoArgs'
    },
    'override_text_encoder_quant': {
        'kind': 'typed',
        'path': 'engine.quantization.text_encoder_quant',
        'source': 'FastVideoArgs'
    },
    'override_text_encoder_safetensors': {
        'kind': 'typed',
        'path': 'pipeline.components.text_encoder_weights',
        'source': 'FastVideoArgs'
    },
    'override_transformer_cls_name': {
        'kind': 'typed',
        'path': 'pipeline.components.override_transformer_cls_name',
        'source': 'FastVideoArgs'
    },
    'pin_cpu_memory': {
        'kind': 'typed',
        'none_value': True,
        'path': 'engine.offload.pin_cpu_memory',
        'source': 'FastVideoArgs'
    },
    'pipeline_config': {
        'kind': 'pipeline_config',
        'note': 'resolved_config.pipeline_config, the frozen model definition. Read a '
        'typed-home attribute (see pipeline_config_typed_homes) from its typed path.',
        'path': 'pipeline_config',
        'source': 'FastVideoArgs'
    },
    'precondition_outputs': {
        'kind': 'typed',
        'path': 'training.model.precondition_outputs',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'pred_decay_type': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'pred_decay_weight': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'preprocess_config': {
        'kind': 'typed',
        'note': 'the preprocess section of PreprocessRunConfig; None on the other roots',
        'other_roots_value': None,
        'path': 'preprocess',
        'roots': ['PreprocessRunConfig'],
        'source': 'FastVideoArgs'
    },
    'pretrained_model_name_or_path': {
        'kind': 'typed',
        'path': 'model_path',
        'source': 'TrainingArgs'
    },
    'prompt_txt': {
        'fallback': 'experimental',
        'kind': 'removed',
        'note': 'requests carry inputs.prompt_path; the fallback reads '
        'pipeline.experimental.prompt_txt',
        'path': 'request inputs.prompt_path',
        'source': 'FastVideoArgs'
    },
    'ray_placement_group': {
        'default': None,
        'kind': 'runtime_state',
        'path': 'RayDistributedExecutor (returned by initialize_ray_cluster)',
        'source': 'FastVideoArgs'
    },
    'ray_runtime_env': {
        'default': None,
        'kind': 'runtime_state',
        'note': 'initialized from pipeline.experimental.ray_runtime_env',
        'path': 'RayDistributedExecutor constructor argument',
        'source': 'FastVideoArgs'
    },
    'real_score_guidance_scale': {
        'kind': 'typed',
        'path': 'training.distillation.real_score_guidance_scale',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'real_score_model_path': {
        'kind': 'typed',
        'path': 'training.distillation.real_score_model_path',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'refine_add_noise': {
        'fallback': 'generic_refine_alias',
        'kind': 'removed',
        'note': 'generic alias of the LTX-2 refine field; the fallback returns the alias value '
        'that pipeline.experimental / pipeline.preset_overrides wrote, else None',
        'path': 'pipeline.ltx2.refine.add_noise',
        'source': 'FastVideoArgs'
    },
    'refine_audio_noise_path': {
        'fallback':
        'generic_refine_alias',
        'kind':
        'removed',
        'note':
        'generic alias of the LTX-2 refine field; the fallback returns the '
        'alias value that pipeline.experimental / pipeline.preset_overrides '
        'wrote, else None',
        'path':
        'pipeline.ltx2.refine.audio_noise_path',
        'source':
        'FastVideoArgs'
    },
    'refine_enabled': {
        'fallback': 'generic_refine_alias',
        'kind': 'removed',
        'note': 'generic alias of the LTX-2 refine field; the fallback returns the alias value '
        'that pipeline.experimental / pipeline.preset_overrides wrote, else None',
        'path': 'pipeline.ltx2.refine.enabled',
        'source': 'FastVideoArgs'
    },
    'refine_guidance_scale': {
        'fallback':
        'generic_refine_alias',
        'kind':
        'removed',
        'note':
        'generic alias of the LTX-2 refine field; the fallback returns the alias '
        'value that pipeline.experimental / pipeline.preset_overrides wrote, else '
        'None',
        'path':
        'pipeline.ltx2.refine.guidance_scale',
        'source':
        'FastVideoArgs'
    },
    'refine_lora_path': {
        'fallback': 'generic_refine_alias',
        'kind': 'removed',
        'note': 'generic alias of the LTX-2 refine field; the fallback returns the alias value '
        'that pipeline.experimental / pipeline.preset_overrides wrote, else None',
        'path': 'pipeline.ltx2.refine.lora_path',
        'source': 'FastVideoArgs'
    },
    'refine_noise_path': {
        'fallback':
        'generic_refine_alias',
        'kind':
        'removed',
        'note':
        'generic alias of the LTX-2 refine field; the fallback returns the alias '
        'value that pipeline.experimental / pipeline.preset_overrides wrote, else '
        'None',
        'path':
        'pipeline.ltx2.refine.noise_path',
        'source':
        'FastVideoArgs'
    },
    'refine_num_inference_steps': {
        'fallback':
        'generic_refine_alias',
        'kind':
        'removed',
        'note':
        'generic alias of the LTX-2 refine field; the fallback returns the '
        'alias value that pipeline.experimental / pipeline.preset_overrides '
        'wrote, else None',
        'path':
        'pipeline.ltx2.refine.num_inference_steps',
        'source':
        'FastVideoArgs'
    },
    'refine_transformer_path': {
        'fallback':
        'generic_refine_alias',
        'kind':
        'removed',
        'note':
        'generic alias of the LTX-2 refine field; the fallback returns the '
        'alias value that pipeline.experimental / pipeline.preset_overrides '
        'wrote, else None',
        'path':
        'pipeline.ltx2.refine.transformer_path',
        'source':
        'FastVideoArgs'
    },
    'refine_upsampler_path': {
        'fallback':
        'generic_refine_alias',
        'kind':
        'removed',
        'note':
        'generic alias of the LTX-2 refine field; the fallback returns the alias '
        'value that pipeline.experimental / pipeline.preset_overrides wrote, else '
        'None',
        'path':
        'pipeline.components.upsampler_weights',
        'source':
        'FastVideoArgs'
    },
    'resume_from_checkpoint': {
        'kind': 'typed',
        'path': 'training.checkpoint.resume_from_checkpoint',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'revision': {
        'kind': 'typed',
        'path': 'revision',
        'source': 'FastVideoArgs'
    },
    'same_step_across_blocks': {
        'kind': 'typed',
        'path': 'training.self_forcing.same_step_across_blocks',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'scale_lr': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'scheduler_type': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'seed': {
        'kind': 'typed',
        'path': 'training.data.seed',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'selective_checkpointing': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'simulate_generator_forward': {
        'kind': 'typed',
        'path': 'training.distillation.simulate_generator_forward',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'sp_size': {
        'kind': 'typed',
        'none_value': -1,
        'path': 'engine.parallelism.sp_size',
        'source': 'FastVideoArgs'
    },
    'taeh3_checkpoint': {
        'kind': 'typed',
        'path': 'pipeline.minimax_h3.taeh3_checkpoint',
        'source': 'FastVideoArgs'
    },
    'taeh3_chunk_size': {
        'kind': 'typed',
        'none_value': 5,
        'path': 'pipeline.minimax_h3.taeh3_chunk_size',
        'source': 'FastVideoArgs'
    },
    'text_encoder_cpu_offload': {
        'kind': 'typed',
        'none_value': True,
        'path': 'engine.offload.text_encoder',
        'source': 'FastVideoArgs'
    },
    'torch_compile_kwargs': {
        'derive': 'torch_compile_kwargs',
        'kind': 'derived',
        'note': 'backend/fullgraph/mode/dynamic that are not None, then extras; '
        'fastvideo.api.compat._compile_config_to_torch_kwargs',
        'path': 'engine.compile',
        'source': 'FastVideoArgs'
    },
    'torch_compile_kwargs_audio_vae': {
        'kind': 'typed',
        'none_value': {},
        'path': 'engine.compile.audio_vae_kwargs',
        'source': 'FastVideoArgs'
    },
    'torch_compile_kwargs_dit': {
        'kind': 'typed',
        'none_value': {},
        'path': 'engine.compile.dit_kwargs',
        'source': 'FastVideoArgs'
    },
    'torch_compile_kwargs_text_encoder': {
        'kind': 'typed',
        'none_value': {},
        'path': 'engine.compile.text_encoder_kwargs',
        'source': 'FastVideoArgs'
    },
    'torch_compile_kwargs_vae': {
        'kind': 'typed',
        'none_value': {},
        'path': 'engine.compile.vae_kwargs',
        'source': 'FastVideoArgs'
    },
    'tp_size': {
        'kind': 'typed',
        'none_value': -1,
        'path': 'engine.parallelism.tp_size',
        'source': 'FastVideoArgs'
    },
    'tracker_project_name': {
        'kind': 'typed',
        'path': 'training.tracker.project_name',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'trackers': {
        'kind': 'typed',
        'path': 'training.tracker.trackers',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'train_batch_size': {
        'kind': 'typed',
        'path': 'training.data.train_batch_size',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'train_sp_batch_size': {
        'kind': 'typed',
        'path': 'training.data.train_sp_batch_size',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'training_cfg_rate': {
        'kind': 'typed',
        'path': 'training.data.training_cfg_rate',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'training_state_checkpointing_steps': {
        'kind': 'typed',
        'path': 'training.checkpoint.training_state_checkpointing_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'transformer_quant': {
        'derive': 'transformer_quant',
        'kind': 'derived',
        'note': 'the QuantizationConfig instance for the typed name; materialization pins it '
        'on pipeline_config.dit_config.quant_config',
        'path': 'engine.quantization.transformer_quant',
        'source': 'FastVideoArgs'
    },
    'trust_remote_code': {
        'kind': 'typed',
        'none_value': False,
        'path': 'trust_remote_code',
        'source': 'FastVideoArgs'
    },
    'use_ema': {
        'kind': 'typed',
        'path': 'training.ema.enabled',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'use_fsdp_inference': {
        'kind': 'typed',
        'none_value': False,
        'path': 'engine.use_fsdp_inference',
        'source': 'FastVideoArgs'
    },
    'vae_cpu_offload': {
        'kind': 'typed',
        'none_value': True,
        'path': 'engine.offload.vae',
        'source': 'FastVideoArgs'
    },
    'vae_parallel_decode': {
        'kind': 'typed',
        'none_value': False,
        'path': 'pipeline.minimax_h3.vae_parallel_decode',
        'source': 'FastVideoArgs'
    },
    'vae_parallel_decode_strategy': {
        'kind': 'typed',
        'path': 'pipeline.minimax_h3.vae_parallel_decode_strategy',
        'source': 'FastVideoArgs'
    },
    'vae_parallel_encode': {
        'kind': 'typed',
        'none_value': False,
        'path': 'pipeline.minimax_h3.vae_parallel_encode',
        'source': 'FastVideoArgs'
    },
    'validation_dataset_file': {
        'kind': 'typed',
        'path': 'training.validation.dataset_file',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'validation_guidance_scale': {
        'convert': 'string',
        'kind': 'typed',
        'path': 'training.validation.guidance_scale',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'validation_preprocessed_path': {
        'kind': 'removed',
        'note': 'no readers; no typed field',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'validation_sampling_steps': {
        'convert': 'comma_string',
        'kind': 'typed',
        'path': 'training.validation.sampling_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'validation_steps': {
        'kind': 'typed',
        'path': 'training.validation.every_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'video_decode_backend': {
        'kind': 'typed',
        'none_value': 'h3-vae',
        'path': 'pipeline.minimax_h3.video_decode_backend',
        'source': 'FastVideoArgs'
    },
    'visualization_steps': {
        'kind': 'typed',
        'path': 'training.validation.visualization_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'wandb_run_name': {
        'kind': 'typed',
        'path': 'training.tracker.run_name',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'warp_denoising_step': {
        'kind': 'typed',
        'path': 'training.distillation.warp_denoising_step',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'weight_decay': {
        'kind': 'typed',
        'path': 'training.optimizer.weight_decay',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'weight_only_checkpointing_steps': {
        'kind': 'typed',
        'path': 'training.checkpoint.weight_only_checkpointing_steps',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'weighting_scheme': {
        'kind': 'typed',
        'path': 'training.model.weighting_scheme',
        'roots': ['TrainingRunConfig'],
        'source': 'TrainingArgs'
    },
    'workload_type': {
        'convert': 'workload_type',
        'kind': 'typed',
        'none_value': 't2v',
        'path': 'pipeline.workload_type',
        'source': 'FastVideoArgs'
    }
}
PIPELINE_CONFIG_TYPED_HOMES: dict[str, str] = {
    'boundary_ratio': 'pipeline.boundary_ratio',
    'bsa_cdf_threshold': 'pipeline.longcat.bsa_cdf_threshold',
    'bsa_chunk_k': 'pipeline.longcat.bsa_chunk_k',
    'bsa_chunk_q': 'pipeline.longcat.bsa_chunk_q',
    'bsa_sparsity': 'pipeline.longcat.bsa_sparsity',
    'disable_autocast': 'engine.disable_autocast',
    'dit_precision': 'engine.precision.dit',
    'dmd_denoising_steps': 'pipeline.dmd_denoising_steps',
    'embedded_cfg_scale': 'pipeline.embedded_cfg_scale',
    'enable_bsa': 'pipeline.longcat.enable_bsa',
    'flow_shift': 'pipeline.flow_shift',
    'image_encoder_precision': 'engine.precision.image_encoder',
    'model_path': 'model_path',
    'pipeline_config_path': 'pipeline.components.pipeline_config_path',
    'text_encoder_precisions': 'engine.precision.text_encoders',
    'vae_decode_precision': 'engine.precision.vae_decode',
    'vae_precision': 'engine.precision.vae',
    'vae_sp': 'pipeline.vae_sp',
    'vae_tiling': 'pipeline.vae_tiling'
}
# END GENERATED FLAT_TO_TYPED


def _root_name(resolved: Any) -> str:
    return object.__getattribute__(resolved, "_struct").config_class.__name__


def _input_mapping(resolved: Any, path: str) -> dict[str, Any]:
    """Plain copy of a dict-valued field such as ``pipeline.experimental``; empty when the root lacks it."""
    present, node = _lookup(object.__getattribute__(resolved, "_struct"), path)
    return _to_plain(node) if present and isinstance(node, dict) else {}


def generic_refine_aliases(preset_overrides: Mapping[str, Any], experimental: Mapping[str, Any]) -> dict[str, Any]:
    """The ``refine_*`` values that ``FastVideoArgs`` received for these inputs, by generic key.

    ``preset_overrides.refine.enabled`` sets ``refine_enabled``; top-level ``refine_*`` keys of
    ``preset_overrides``, then of ``experimental``, set their own names. A missing key is absent from the result.
    """
    values: dict[str, Any] = {}
    refine = preset_overrides.get("refine")
    if isinstance(refine, Mapping) and "enabled" in refine:
        values["refine_enabled"] = refine["enabled"]
    for source in (preset_overrides, experimental):
        for name in GENERIC_REFINE_PATHS:
            if name in source:
                values[name] = source[name]
    return values


def resolved_generic_refine_aliases(resolved: Any) -> dict[str, Any]:
    """:func:`generic_refine_aliases` for the inputs of a resolved config."""
    return generic_refine_aliases(_input_mapping(resolved, "pipeline.preset_overrides"),
                                  _input_mapping(resolved, "pipeline.experimental"))


def initial_runtime_state(resolved: Any) -> dict[str, Any]:
    """Runtime state of a freshly resolved config, with the ``FastVideoArgs`` defaults.

    ``ray_runtime_env`` has no typed field; ``pipeline.experimental.ray_runtime_env`` supplies it.
    """
    return {
        "model_paths": {},
        "model_loaded": {
            "transformer": True,
            "vae": True,
            "upsampler": True,
        },
        "ray_placement_group": None,
        "ray_runtime_env": deepcopy(_input_mapping(resolved, "pipeline.experimental").get("ray_runtime_env")),
    }


def typed_path_of_flat_name(name: str) -> str | None:
    """Typed path that holds the value of a flat name, or ``None`` when the name has no typed field."""
    entry = FLAT_TO_TYPED.get(name)
    if entry is not None and entry["kind"] == "typed":
        return entry["path"]
    return GENERIC_REFINE_PATHS.get(name)


def _convert(value: Any, convert: str | None) -> Any:
    """Return a typed value in the flat format that ``FastVideoArgs`` or ``TrainingArgs`` used."""
    if value is None or convert is None:
        return value
    if convert == "workload_type":
        from fastvideo.api.schema import WorkloadType

        return WorkloadType(value)
    if convert == "execution_mode":
        from fastvideo.api.schema import ExecutionMode

        return ExecutionMode(value)
    if convert == "comma_string":
        return ",".join(str(item) for item in value)
    if convert == "string":
        return str(value)
    raise ValueError(f"unknown flat format {convert!r}")


def _derive(resolved: Any, entry: dict[str, Any]) -> Any:
    """Value of a ``derived`` entry, computed from its typed source path."""
    struct = object.__getattribute__(resolved, "_struct")
    if entry["derive"] == "torch_compile_kwargs":
        from fastvideo.api.compat import _compile_config_to_torch_kwargs

        return _compile_config_to_torch_kwargs(_to_config(_lookup(struct, "engine.compile")[1]))
    if entry["derive"] == "transformer_quant":
        present, name = _lookup(struct, entry["path"])
        if not present or name is None:
            return None
        from fastvideo.layers.quantization import get_quantization_config

        return get_quantization_config(name)()
    raise ValueError(f"unknown derived flat value {entry['derive']!r}")


def _flat_value(resolved: Any, name: str, entry: dict[str, Any]) -> Any:
    """Value of one flat name; raises ``AttributeError`` when the name has no value on this root."""
    root = _root_name(resolved)
    roots = entry.get("roots")
    if roots is not None and root not in roots:
        if "other_roots_value" in entry:
            return deepcopy(entry["other_roots_value"])
        raise AttributeError(f"{root} has no field {name!r}; it exists on {', '.join(roots)}")
    kind = entry["kind"]
    if kind == "typed":
        # The table names only paths of the root, so a path that is not present runs through an optional section
        # that is None, such as engine.quantization; its value is None.
        _, node = _lookup(object.__getattribute__(resolved, "_struct"), entry["path"])
        value = _to_config(node) if isinstance(node, _Struct) else _to_plain(node)
        if value is None:
            if root in _COMMAND_LINE_ROOTS and "cli_none_value" in entry:
                value = deepcopy(entry["cli_none_value"])
            else:
                value = deepcopy(entry.get("none_value"))
        return _convert(value, entry.get("convert"))
    if kind == "pipeline_config":
        return resolved.pipeline_config
    if kind == "property":
        return getattr(resolved, entry["path"])
    if kind == "derived":
        return _derive(resolved, entry)
    if kind == "removed":
        if entry.get("fallback") == "experimental":
            return deepcopy(_input_mapping(resolved, "pipeline.experimental").get(name))
        if entry.get("fallback") == "generic_refine_alias":
            return resolved_generic_refine_aliases(resolved).get(name)
        raise AttributeError(f"{root} has no field {name!r}: {entry.get('note', 'removed')}")
    raise AttributeError(f"{root} has no field {name!r}")


def read_flat_name(resolved: Any, name: str) -> Any:
    """Value of the ``FastVideoArgs`` / ``TrainingArgs`` attribute ``name`` for a resolved config.

    Runtime-state names return the shared mutable state. Every other value is computed once per config object and
    returned again on later reads. An unknown name raises ``AttributeError``, so ``getattr`` defaults still apply.
    """
    if name in RUNTIME_STATE_NAMES:
        return object.__getattribute__(resolved, "_runtime_state")[name]
    entry = FLAT_TO_TYPED.get(name)
    if entry is None:
        raise AttributeError(f"{_root_name(resolved)} has no field {name!r}")
    cache = object.__getattribute__(resolved, "_flat_values")
    if name not in cache:
        cache[name] = _flat_value(resolved, name, entry)
    return cache[name]


def sync_pipeline_config_mirrors(resolved: Any) -> None:
    """Copy the typed values of the latest override onto the ``PipelineConfig`` attributes that mirror them.

    Readers of a mirror attribute, such as ``pipeline_config.dmd_denoising_steps``, then see an override of its
    typed path, as they did when ``FastVideoArgs.override`` wrote the attribute directly. ``None`` leaves the
    attribute unchanged, as the flat keyword path did.
    """
    pipeline_config = resolved.pipeline_config
    override_log = object.__getattribute__(resolved, "_override_log")
    if pipeline_config is None or not override_log:
        return
    attributes = {path: attribute for attribute, path in PIPELINE_CONFIG_TYPED_HOMES.items()}
    _, values = override_log[-1]
    for path, value in values.items():
        attribute = attributes.get(path)
        if attribute is None or value is None or not hasattr(pipeline_config, attribute):
            continue
        mirrored = tuple(value) if attribute == "text_encoder_precisions" else deepcopy(value)
        object.__setattr__(pipeline_config, attribute, mirrored)


__all__ = [
    "FLAT_TO_TYPED",
    "GENERIC_REFINE_PATHS",
    "PIPELINE_CONFIG_TYPED_HOMES",
    "RUNTIME_STATE_NAMES",
    "generic_refine_aliases",
    "initial_runtime_state",
    "read_flat_name",
    "resolved_generic_refine_aliases",
    "sync_pipeline_config_mirrors",
    "typed_path_of_flat_name",
]
