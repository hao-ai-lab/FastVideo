# SPDX-License-Identifier: Apache-2.0
"""Guard: FastVideo code follows the environment-variable policy in
docs/contributing/env_vars.md.

The test parses every Python file under fastvideo/ (except fastvideo/third_party/
and the registry fastvideo/envs.py) with ``ast`` and reports code that:

- reads the environment directly for a name outside EXTERNAL_ALLOWLIST;
- writes the environment directly (os.environ, os.putenv, monkeypatch.setenv);
- uses the whole environment (os.environ.copy(), dict(os.environ), patch.dict);
- uses a registry field without one of its methods (``envs.X == "a"``);
- calls ``envs.X.get()`` outside a function, so the value is read at import.

It also checks every entry in fastvideo/envs.py (FASTVIDEO_ prefix, category,
description, at least one reader) and checks that the table in
docs/contributing/env_vars.md matches the registry.

Violations that existed when the policy was introduced are listed in
KNOWN_VIOLATIONS. The list only shrinks: a violation that is not listed fails,
and a listed violation that no longer exists also fails, so the entry gets
deleted.

Run ``python fastvideo/tests/contract/test_env_policy.py`` to regenerate the
table in docs/contributing/env_vars.md after editing fastvideo/envs.py.

Static analysis only: fastvideo/envs.py is loaded as a standalone file, so the
test imports neither fastvideo nor torch.
"""
import ast
import importlib.util
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "fastvideo"
REGISTRY_PATH = PACKAGE_ROOT / "envs.py"
DOC_PATH = REPO_ROOT / "docs" / "contributing" / "env_vars.md"
POLICY_DOC = "docs/contributing/env_vars.md"
EXCLUDED_DIRS = (PACKAGE_ROOT / "third_party", )

DOC_TABLE_BEGIN = "<!-- BEGIN GENERATED ENV TABLE: python fastvideo/tests/contract/test_env_policy.py -->"
DOC_TABLE_END = "<!-- END GENERATED ENV TABLE -->"

# Variables that other tools own. Code may read them directly with a literal
# name. A trailing "*" allows every name with that prefix.
EXTERNAL_ALLOWLIST = {
    "CUDA_*": "CUDA runtime and device selection.",
    "NCCL_*": "NCCL communication library.",
    "TORCH_*": "PyTorch runtime settings.",
    "OMP_*": "OpenMP threading.",
    "HF_HUB_*": "huggingface_hub settings.",
    "RANK": "Set by torchrun and other launchers.",
    "LOCAL_RANK": "Set by torchrun and other launchers.",
    "WORLD_SIZE": "Set by torchrun and other launchers.",
    "MASTER_ADDR": "Set by torchrun and other launchers.",
    "MASTER_PORT": "Set by torchrun and other launchers.",
    "HOME": "User home directory.",
    "PATH": "Executable search path.",
    "HF_TOKEN": "Token variable that huggingface_hub reads.",
    "HUGGING_FACE_HUB_TOKEN": "Token variable that huggingface_hub reads.",
}

# Methods that registry fields expose; ``get`` and ``is_set`` count as reads.
REGISTRY_METHODS = {"get", "set", "override", "is_set", "clear"}
REGISTRY_READ_METHODS = {"get", "is_set"}

# Violations that existed when the policy was introduced, as
# "<path>: <kind> <name>" -> number of occurrences. Lower or delete an entry when
# its violations are fixed; never add one. Kinds are described in
# docs/contributing/env_vars.md.
KNOWN_VIOLATIONS: dict[str, int] = {
    'fastvideo/attention/backends/flash_attn.py: read FASTVIDEO_NVFP4_FA4': 1,
    'fastvideo/attention/backends/video_sparse_attn_h3_probe.py: read FASTVIDEO_H3_VSA_PROBE': 1,
    'fastvideo/attention/layer.py: read FASTVIDEO_DISABLE_ATTENTION_COMPILE': 2,
    'fastvideo/attention/selector.py: read <dynamic>': 1,
    'fastvideo/attention/utils/flash_attn_default.py: import-time-read FASTVIDEO_FA4': 1,
    'fastvideo/benchmarks/eval_metalfx_rife.py: write FASTVIDEO_MLX_COMPILE': 1,
    'fastvideo/benchmarks/mlx_fastwan_bench.py: read FASTVIDEO_MLX_COMPILE': 1,
    'fastvideo/benchmarks/mlx_fastwan_bench.py: read FASTVIDEO_MLX_FAST_NORM': 1,
    'fastvideo/benchmarks/mlx_fastwan_bench.py: write FASTVIDEO_MLX_COMPILE': 1,
    'fastvideo/distributed/device_communicators/cpu_communicator.py: read VLLM_DIST_IDENT': 1,
    'fastvideo/entrypoints/cli/utils.py: whole-environ': 1,
    'fastvideo/entrypoints/openai/api_server.py: write FASTVIDEO_STAGE_LOGGING': 1,
    'fastvideo/entrypoints/streaming/prompt/providers/cerebras.py: read <dynamic>': 1,
    'fastvideo/entrypoints/streaming/prompt/providers/groq.py: read <dynamic>': 1,
    'fastvideo/entrypoints/streaming/worker.py: write CUDA_VISIBLE_DEVICES': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_FFMPEG_BIN': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_NVENC_BF': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_NVENC_PRESET': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_NVENC_QP': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_NVENC_RC': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_NVENC_TUNE': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_OUTPUT_PIX_FMT': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_VIDEO_CODEC': 1,
    'fastvideo/entrypoints/video_generator.py: read FASTVIDEO_X264_PRESET': 1,
    'fastvideo/entrypoints/video_generator.py: write CUTE_DSL_ENABLE_TVM_FFI': 1,
    'fastvideo/entrypoints/video_generator.py: write FASTVIDEO_NVFP4_FA4': 1,
    'fastvideo/envs.py: prefix CMAKE_BUILD_TYPE': 1,
    'fastvideo/envs.py: prefix CUDA_VISIBLE_DEVICES': 1,
    'fastvideo/envs.py: prefix HCCL_SO_PATH': 1,
    'fastvideo/envs.py: prefix LD_LIBRARY_PATH': 1,
    'fastvideo/envs.py: prefix LOCAL_RANK': 1,
    'fastvideo/envs.py: prefix MAX_JOBS': 1,
    'fastvideo/envs.py: prefix NVCC_THREADS': 1,
    'fastvideo/envs.py: prefix VERBOSE': 1,
    'fastvideo/envs.py: unread CMAKE_BUILD_TYPE': 1,
    'fastvideo/envs.py: unread CUDA_VISIBLE_DEVICES': 1,
    'fastvideo/envs.py: unread FASTVIDEO_ENGINE_ITERATION_TIMEOUT_S': 1,
    'fastvideo/envs.py: unread FASTVIDEO_RINGBUFFER_WARNING_INTERVAL': 1,
    'fastvideo/envs.py: unread FASTVIDEO_SERVER_DEV_MODE': 1,
    'fastvideo/envs.py: unread FASTVIDEO_TARGET_DEVICE': 1,
    'fastvideo/envs.py: unread FASTVIDEO_TEST_DYNAMO_FULLGRAPH_CAPTURE': 1,
    'fastvideo/envs.py: unread FASTVIDEO_TRACE_FUNCTION': 1,
    'fastvideo/envs.py: unread FASTVIDEO_USE_PRECOMPILED': 1,
    'fastvideo/envs.py: unread LD_LIBRARY_PATH': 1,
    'fastvideo/envs.py: unread MAX_JOBS': 1,
    'fastvideo/envs.py: unread NVCC_THREADS': 1,
    'fastvideo/envs.py: unread VERBOSE': 1,
    'fastvideo/eval/__init__.py: write TORCH_HOME': 1,
    'fastvideo/eval/datasets/physics_iq.py: read FASTVIDEO_PHYSICS_IQ_BUCKET_URL': 1,
    'fastvideo/eval/datasets/vbench.py: read VBENCH_FULL_INFO_JSON': 1,
    'fastvideo/eval/metrics/audio/frechet_distance/metric.py: read <dynamic>': 1,
    'fastvideo/eval/metrics/common/fvd/metric.py: read <dynamic>': 1,
    'fastvideo/eval/metrics/judge/third_person_separation/metric.py: read <dynamic>': 1,
    'fastvideo/eval/metrics/vbench/scene/metric.py: write VIDEO_MAX_PIXELS': 1,
    'fastvideo/eval/models.py: read FASTVIDEO_EVAL_CACHE': 1,
    'fastvideo/logger.py: import-time-read FASTVIDEO_CONFIGURE_LOGGING': 1,
    'fastvideo/logger.py: import-time-read FASTVIDEO_LOGGING_CONFIG_PATH': 1,
    'fastvideo/logger.py: import-time-read FASTVIDEO_LOGGING_LEVEL': 1,
    'fastvideo/logger.py: import-time-read FASTVIDEO_LOGGING_PREFIX': 1,
    'fastvideo/mlx_runtime/fastwan.py: read FASTVIDEO_MLX_COMPILE': 1,
    'fastvideo/mlx_runtime/fastwan.py: read FASTVIDEO_MLX_DQ_GEMM': 1,
    'fastvideo/mlx_runtime/fastwan.py: read FASTVIDEO_MLX_FAST_NORM': 1,
    'fastvideo/mlx_runtime/fastwan.py: read FASTVIDEO_MLX_WINDOW': 1,
    'fastvideo/mlx_runtime/fastwan.py: read FASTVIDEO_MLX_WINDOW_SINK': 1,
    'fastvideo/mlx_runtime/memory.py: write <dynamic>': 1,
    'fastvideo/mlx_runtime/wan22.py: read FASTVIDEO_MLX_COMPILE': 1,
    'fastvideo/models/encoders/gemma.py: read LTX2_FASTVIDEO_GEMMA_LOG': 2,
    'fastvideo/models/encoders/reason1.py: read FASTVIDEO_REASON1_WEIGHTS_PATH': 1,
    'fastvideo/models/loader/benchmarks/benchmark_weight_loading.py: write MASTER_ADDR': 1,
    'fastvideo/models/loader/benchmarks/benchmark_weight_loading.py: write MASTER_PORT': 1,
    'fastvideo/models/loader/benchmarks/benchmark_weight_loading.py: write RANK': 1,
    'fastvideo/models/loader/benchmarks/benchmark_weight_loading.py: write WORLD_SIZE': 1,
    'fastvideo/models/loader/fsdp_load.py: read FASTVIDEO_FSDP2_AUTOWRAP': 1,
    'fastvideo/models/loader/fsdp_load.py: read FASTVIDEO_FSDP2_MIN_PARAMS': 1,
    'fastvideo/models/loader/fsdp_load.py: read FASTVIDEO_H3_VSA_PROBE': 1,
    'fastvideo/models/vaes/ltx2vae.py: read <dynamic>': 1,
    'fastvideo/performance/hf_store.py: read HF_REPO_ID': 1,
    'fastvideo/performance/hf_store.py: read PERFORMANCE_TRACKING_SYNC_REUSE_TTL_SECONDS': 1,
    'fastvideo/performance_dashboard/api.py: read PERFORMANCE_TRACKING_ROOT': 1,
    'fastvideo/pipelines/basic/ltx2/stages/ltx2_audio_decoding.py: read LTX2_DISABLE_AUDIO_AUTOCAST': 1,
    'fastvideo/pipelines/basic/ltx2/stages/ltx2_denoising.py: read FASTVIDEO_NVTX_PROFILE': 1,
    'fastvideo/pipelines/basic/ltx2/stages/ltx2_denoising.py: read LTX2_USE_DISTILLED_SIGMAS': 1,
    'fastvideo/pipelines/basic/magi_human/magi_human_pipeline.py: write HF_TOKEN': 1,
    'fastvideo/pipelines/preprocess/preprocess_kandinsky5_overfit.py: read KANDINSKY5_OVERFIT_DATA_DIR': 1,
    'fastvideo/pipelines/preprocess/preprocess_kandinsky5_overfit.py: read KANDINSKY5_OVERFIT_OUTPUT_DIR': 1,
    'fastvideo/pipelines/preprocess/preprocess_kandinsky5_overfit.py: write MASTER_ADDR': 1,
    'fastvideo/pipelines/preprocess/preprocess_kandinsky5_overfit.py: write MASTER_PORT': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: read LTX2_OVERFIT_CAPTION_JSON': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: read LTX2_OVERFIT_DATA_DIR': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: read LTX2_OVERFIT_MODEL': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: read LTX2_OVERFIT_NUM_COPIES': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: read LTX2_OVERFIT_OUTPUT_DIR': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: read LTX2_OVERFIT_VIDEO_SUBDIR': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: write LOCAL_RANK': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: write MASTER_ADDR': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: write MASTER_PORT': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: write RANK': 1,
    'fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py: write WORLD_SIZE': 1,
    'fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py: write LOCAL_RANK': 1,
    'fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py: write MASTER_ADDR': 1,
    'fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py: write MASTER_PORT': 1,
    'fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py: write RANK': 1,
    'fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py: write WORLD_SIZE': 1,
    'fastvideo/pipelines/stages/denoising.py: read FASTVIDEO_FLUX2_DISABLE_BF16_REDUCED_PRECISION_REDUCTION': 1,
    'fastvideo/pipelines/stages/latent_preparation.py: read FASTVIDEO_COSMOS25_LOG_KNOBS': 1,
    'fastvideo/tests/api/test_attention_selector_resolution.py: write FASTVIDEO_ATTENTION_BACKEND': 17,
    'fastvideo/tests/attention/test_flash_attn_no_pad_resolver.py: write FASTVIDEO_FA4': 3,
    'fastvideo/tests/attention/test_flash_attn_nvfp4_opt_out.py: write FASTVIDEO_NVFP4_FA4': 4,
    'fastvideo/tests/attention/test_vsa_h3_backward.py: write FASTVIDEO_KERNEL_VSA_FORCE_TRITON': 1,
    'fastvideo/tests/attention/test_vsa_h3_backward.py: write FASTVIDEO_VSA_CUTEDSL': 2,
    'fastvideo/tests/attention/test_vsa_h3_backward.py: write FASTVIDEO_VSA_TRITON': 2,
    'fastvideo/tests/attention/test_vsa_h3_sm100a_route.py: write <dynamic>': 22,
    'fastvideo/tests/contract/test_profiler_regions.py: whole-environ': 2,
    'fastvideo/tests/contract/test_profiler_regions.py: write FASTVIDEO_NVTX_PROFILE': 4,
    'fastvideo/tests/contract/test_ssim_ci_runner.py: write CUDA_VISIBLE_DEVICES': 1,
    'fastvideo/tests/contract/test_tensor_golden_isolation.py: read FASTVIDEO_ATTENTION_BACKEND': 2,
    'fastvideo/tests/contract/test_tensor_golden_isolation.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/contract/test_wan_validation_order.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_hunyuanvideo.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_lingbot_video.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_ltx2.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_wan.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_ulysses_a2a_parity.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_ulysses_fault_injection.py: read FASTVIDEO_ULYSSES_FAULT_RANK': 1,
    'fastvideo/tests/distributed/test_ulysses_fault_injection.py: read FASTVIDEO_ULYSSES_FAULT_STAGE': 1,
    'fastvideo/tests/distributed/test_ulysses_fault_injection.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_ulysses_fault_injection.py: write FASTVIDEO_ULYSSES_A2A': 1,
    'fastvideo/tests/encoders/test_clip_encoder.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_clip_encoder.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_hyt5_encoder.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_hyt5_encoder.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_llama_encoder.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_llama_encoder.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_minimax_h3_qwen3_vl_checkpoint_fp8.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_minimax_h3_qwen3_vl_checkpoint_fp8.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_minimax_h3_qwen3_vl_checkpoint_nvfp4.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_minimax_h3_qwen3_vl_checkpoint_nvfp4.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_minimax_h3_qwen3_vl_truncation.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_minimax_h3_qwen3_vl_truncation.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_qwen2_5_encoder.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_qwen2_5_encoder.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_siglip_encoder.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_siglip_encoder.py: write MASTER_PORT': 1,
    'fastvideo/tests/encoders/test_t5_encoder.py: write MASTER_ADDR': 1,
    'fastvideo/tests/encoders/test_t5_encoder.py: write MASTER_PORT': 1,
    'fastvideo/tests/entrypoints/streaming/test_prompt_providers.py: write CEREBRAS_API_KEY': 3,
    'fastvideo/tests/entrypoints/streaming/test_prompt_providers.py: write GROQ_API_KEY': 2,
    'fastvideo/tests/entrypoints/test_openai_api_integration.py: whole-environ': 1,
    'fastvideo/tests/entrypoints/test_openai_video_client.py: whole-environ': 1,
    'fastvideo/tests/entrypoints/test_video_generator.py: write FASTVIDEO_FFMPEG_BIN': 1,
    'fastvideo/tests/entrypoints/test_video_generator.py: write FASTVIDEO_OUTPUT_PIX_FMT': 1,
    'fastvideo/tests/entrypoints/test_video_generator.py: write FASTVIDEO_VIDEO_CODEC': 1,
    'fastvideo/tests/golden_gate/_harness.py: read <dynamic>': 1,
    'fastvideo/tests/golden_gate/_harness.py: read FASTVIDEO_FA4': 2,
    'fastvideo/tests/golden_gate/_harness.py: read FASTVIDEO_GOLDEN_GATE_DIR': 1,
    'fastvideo/tests/golden_gate/_harness.py: read FASTVIDEO_SSIM_REFERENCE_HF_REPO': 1,
    'fastvideo/tests/golden_gate/_harness.py: write CUBLAS_WORKSPACE_CONFIG': 1,
    'fastvideo/tests/golden_gate/_harness.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/golden_gate/_harness.py: write LOCAL_RANK': 1,
    'fastvideo/tests/golden_gate/_harness.py: write MASTER_ADDR': 1,
    'fastvideo/tests/golden_gate/_harness.py: write MASTER_PORT': 1,
    'fastvideo/tests/golden_gate/_harness.py: write RANK': 1,
    'fastvideo/tests/golden_gate/_harness.py: write WORLD_SIZE': 1,
    'fastvideo/tests/golden_gate/_tensor_golden.py: read FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/golden_gate/_tensor_golden.py: write FASTVIDEO_ATTENTION_BACKEND': 3,
    'fastvideo/tests/golden_gate/test_kandinsky5.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/golden_gate/test_kandinsky5.py: write FASTVIDEO_FA4': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: read FASTVIDEO_ATTENTION_BACKEND': 2,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: read FASTVIDEO_SSIM_REFERENCE_HF_REPO': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: read MINIMAX_H3_GATE_GOLDEN_DIR': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: read MINIMAX_H3_GATE_LAYER': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: read MINIMAX_H3_MODEL_ROOT': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: write CUBLAS_WORKSPACE_CONFIG': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: write LOCAL_RANK': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: write MASTER_ADDR': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: write MASTER_PORT': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: write RANK': 1,
    'fastvideo/tests/golden_gate/test_minimax_h3_t2v.py: write WORLD_SIZE': 1,
    'fastvideo/tests/golden_gate/test_wan_denoising.py: write FASTVIDEO_CFG_GATE_STEP': 1,
    'fastvideo/tests/hooks/test_activation_trace.py: write FASTVIDEO_TRACE_ACTIVATIONS': 6,
    'fastvideo/tests/hooks/test_activation_trace.py: write FASTVIDEO_TRACE_LAYERS': 5,
    'fastvideo/tests/hooks/test_activation_trace.py: write FASTVIDEO_TRACE_OUTPUT': 5,
    'fastvideo/tests/hooks/test_activation_trace.py: write FASTVIDEO_TRACE_STATS': 1,
    'fastvideo/tests/hooks/test_activation_trace.py: write FASTVIDEO_TRACE_STEPS': 1,
    'fastvideo/tests/inference/bsa/test_bsa_inference.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/inference/lora/test_lora_inference_similarity.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/inference/lora/test_lora_inference_similarity.py: write MASTER_ADDR': 1,
    'fastvideo/tests/inference/lora/test_lora_inference_similarity.py: write MASTER_PORT': 1,
    'fastvideo/tests/inference/test_basic_fasth3_profile.py: write <dynamic>': 2,
    'fastvideo/tests/inference/test_inference_regional_compile.py: write FASTVIDEO_DISABLE_ATTENTION_COMPILE': 9,
    'fastvideo/tests/inference/test_inference_regional_compile.py: write FASTVIDEO_H3_VSA_PROBE': 4,
    'fastvideo/tests/inference/test_inference_regional_compile.py: write FASTVIDEO_VSA_SM100A': 3,
    'fastvideo/tests/inference/vmoba/test_vmoba_inference.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/layers/test_rmsnorm_forward_dispatch.py: whole-environ': 1,
    'fastvideo/tests/mlx/test_memory_limits.py: read PYTORCH_MPS_HIGH_WATERMARK_RATIO': 1,
    'fastvideo/tests/mlx/test_memory_limits.py: read PYTORCH_MPS_LOW_WATERMARK_RATIO': 1,
    'fastvideo/tests/mlx/test_memory_limits.py: write PYTORCH_MPS_HIGH_WATERMARK_RATIO': 1,
    'fastvideo/tests/mlx/test_memory_limits.py: write PYTORCH_MPS_LOW_WATERMARK_RATIO': 1,
    'fastvideo/tests/mlx/test_mlx_affine_dq_gemm.py: read FASTVIDEO_MLX_DQ_GEMM': 1,
    'fastvideo/tests/mlx/test_mlx_affine_dq_gemm.py: write FASTVIDEO_MLX_DQ_GEMM': 11,
    'fastvideo/tests/mlx/test_mlx_minimax_h3_vsa_regressions.py: write MASTER_ADDR': 1,
    'fastvideo/tests/mlx/test_mlx_minimax_h3_vsa_regressions.py: write MASTER_PORT': 1,
    'fastvideo/tests/mlx/test_mlx_taeh3.py: read TAEH3_REFERENCE_DIR': 1,
    'fastvideo/tests/mlx/test_mlx_wan22_prompt_cache_fingerprint.py: write HOME': 1,
    'fastvideo/tests/mlx/test_mlx_wan22_real_weights_parity.py: read FASTVIDEO_WAN22_5B_ALLOW_LOW_MEMORY': 1,
    'fastvideo/tests/mlx/test_mlx_wan22_real_weights_parity.py: read FASTVIDEO_WAN22_5B_ROOT': 1,
    'fastvideo/tests/mlx/test_mlx_wan22_real_weights_parity.py: write MASTER_ADDR': 1,
    'fastvideo/tests/mlx/test_mlx_wan22_real_weights_parity.py: write MASTER_PORT': 1,
    'fastvideo/tests/mlx/tiny_h3.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/mlx/tiny_h3.py: write MASTER_ADDR': 1,
    'fastvideo/tests/mlx/tiny_h3.py: write MASTER_PORT': 1,
    'fastvideo/tests/mlx/tiny_wan.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/mlx/tiny_wan.py: write MASTER_ADDR': 1,
    'fastvideo/tests/mlx/tiny_wan.py: write MASTER_PORT': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read <dynamic>': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read CFLAGS': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read CMAKE_ARGS': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read CXXFLAGS': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read FASTVIDEO_KERNEL_CACHE_ROOT': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read FASTVIDEO_KERNEL_PREBUILT_INFO': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read GPU_BACKEND': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: read LDFLAGS': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: whole-environ': 1,
    'fastvideo/tests/modal/kernel_cache_smoke.py: read BUILDKITE_COMMIT': 1,
    'fastvideo/tests/modal/kernel_cache_smoke.py: read BUILDKITE_REPO': 1,
    'fastvideo/tests/modal/kernel_cache_smoke.py: read FASTVIDEO_MODAL_IMAGE': 1,
    'fastvideo/tests/modal/kernel_cache_smoke.py: read IMAGE_VERSION': 1,
    'fastvideo/tests/modal/kernel_cache_smoke.py: read UV_TORCH_BACKEND': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read <dynamic>': 2,
    'fastvideo/tests/modal/launch_l40s_job.py: read BUILDKITE_COMMIT': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read BUILDKITE_PULL_REQUEST': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read BUILDKITE_REPO': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read FASTVIDEO_MODAL_IMAGE': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read FASTVIDEO_MODAL_VOLUME': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read FASTVIDEO_PERFORMANCE_PROFILE_VERSION': 2,
    'fastvideo/tests/modal/launch_l40s_job.py: read IMAGE_VERSION': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read UV_TORCH_BACKEND': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: whole-environ': 1,
    'fastvideo/tests/modal/modal_image_utils.py: read UV_TORCH_BACKEND': 1,
    'fastvideo/tests/modal/pr_test.py: read <dynamic>': 2,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_BRANCH': 1,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_BUILD_ID': 1,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_BUILD_URL': 1,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_COMMIT': 2,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_JOB_ID': 1,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_PULL_REQUEST': 2,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_REPO': 2,
    'fastvideo/tests/modal/pr_test.py: read BUILDKITE_SOURCE': 1,
    'fastvideo/tests/modal/pr_test.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/modal/pr_test.py: read HF_API_KEY': 1,
    'fastvideo/tests/modal/pr_test.py: read IMAGE_VERSION': 1,
    'fastvideo/tests/modal/pr_test.py: read TEST_SCOPE': 1,
    'fastvideo/tests/modal/pr_test.py: read UV_TORCH_BACKEND': 1,
    'fastvideo/tests/modal/pr_test.py: read WANDB_API_KEY': 1,
    'fastvideo/tests/modal/ssim_test.py: read <dynamic>': 1,
    'fastvideo/tests/modal/ssim_test.py: read BUILDKITE_COMMIT': 1,
    'fastvideo/tests/modal/ssim_test.py: read BUILDKITE_PULL_REQUEST': 1,
    'fastvideo/tests/modal/ssim_test.py: read BUILDKITE_REPO': 1,
    'fastvideo/tests/modal/ssim_test.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/modal/ssim_test.py: read IMAGE_VERSION': 1,
    'fastvideo/tests/modal/ssim_test.py: read UV_TORCH_BACKEND': 1,
    'fastvideo/tests/modal/ssim_test.py: whole-environ': 2,
    'fastvideo/tests/modal/test_kernel_build_cache.py: write <dynamic>': 2,
    'fastvideo/tests/modal/test_kernel_build_cache.py: write TORCH_CUDA_ARCH_LIST': 2,
    'fastvideo/tests/modal/test_pr_test.py: write BUILDKITE_COMMIT': 2,
    'fastvideo/tests/modal/test_pr_test.py: write BUILDKITE_PULL_REQUEST': 2,
    'fastvideo/tests/modal/test_pr_test.py: write BUILDKITE_REPO': 2,
    'fastvideo/tests/nightly/test_e2e_dmd_t2v_crush_smol.py: write WANDB_MODE': 1,
    'fastvideo/tests/nightly/test_e2e_i2v_overfit_single_sample.py: write WANDB_MODE': 1,
    'fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py: read <dynamic>': 1,
    'fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py: read KANDINSKY5_E2E_NUM_GPUS': 1,
    'fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py: whole-environ': 3,
    'fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py: write WANDB_MODE': 1,
    'fastvideo/tests/nightly/test_e2e_ltx2_overfit_new_stack.py: read FASTVIDEO_NIGHTLY': 1,
    'fastvideo/tests/nightly/test_e2e_ltx2_overfit_new_stack.py: whole-environ': 2,
    'fastvideo/tests/nightly/test_e2e_ltx2_overfit_new_stack.py: write FASTVIDEO_NIGHTLY': 1,
    'fastvideo/tests/nightly/test_e2e_overfit_single_sample.py: write WANDB_MODE': 1,
    'fastvideo/tests/performance/compare_baseline.py: read BUILDKITE_BRANCH': 2,
    'fastvideo/tests/performance/compare_baseline.py: read BUILDKITE_BUILD_ID': 1,
    'fastvideo/tests/performance/compare_baseline.py: read BUILDKITE_BUILD_URL': 1,
    'fastvideo/tests/performance/compare_baseline.py: read BUILDKITE_COMMIT': 3,
    'fastvideo/tests/performance/compare_baseline.py: read BUILDKITE_JOB_ID': 1,
    'fastvideo/tests/performance/compare_baseline.py: read BUILDKITE_PULL_REQUEST': 2,
    'fastvideo/tests/performance/compare_baseline.py: read GITHUB_STEP_SUMMARY': 1,
    'fastvideo/tests/performance/compare_baseline.py: read PERFORMANCE_TRACKING_ROOT': 1,
    'fastvideo/tests/performance/compare_baseline.py: read PERF_PYTEST_RC': 4,
    'fastvideo/tests/performance/compare_baseline.py: read PERF_REPORTS_DIR': 1,
    'fastvideo/tests/performance/compare_baseline.py: read PERF_RUN_SOURCE': 1,
    'fastvideo/tests/performance/compare_baseline.py: read PERF_UPLOAD_POLICY': 1,
    'fastvideo/tests/performance/compare_baseline.py: read TEST_SCOPE': 2,
    'fastvideo/tests/performance/dashboard.py: read BUILDKITE_COMMIT': 1,
    'fastvideo/tests/performance/dashboard.py: read DASHBOARD_DAYS': 1,
    'fastvideo/tests/performance/dashboard.py: read PERFORMANCE_TRACKING_ROOT': 1,
    'fastvideo/tests/performance/dashboard.py: read PERF_REPORTS_DIR': 1,
    'fastvideo/tests/performance/identity.py: read FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/performance/identity.py: whole-environ': 1,
    'fastvideo/tests/performance/seed_baseline.py: read PERFORMANCE_RESEED_STAGING_ROOT': 1,
    'fastvideo/tests/performance/seed_baseline.py: read PERFORMANCE_TRACKING_ROOT': 1,
    'fastvideo/tests/performance/seed_baseline.py: read USER': 1,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write BUILDKITE_BRANCH': 3,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write BUILDKITE_BUILD_ID': 3,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write BUILDKITE_BUILD_URL': 2,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write BUILDKITE_JOB_ID': 3,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write BUILDKITE_PULL_REQUEST': 3,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write PERF_PYTEST_RC': 8,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write PERF_RUN_SOURCE': 23,
    'fastvideo/tests/performance/test_compare_baseline_policy.py: write TEST_SCOPE': 3,
    'fastvideo/tests/performance/test_dashboard_service.py: write <dynamic>': 1,
    'fastvideo/tests/performance/test_dashboard_service.py: write HF_TOKEN': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read BUILDKITE_BRANCH': 2,
    'fastvideo/tests/performance/test_inference_performance.py: read BUILDKITE_BUILD_ID': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read BUILDKITE_BUILD_URL': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read BUILDKITE_COMMIT': 2,
    'fastvideo/tests/performance/test_inference_performance.py: read BUILDKITE_JOB_ID': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read BUILDKITE_PULL_REQUEST': 2,
    'fastvideo/tests/performance/test_inference_performance.py: read FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read FASTVIDEO_PERFORMANCE_PROFILE_VERSION': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read FASTVIDEO_STAGE_LOGGING': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read IMAGE_VERSION': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read PERF_RUN_SOURCE': 1,
    'fastvideo/tests/performance/test_inference_performance.py: read TEST_SCOPE': 2,
    'fastvideo/tests/performance/test_inference_performance.py: write FASTVIDEO_STAGE_LOGGING': 3,
    'fastvideo/tests/performance/test_inference_performance_identity.py: write FASTVIDEO_ATTENTION_BACKEND': 5,
    'fastvideo/tests/performance/test_inference_performance_identity.py: write FASTVIDEO_CONTAINER_IMAGE_REF': 2,
    'fastvideo/tests/performance/test_inference_performance_identity.py: write FASTVIDEO_FA4': 2,
    'fastvideo/tests/performance/test_inference_performance_identity.py: write FASTVIDEO_PERFORMANCE_PROFILE_VERSION': 1,
    'fastvideo/tests/performance/test_inference_performance_identity.py: write IMAGE_VERSION': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write BUILDKITE_BRANCH': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write BUILDKITE_BUILD_ID': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write BUILDKITE_BUILD_URL': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write BUILDKITE_COMMIT': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write BUILDKITE_JOB_ID': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write BUILDKITE_PULL_REQUEST': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write PERF_RUN_SOURCE': 1,
    'fastvideo/tests/performance/test_inference_performance_result_schema.py: write TEST_SCOPE': 1,
    'fastvideo/tests/ssim/bootstrap_references.py: read <dynamic>': 3,
    'fastvideo/tests/ssim/ci_runner.py: whole-environ': 1,
    'fastvideo/tests/ssim/conftest.py: read <dynamic>': 1,
    'fastvideo/tests/ssim/conftest.py: read FASTVIDEO_SSIM_MODEL_ID': 1,
    'fastvideo/tests/ssim/conftest.py: read FASTVIDEO_SSIM_SKIP_REFERENCE_DOWNLOAD': 1,
    'fastvideo/tests/ssim/conftest.py: write <dynamic>': 3,
    'fastvideo/tests/ssim/inference_similarity_utils.py: read FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/ssim/inference_similarity_utils.py: write FASTVIDEO_ATTENTION_BACKEND': 3,
    'fastvideo/tests/ssim/reference_utils.py: read <dynamic>': 1,
    'fastvideo/tests/ssim/reference_videos_cli.py: read <dynamic>': 3,
    'fastvideo/tests/ssim/test_dreamx_world_similarity.py: read DREAMX_WORLD_AR_SSIM_MODEL_PATH': 1,
    'fastvideo/tests/ssim/test_dreamx_world_similarity.py: read DREAMX_WORLD_SSIM_MODEL_PATH': 1,
    'fastvideo/tests/ssim/test_flux_t2i_similarity.py: read FLUX_T2I_MODEL_DIR': 1,
    'fastvideo/tests/ssim/test_gamecraft_similarity.py: read GAMECRAFT_MODEL_PATH': 1,
    'fastvideo/tests/ssim/test_gamecraft_similarity.py: write FASTVIDEO_ATTENTION_BACKEND': 2,
    'fastvideo/tests/ssim/test_gen3c_similarity.py: read GEN3C_MODEL_PATH': 1,
    'fastvideo/tests/ssim/test_gen3c_similarity.py: read GEN3C_TEST_IMAGE_PATH': 1,
    'fastvideo/tests/ssim/test_gen3c_similarity.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/ssim/test_glm_image_similarity.py: read GLM_IMAGE_LOCAL_WEIGHTS_DIR': 1,
    'fastvideo/tests/ssim/test_glm_image_similarity.py: read GLM_IMAGE_MODEL_DIR': 1,
    'fastvideo/tests/ssim/test_kandinsky5_similarity.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/ssim/test_kandinsky5_similarity.py: write FASTVIDEO_FA4': 3,
    'fastvideo/tests/ssim/test_lingbot_similarity.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/ssim/test_longcat_similarity.py: write FASTVIDEO_ATTENTION_BACKEND': 3,
    'fastvideo/tests/ssim/test_matrixgame2_similarity.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/ssim/test_matrixgame3_similarity.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/ssim/test_sd35_similarity.py: read SD35_MODEL_DIR': 1,
    'fastvideo/tests/ssim/test_zimage_similarity.py: read ZIMAGE_MODEL_DIR': 1,
    'fastvideo/tests/ssim/test_zimage_similarity.py: read ZIMAGE_MODEL_REVISION': 1,
    'fastvideo/tests/stages/_denoising_fixtures.py: write FASTVIDEO_CFG_GATE_STEP': 2,
    'fastvideo/tests/stages/test_kandinsky5_dmd_stage_backend_engages.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/stages/test_minimax_h3_encoding_offload.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/train/methods/grad_norm_regression.py: read <dynamic>': 1,
    'fastvideo/tests/train/methods/test_cosmos_finetune.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_cosmos_finetune.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_ltx2_finetune.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_ltx2_finetune.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_matrixgame2_finetune.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_matrixgame2_finetune.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_minimax_h3_finetune.py: whole-environ': 2,
    'fastvideo/tests/train/methods/test_streaming_long_tuning.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_streaming_long_tuning.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_wan_causal_cd.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_wan_causal_cd.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_wan_causal_dfsft.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_wan_causal_dfsft.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_wan_causal_dfsft_framewise.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_wan_causal_dfsft_framewise.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_wan_causal_tfsft.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_wan_causal_tfsft.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/methods/test_wan_finetune.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/methods/test_wan_finetune.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_kandinsky5_qat_attention_engages.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/train/models/test_kandinsky5_qat_attention_engages.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_kandinsky5_qat_attention_engages.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_cosmos.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_cosmos.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_hunyuan.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_hunyuan.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_kandinsky5.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/train/models/test_load_kandinsky5.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_kandinsky5.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_longcat.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_longcat.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_ltx2.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_ltx2.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_matrixgame2.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_matrixgame2.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_wan.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_wan.py: write MASTER_PORT': 1,
    'fastvideo/tests/train/models/test_load_wan_causal.py: write MASTER_ADDR': 1,
    'fastvideo/tests/train/models/test_load_wan_causal.py: write MASTER_PORT': 1,
    'fastvideo/tests/training/VSA/test_training_loss_VSA.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/training/VSA/test_training_loss_VSA.py: write WANDB_MODE': 1,
    'fastvideo/tests/training/Vanilla/mfu_calculation.py: read PYTHONPATH': 1,
    'fastvideo/tests/training/Vanilla/mfu_calculation.py: write MASTER_ADDR': 1,
    'fastvideo/tests/training/Vanilla/mfu_calculation.py: write MASTER_PORT': 1,
    'fastvideo/tests/training/Vanilla/mfu_calculation.py: write PYTHONPATH': 1,
    'fastvideo/tests/training/Vanilla/mfu_calculation.py: write WANDB_MODE': 1,
    'fastvideo/tests/training/Vanilla/test_training_loss.py: write MASTER_ADDR': 1,
    'fastvideo/tests/training/Vanilla/test_training_loss.py: write MASTER_PORT': 1,
    'fastvideo/tests/training/Vanilla/test_training_loss.py: write WANDB_MODE': 1,
    'fastvideo/tests/training/distill/test_anyflow_smoke.py: whole-environ': 2,
    'fastvideo/tests/training/distill/test_distill_dmd.py: write MASTER_ADDR': 1,
    'fastvideo/tests/training/distill/test_distill_dmd.py: write MASTER_PORT': 1,
    'fastvideo/tests/training/distill/test_distill_dmd.py: write WANDB_MODE': 1,
    'fastvideo/tests/training/lora/test_lora_training.py: write MASTER_ADDR': 1,
    'fastvideo/tests/training/lora/test_lora_training.py: write MASTER_PORT': 1,
    'fastvideo/tests/training/lora/test_lora_training.py: write WANDB_MODE': 1,
    'fastvideo/tests/training/self-forcing/test_self_forcing.py: write MASTER_ADDR': 1,
    'fastvideo/tests/training/self-forcing/test_self_forcing.py: write MASTER_PORT': 1,
    'fastvideo/tests/training/self-forcing/test_self_forcing.py: write WANDB_MODE': 1,
    'fastvideo/tests/transformers/test_cosmos.py: write MASTER_ADDR': 1,
    'fastvideo/tests/transformers/test_cosmos.py: write MASTER_PORT': 1,
    'fastvideo/tests/transformers/test_cosmos2_5.py: write MASTER_ADDR': 1,
    'fastvideo/tests/transformers/test_cosmos2_5.py: write MASTER_PORT': 1,
    'fastvideo/tests/transformers/test_flux.py: read FLUX_TRANSFORMER_PATH': 1,
    'fastvideo/tests/transformers/test_flux.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/transformers/test_flux.py: write MASTER_ADDR': 1,
    'fastvideo/tests/transformers/test_flux.py: write MASTER_PORT': 1,
    'fastvideo/tests/transformers/test_hunyuangamecraft.py: write DISABLE_SP': 1,
    'fastvideo/tests/transformers/test_hunyuangamecraft.py: write MASTER_ADDR': 1,
    'fastvideo/tests/transformers/test_hunyuangamecraft.py: write MASTER_PORT': 1,
    'fastvideo/tests/transformers/test_hunyuangamecraft.py: write TORCHDYNAMO_DISABLE': 1,
    'fastvideo/tests/transformers/test_hyworld.py: read FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/transformers/test_hyworld.py: write MASTER_ADDR': 1,
    'fastvideo/tests/transformers/test_hyworld.py: write MASTER_PORT': 1,
    'fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py: write FASTVIDEO_ATTENTION_BACKEND': 2,
    'fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py: write FASTVIDEO_FA4': 1,
    'fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py: write LOCAL_RANK': 2,
    'fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py: write MASTER_ADDR': 2,
    'fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py: write MASTER_PORT': 2,
    'fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py: write RANK': 2,
    'fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py: write WORLD_SIZE': 2,
    'fastvideo/tests/transformers/test_wanvideo.py: write MASTER_ADDR': 1,
    'fastvideo/tests/transformers/test_wanvideo.py: write MASTER_PORT': 1,
    'fastvideo/tests/vaes/test_hunyuan15_vae.py: write MASTER_ADDR': 1,
    'fastvideo/tests/vaes/test_hunyuan15_vae.py: write MASTER_PORT': 1,
    'fastvideo/tests/vaes/test_wan_vae.py: write MASTER_ADDR': 1,
    'fastvideo/tests/vaes/test_wan_vae.py: write MASTER_PORT': 1,
    'fastvideo/tests/worker/test_gpu_worker.py: write FASTVIDEO_NVTX_PROFILE': 2,
    'fastvideo/tests/worker/test_gpu_worker.py: write LOCAL_RANK': 1,
    'fastvideo/train/entrypoint/dcp_to_diffusers.py: write <dynamic>': 1,
    'fastvideo/train/entrypoint/train.py: write FASTVIDEO_ATTENTION_BACKEND': 2,
    'fastvideo/training/self_forcing_distillation_pipeline.py: read FASTVIDEO_FSDP2_AUTOWRAP': 1,
    'fastvideo/utils.py: read <dynamic>': 5,
    'fastvideo/utils.py: read FASTVIDEO_WORKER_MULTIPROC_METHOD': 1,
    'fastvideo/utils.py: write <dynamic>': 1,
    'fastvideo/utils.py: write FASTVIDEO_WORKER_MULTIPROC_METHOD': 1,
    'fastvideo/worker/gpu_worker.py: write LOCAL_RANK': 1,
    'fastvideo/worker/gpu_worker.py: write NCCL_ASYNC_ERROR_HANDLING': 1,
    'fastvideo/worker/gpu_worker.py: write RANK': 1,
    'fastvideo/worker/gpu_worker.py: write TORCH_NCCL_AVOID_RECORD_STREAMS': 1,
    'fastvideo/worker/gpu_worker.py: write WORLD_SIZE': 1,
    'fastvideo/worker/ray_distributed_executor.py: read <dynamic>': 2,
    'fastvideo/worker/ray_distributed_executor.py: read RAY_USAGE_STATS_ENABLED': 1,
    'fastvideo/worker/ray_distributed_executor.py: whole-environ': 1,
    'fastvideo/worker/ray_distributed_executor.py: write RAY_USAGE_STATS_ENABLED': 1,
    'fastvideo/worker/ray_env.py: import-time-read FASTVIDEO_CONFIG_ROOT': 1,
    'fastvideo/worker/ray_env.py: read <dynamic>': 1,
    'fastvideo/worker/ray_utils.py: whole-environ': 1,
    'fastvideo/worker/worker_base.py: read <dynamic>': 1,
    'fastvideo/worker/worker_base.py: write <dynamic>': 1,
}


def load_registry() -> ModuleType:
    """Load fastvideo/envs.py as a standalone module, without importing the fastvideo package."""
    spec = importlib.util.spec_from_file_location("_fastvideo_envs_registry", REGISTRY_PATH)
    assert spec is not None and spec.loader is not None
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    return registry


def is_allowlisted(name: str) -> bool:
    for pattern in EXTERNAL_ALLOWLIST:
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return True
        if name == pattern:
            return True
    return False


def _literal(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "<dynamic>"


class EnvAccessScanner:
    """Find environment accesses and registry-field uses in one module.

    ``violations`` holds (kind, name, line) tuples; ``registry_reads`` counts
    ``envs.X.get()`` and ``envs.X.is_set()`` calls by variable name.
    """

    def __init__(self, tree: ast.Module, registry_names: set[str]) -> None:
        self.registry_names = registry_names
        self.violations: list[tuple[str, str, int]] = []
        self.registry_reads: Counter[str] = Counter()
        self.parents: dict[ast.AST, ast.AST] = {}
        self._collect_aliases(tree)
        self._visit(tree, in_function=False)

    def _collect_aliases(self, tree: ast.Module) -> None:
        """Record the local names bound to os, os.environ, os.getenv-like functions, and fastvideo.envs."""
        self.os_names: set[str] = set()
        self.environ_names: set[str] = set()
        self.getenv_names: set[str] = set()
        self.setenv_names: set[str] = set()
        self.envs_module_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os" or alias.name.startswith("os."):
                        self.os_names.add(alias.asname or "os")
                    if alias.name == "fastvideo.envs" and alias.asname:
                        self.envs_module_names.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    local = alias.asname or alias.name
                    if node.module == "os":
                        if alias.name in ("environ", "environb"):
                            self.environ_names.add(local)
                        elif alias.name in ("getenv", "getenvb"):
                            self.getenv_names.add(local)
                        elif alias.name in ("putenv", "unsetenv"):
                            self.setenv_names.add(local)
                    elif node.module == "fastvideo" and alias.name == "envs":
                        self.envs_module_names.add(local)

    def _is_os_attr(self, node: ast.AST, attrs: tuple[str, ...]) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr in attrs and isinstance(node.value, ast.Name)
                and node.value.id in self.os_names)

    def _is_environ(self, node: ast.AST) -> bool:
        return self._is_os_attr(node, ("environ", "environb")) or (isinstance(node, ast.Name)
                                                                    and node.id in self.environ_names)

    def _is_envs_module(self, node: ast.AST) -> bool:
        return ((isinstance(node, ast.Name) and node.id in self.envs_module_names)
                or (isinstance(node, ast.Attribute) and node.attr == "envs"))

    def _visit(self, node: ast.AST, in_function: bool) -> None:
        """Walk the tree, tracking whether each node runs inside a function body."""
        self._check(node, in_function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Decorators, defaults, and annotations run when the function is defined.
            outside = [*getattr(node, "decorator_list", []), *node.args.defaults, *node.args.kw_defaults]
            for child in outside:
                if child is not None:
                    self.parents[child] = node
                    self._visit(child, in_function)
            body = node.body if isinstance(node.body, list) else [node.body]
            for child in body:
                self.parents[child] = node
                self._visit(child, True)
            return
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self._visit(child, in_function)

    def _check(self, node: ast.AST, in_function: bool) -> None:
        parent = self.parents.get(node)
        if self._is_environ(node):
            self._check_environ_use(node, parent)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Attribute) and node.attr in self.registry_names and self._is_envs_module(node.value):
            self._check_registry_use(node, parent, in_function)

    def _check_environ_use(self, node: ast.AST, parent: ast.AST | None) -> None:
        line = node.lineno  # type: ignore[attr-defined]
        if isinstance(parent, ast.Attribute) and isinstance(self.parents.get(parent), ast.Call):
            call = self.parents[parent]
            assert isinstance(call, ast.Call)
            if call.func is parent and parent.attr == "get":
                self.violations.append(("read", _literal(call.args[0] if call.args else None), line))
                return
            if call.func is parent and parent.attr in ("setdefault", "pop"):
                self.violations.append(("write", _literal(call.args[0] if call.args else None), line))
                return
        if isinstance(parent, ast.Subscript) and parent.value is node:
            kind = "read" if isinstance(parent.ctx, ast.Load) else "write"
            self.violations.append((kind, _literal(parent.slice), line))
            return
        if (isinstance(parent, ast.Compare) and len(parent.ops) == 1 and isinstance(parent.ops[0], (ast.In, ast.NotIn))
                and parent.comparators[0] is node):
            self.violations.append(("read", _literal(parent.left), line))
            return
        self.violations.append(("whole-environ", "", line))

    def _check_call(self, node: ast.Call) -> None:
        func = node.func
        name = _literal(node.args[0] if node.args else None)
        if self._is_os_attr(func, ("getenv", "getenvb")) or (isinstance(func, ast.Name)
                                                               and func.id in self.getenv_names):
            self.violations.append(("read", name, node.lineno))
        elif (self._is_os_attr(func, ("putenv", "unsetenv"))
              or (isinstance(func, ast.Name) and func.id in self.setenv_names)
              or (isinstance(func, ast.Attribute) and func.attr in ("setenv", "delenv"))):
            self.violations.append(("write", name, node.lineno))

    def _check_registry_use(self, node: ast.Attribute, parent: ast.AST | None, in_function: bool) -> None:
        if not isinstance(parent, ast.Attribute) or parent.attr not in REGISTRY_METHODS:
            self.violations.append(("bare-field", node.attr, node.lineno))
            return
        if parent.attr in REGISTRY_READ_METHODS:
            self.registry_reads[node.attr] += 1
            if not in_function:
                self.violations.append(("import-time-read", node.attr, node.lineno))


def scanned_files() -> list[Path]:
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py")
                  if path != REGISTRY_PATH and not any(excluded in path.parents for excluded in EXCLUDED_DIRS))


def collect_violations(registry: ModuleType) -> tuple[dict[str, list[int]], Counter[str]]:
    """Return every violation key with its line numbers, plus registry read counts."""
    registry_names = set(registry.environment_variables)
    found: dict[str, list[int]] = {}
    reads: Counter[str] = Counter()
    for path in scanned_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        scanner = EnvAccessScanner(ast.parse(path.read_text(encoding="utf-8"), filename=relative), registry_names)
        reads.update(scanner.registry_reads)
        for kind, name, line in scanner.violations:
            if kind == "read" and is_allowlisted(name):
                continue
            key = f"{relative}: {kind} {name}".rstrip()
            found.setdefault(key, []).append(line)

    registry_path = REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
    for name in registry.environment_variables:
        if not re.fullmatch(r"FASTVIDEO_[A-Z0-9_]+", name):
            found.setdefault(f"{registry_path}: prefix {name}", []).append(0)
        if reads[name] == 0:
            found.setdefault(f"{registry_path}: unread {name}", []).append(0)
    return found, reads


def test_env_access_follows_policy():
    found, _ = collect_violations(load_registry())
    counts = Counter({key: len(lines) for key, lines in found.items()})
    known = Counter(KNOWN_VIOLATIONS)

    new = counts - known
    fixed = known - counts
    messages = []
    if new:
        lines = [f"  {key}  (lines {found[key]})" for key in sorted(new)]
        messages.append(f"New environment-variable policy violations. See {POLICY_DOC} for the rule and the fix:\n" +
                        "\n".join(lines))
    if fixed:
        lines = [f"  {key}" for key in sorted(fixed)]
        messages.append("These KNOWN_VIOLATIONS entries are fixed; delete them from "
                        "fastvideo/tests/contract/test_env_policy.py:\n" + "\n".join(lines))
    assert not messages, "\n\n".join(messages)


def test_registry_entries_have_category_and_description():
    registry = load_registry()
    problems = []
    for name, field in registry.environment_variables.items():
        if field.category not in registry.CATEGORIES:
            problems.append(f"{name}: category {field.category!r} is not in envs.CATEGORIES")
        if not field.doc.strip():
            problems.append(f"{name}: empty description")
    assert not problems, f"Registry entries break {POLICY_DOC}:\n" + "\n".join(problems)


def _render_default(field) -> str:
    if callable(field.default):
        return "computed"
    if field.default is None:
        return "unset"
    return f"`{field.format(field.default)}`" if field.format(field.default) else '`""`'


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("*", "\\*").replace("<", "&lt;").replace(">", "&gt;")


def render_env_table(registry: ModuleType) -> str:
    """Render the registry as an aligned Markdown table, in declaration order."""
    rows = [["Variable", "Type", "Default", "Category", "Description"]]
    for name, field in registry.environment_variables.items():
        rows.append([f"`{name}`", field.type_name, _render_default(field), field.category, _escape(field.doc)])
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    lines = [render_row(rows[0]), render_row(["-" * width for width in widths])]
    lines.extend(render_row(row) for row in rows[1:])
    return "\n".join(lines)


def _split_doc(text: str) -> tuple[str, str, str]:
    """Split the policy doc into the text before the generated table, the table, and the text after it."""
    before, found_begin, rest = text.partition(DOC_TABLE_BEGIN)
    table, found_end, after = rest.partition(DOC_TABLE_END)
    assert found_begin and found_end, f"{POLICY_DOC} must contain the generated-table markers"
    return before + found_begin + "\n", table.strip("\n"), "\n" + found_end + after


def test_env_doc_table_matches_registry():
    _, table, _ = _split_doc(DOC_PATH.read_text(encoding="utf-8"))
    assert table == render_env_table(load_registry()), (
        f"The table in {POLICY_DOC} is out of date. Run `python fastvideo/tests/contract/test_env_policy.py`.")


if __name__ == "__main__":
    head, _, tail = _split_doc(DOC_PATH.read_text(encoding="utf-8"))
    DOC_PATH.write_text(head + render_env_table(load_registry()) + tail, encoding="utf-8")
    print(f"Updated {POLICY_DOC}")
