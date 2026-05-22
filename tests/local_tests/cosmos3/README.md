# Cosmos3 local parity workspace

## Overview

Cosmos3 Phase-1 prep tracks a future FastVideo port from the vllm-omni Cosmos3 integration in PR #3454. The current reference supports a single `Cosmos3OmniDiffusersPipeline` for text-to-video (T2V), image-to-video (I2V), and text-to-image (T2I). Follow-up Cosmos3 capabilities mentioned in the PR body, such as sound generation and action-generation modes, are out of scope for the initial FastVideo port.

## Reference code

- Reference checkout: `/home/william5lin/cosmos3-reference`
- Source PR: <https://github.com/vllm-project/vllm-omni/pull/3454>
- Pinned HEAD: `8536f5b1421f78c7df06af6d96fa195c1ceb6384`
- Key files:
  - `vllm_omni/deploy/cosmos3.yaml`
  - `vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`
  - `vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py`
  - `vllm_omni/diffusion/models/cosmos3/guardrails.py`
  - `tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py`
  - `tests/diffusion/models/cosmos3/test_cosmos3_transformer.py`

## Weight status

PENDING. Do not download weights during Phase 1.

- Candidate serving/model id from the PR body: `nvidia/Cosmos3-Nano`
- Hugging Face API status on 2026-05-22: `401` for `https://huggingface.co/api/models/nvidia/Cosmos3-Nano`; API body reports `Invalid username or password.`
- Hugging Face author search `author=nvidia&search=Cosmos3`: empty list (`[]`).
- NGC API URL requested by handoff returned a Next.js 404 HTML page, not model metadata.

## Parity-test placeholder

Phase 2 should add local parity tests here after the official weights become accessible and FastVideo component prototypes exist. Suggested first targets:

1. Transformer state-dict key/shape inventory versus `Cosmos3VFMTransformer`.
2. Scheduler/default-parameter parity for T2V, I2V, and T2I request modes.
3. Prompt metadata-template parity for duration/resolution and image-vs-video modalities.

## SSIM placeholder

No SSIM references seeded yet. Add SSIM coverage only after a FastVideo inference path can load resolved Cosmos3 weights and generate stable T2V/T2I/I2V outputs.
