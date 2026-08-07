# SPDX-License-Identifier: Apache-2.0
"""Request model for preloading/unloading a resident generator.

Field names and defaults mirror the engine subset of ``CreateJobRequest`` so
the UI can send exactly the values it would put on a job — guaranteeing the
job's generator lookup hits this cache entry.
"""

from pydantic import BaseModel


class GeneratorRequest(BaseModel):
    model_id: str
    workload_type: str = "t2v"
    num_gpus: int = 1
    dit_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    image_encoder_cpu_offload: bool = False
    use_fsdp_inference: bool = False
    enable_torch_compile: bool = False
    vsa_sparsity: float = 0.0
    tp_size: int = -1
    sp_size: int = -1
