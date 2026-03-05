# SPDX-License-Identifier: Apache-2.0
"""Request model for creating a job."""

from pydantic import BaseModel


class CreateJobRequest(BaseModel):
    model_id: str
    prompt: str
    workload_type: str = "t2v"
    job_type: str = "inference"
    image_path: str = ""
    data_path: str = ""
    max_train_steps: int = 1000
    train_batch_size: int = 1
    learning_rate: float = 5e-5
    num_latent_t: int = 20
    validation_dataset_file: str = ""
    lora_rank: int = 32
    negative_prompt: str = ""
    num_inference_steps: int = 50
    num_frames: int = 81
    height: int = 480
    width: int = 832
    guidance_scale: float = 5.0
    guidance_rescale: float = 0.0
    fps: int = 24
    seed: int = 1024
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
