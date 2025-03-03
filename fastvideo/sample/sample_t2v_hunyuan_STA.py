import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange

from fastvideo.models.hunyuan.inference import HunyuanVideoSampler
from fastvideo.models.hunyuan.modules.modulate_layers import modulate
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    mask_strategy=None,
    output_features=False,
    output_features_stride=8,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
    guidance=None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if guidance is None:
        guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)

    img = x = hidden_states
    text_mask = encoder_attention_mask
    t = timestep
    txt = encoder_hidden_states[:, 1:]
    text_states_2 = encoder_hidden_states[:, 0, :self.config.text_states_dim_2]
    _, _, ot, oh, ow = x.shape  # codespell:ignore
    tt, th, tw = (
        ot // self.patch_size[0],  # codespell:ignore
        oh // self.patch_size[1],  # codespell:ignore
        ow // self.patch_size[2],  # codespell:ignore
    )
    original_tt = nccl_info.sp_size * tt
    freqs_cos, freqs_sin = self.get_rotary_pos_embed((original_tt, th, tw))
    # Prepare modulation vectors.
    vec = self.time_in(t)

    # text modulation
    vec = vec + self.vector_in(text_states_2)

    # guidance modulation
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")

        # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
        vec = vec + self.guidance_in(guidance)

    # Embed image and text.
    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    if self.enable_teacache:
        inp = img.clone()
        vec_ = vec.clone()
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
        normed_inp = self.double_blocks[0].img_norm1(inp)
        modulated_inp = modulate(normed_inp, shift=img_mod1_shift, scale=img_mod1_scale)
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.previous_modulated_input).abs().mean() /
                 self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    if self.enable_teacache:
        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            # --------------------- Pass through DiT blocks ------------------------
            for index, block in enumerate(self.double_blocks):
                double_block_args = [img, txt, vec, freqs_cis, text_mask, mask_strategy[index]]
                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if output_features:
                features_list = []
            if len(self.single_blocks) > 0:
                for index, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        (freqs_cos, freqs_sin),
                        text_mask,
                        mask_strategy[index + len(self.double_blocks)],
                    ]
                    x = block(*single_block_args)
                    if output_features and _ % output_features_stride == 0:
                        features_list.append(x[:, :img_seq_len, ...])

            img = x[:, :img_seq_len, ...]
            self.previous_residual = img - ori_img
    else:
        # --------------------- Pass through DiT blocks ------------------------
        for index, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis, text_mask, mask_strategy[index]]
            img, txt = block(*double_block_args)
        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if output_features:
            features_list = []
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    (freqs_cos, freqs_sin),
                    text_mask,
                    mask_strategy[index + len(self.double_blocks)],
                ]
                x = block(*single_block_args)
                if output_features and _ % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])

        img = x[:, :img_seq_len, ...]

    # ---------------------------- Final layer ------------------------------
    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    img = self.unpatchify(img, tt, th, tw)
    assert not return_dict, "return_dict is not supported."
    if output_features:
        features_list = torch.stack(features_list, dim=0)
    else:
        features_list = None
    return (img, features_list)


def initialize_distributed():
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)


def main(args):
    initialize_distributed()
    print(nccl_info.sp_size)

    print(args)
    models_root_path = Path(args.model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    # Create save folder to save the samples
    save_path = args.output_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # teacache
    hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = args.enable_teacache
    hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.num_inference_steps
    hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh = args.rel_l1_thresh  # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
    hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_modulated_input = None
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = teacache_forward

    with open(args.mask_strategy_file_path, 'r') as f:
        mask_strategy = json.load(f)
    if args.prompt.endswith('.txt'):
        with open(args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [args.prompt]

    for prompt in prompts:
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            height=args.height,
            width=args.width,
            video_length=args.num_frames,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            mask_strategy=mask_strategy,
        )
        videos = rearrange(outputs["samples"], "b c t h w -> t b c h w")
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            outputs.append((x * 255).numpy().astype(np.uint8))
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        imageio.mimsave(os.path.join(args.output_path, f"{prompt[:100]}.mp4"), outputs, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--fps", type=int, default=24)

    # Additional parameters
    parser.add_argument(
        "--sliding_block_size",
        type=str,
        default="8,6,10",
        help="Sliding block size for sliding block attention.",
    )
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument("--neg_prompt", type=str, default=None, help="Negative prompt for sampling.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument("--flow_shift", type=int, default=7, help="Flow shift parameter.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default="data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver", type=str, default="euler", help="Solver for flow matching.")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help=
        "Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)
    parser.add_argument("--vae-sp", action="store_true", default=False)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template", type=str, default="dit-llm-encode")
    parser.add_argument("--prompt-template-video", type=str, default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)
    parser.add_argument("--skip_time_steps", type=int, default=10)
    parser.add_argument(
        "--mask_strategy_selected",
        type=lambda x: [int(i) for i in x.strip('[]').split(',')],  # Convert string to list of integers
        default=[1, 2, 6],  # Now can be directly set as a list
        help="order of candidates")
    parser.add_argument(
        "--rel_l1_thresh",
        type=float,
        default=0.15,
        help="0.1 for 1.6x speedup, 0.15 for 2.1x speedup",
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Use teacache for speeding up inference",
    )
    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Use torch.compile for speeding up STA inference without teacache",
    )
    parser.add_argument("--mask_strategy_file_path", type=str, default="assets/mask_strategy.json")
    args = parser.parse_args()
    # process for vae sequence parallel
    if args.vae_sp and not args.vae_tiling:
        raise ValueError("Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True.")
    if args.enable_teacache and args.enable_torch_compile:
        raise ValueError(
            "--enable_teacache and --enable_torch_compile cannot be used simultaneously. Please enable only one of these options."
        )
    main(args)
