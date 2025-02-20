import argparse
import json
import os
import types
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange, repeat

from fastvideo.models.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from fastvideo.models.stepvideo.diffusion.video_pipeline import StepVideoPipeline
from fastvideo.models.stepvideo.modules.model import StepVideoModel
from fastvideo.models.stepvideo.utils import setup_seed
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info


def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    main_print(f"world_size: {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="StepVideo inference script")

    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)

    args = parser.parse_args(namespace=namespace)

    return args


def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Extra models args, including vae, text encoders and tokenizers)")

    group.add_argument(
        "--vae_url",
        type=str,
        default='127.0.0.1',
        help="vae url.",
    )
    group.add_argument(
        "--caption_url",
        type=str,
        default='127.0.0.1',
        help="caption url.",
    )

    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule args")

    # Flow Matching
    group.add_argument(
        "--time_shift",
        type=float,
        default=13,
        help="Shift factor for flow matching schedulers.",
    )
    group.add_argument(
        "--flow_reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    group.add_argument(
        "--flow_solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )

    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")

    # ======================== Model loads ========================
    group.add_argument(
        "--model_dir",
        type=str,
        default="./ckpts",
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--model_resolution",
        type=str,
        default="540p",
        choices=["540p"],
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )

    # ======================== Inference general setting ========================
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    group.add_argument(
        "--infer_steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    group.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save the generated samples.",
    )
    group.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Suffix for the names of saved samples.",
    )
    group.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate for each prompt.",
    )
    # ---sample size---
    group.add_argument(
        "--num_frames",
        type=int,
        default=204,
        help="How many frames to sample from a video. ",
    )
    group.add_argument(
        "--height",
        type=int,
        default=768,
        help="The height of video sample",
    )
    group.add_argument(
        "--width",
        type=int,
        default=768,
        help="The width of video sample",
    )
    # --- prompt ---
    group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for sampling during evaluation.",
    )
    group.add_argument("--seed", type=int, default=1234, help="Seed for evaluation.")

    # Classifier-Free Guidance
    group.add_argument("--pos_magic",
                       type=str,
                       default="超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。",
                       help="Positive magic prompt for sampling.")
    group.add_argument("--neg_magic",
                       type=str,
                       default="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。",
                       help="Negative magic prompt for sampling.")
    group.add_argument("--cfg_scale", type=float, default=9.0, help="Classifier free guidance scale.")
    group.add_argument("--mask_search_files_path", type=str, default="assets/mask_strategy.json")
    group.add_argument("--mask_strategy_file_path", type=str, default="assets/mask_strategy_stepvideo.json")
    group.add_argument("--skip_time_steps", type=int, default=10)
    group.add_argument(
        "--mask_strategy_selected",
        type=lambda x: [int(i) for i in x.strip('[]').split(',')],  # Convert string to list of integers
        default=[1, 2, 6],  # Now can be directly set as a list
        help="order of candidates")
    parser.add_argument(
        "--rel_l1_thresh",
        type=float,
        default=0,
        help="0.22 for 1.67x speedup, 0.23 for 2.1x speedup",
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Use teacache for speeding up inference",
    )
    return parser


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_hidden_states_2: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Dict[str, torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    fps: torch.Tensor = None,
    return_dict: bool = True,
    mask_strategy=None,
):
    assert hidden_states.ndim == 5
    "hidden_states's shape should be (bsz, f, ch, h ,w)"

    bsz, frame, _, height, width = hidden_states.shape
    height, width = height // self.patch_size, width // self.patch_size

    hidden_states = self.patchfy(hidden_states)
    len_frame = hidden_states.shape[1]

    if self.use_additional_conditions:
        added_cond_kwargs = {
            "resolution": torch.tensor([(height, width)] * bsz, device=hidden_states.device, dtype=hidden_states.dtype),
            "nframe": torch.tensor([frame] * bsz, device=hidden_states.device, dtype=hidden_states.dtype),
            "fps": fps
        }
    else:
        added_cond_kwargs = {}

    timestep, embedded_timestep = self.adaln_single(timestep, added_cond_kwargs=added_cond_kwargs)

    encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))

    if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
        clip_embedding = self.clip_projection(encoder_hidden_states_2)
        encoder_hidden_states = torch.cat([clip_embedding, encoder_hidden_states], dim=1)

    hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()

    embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()

    shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)

    encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask,
                                                              encoder_hidden_states,
                                                              q_seqlen=frame * len_frame)

    if self.enable_teacache:
        hidden_states_ = hidden_states.clone()

        normed_hidden_states = self.transformer_blocks[0].norm1(hidden_states_)
        normed_hidden_states = rearrange(normed_hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)

        modulated_inp = normed_hidden_states * (1 + scale) + shift

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [6.74352814e+03, -2.22814115e+03, 2.55029094e+02, -1.12338285e+01, 2.84921593e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.previous_modulated_input).abs().mean() /
                 self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                # print(f"accumulated_rel_l1_distance: {self.accumulated_rel_l1_distance}")
                should_calc = False
            else:
                # print(f"accumulated_rel_l1_distance: {self.accumulated_rel_l1_distance}")
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    if self.enable_teacache:
        if not should_calc:
            # print(f"skip step {self.cnt}")
            hidden_states += self.previous_residual
        else:
            # print(f"calc step {self.cnt}")
            ori_hidden_states = hidden_states.clone()
            hidden_states = self.block_forward(hidden_states,
                                               encoder_hidden_states,
                                               timestep=timestep,
                                               rope_positions=[frame, height, width],
                                               attn_mask=attn_mask,
                                               parallel=self.parallel,
                                               mask_strategy=mask_strategy)
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        # --------------------- Pass through DiT blocks ------------------------
        hidden_states = self.block_forward(hidden_states,
                                           encoder_hidden_states,
                                           timestep=timestep,
                                           rope_positions=[frame, height, width],
                                           attn_mask=attn_mask,
                                           parallel=self.parallel,
                                           mask_strategy=mask_strategy)

    # ---------------------------- Final layer ------------------------------
    hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)

    hidden_states = self.norm_out(hidden_states)
    # Modulation
    hidden_states = hidden_states * (1 + scale) + shift
    hidden_states = self.proj_out(hidden_states)

    # unpatchify
    hidden_states = hidden_states.reshape(shape=(-1, height, width, self.patch_size, self.patch_size,
                                                 self.out_channels))

    hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
    output = hidden_states.reshape(shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size))

    output = rearrange(output, '(b f) c h w -> b f c h w', f=frame)
    if return_dict:
        return {'x': output}
    return output


if __name__ == "__main__":
    args = parse_args()
    initialize_distributed()
    main_print(f"sequence parallel size: {nccl_info.sp_size}")
    device = torch.cuda.current_device()

    setup_seed(args.seed)
    main_print("Loading model, this might take a while...")
    transformer = StepVideoModel.from_pretrained(os.path.join(args.model_dir, "transformer"),
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device)
    if args.enable_teacache:
        transformer.forward = types.MethodType(teacache_forward, transformer)
    scheduler = FlowMatchDiscreteScheduler()
    pipeline = StepVideoPipeline(transformer, scheduler, save_path=args.save_path)
    pipeline.setup_api(
        vae_url=args.vae_url,
        caption_url=args.caption_url,
    )

    # TeaCache
    pipeline.transformer.__class__.enable_teacache = True
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.num_steps = args.infer_steps
    pipeline.transformer.__class__.rel_l1_thresh = args.rel_l1_thresh  # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
    pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    pipeline.transformer.__class__.previous_modulated_input = None
    pipeline.transformer.__class__.previous_residual = None

    with open(args.mask_strategy_file_path, 'r') as f:
        mask_strategy = json.load(f)

    if args.prompt.endswith('.txt'):
        with open(args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [args.prompt]
    for prompt in prompts:
        main_print(f"Generating video for prompt: {prompt}")
        videos = pipeline(prompt=prompt,
                          num_frames=args.num_frames,
                          height=args.height,
                          width=args.width,
                          num_inference_steps=args.infer_steps,
                          guidance_scale=args.cfg_scale,
                          time_shift=args.time_shift,
                          pos_magic=args.pos_magic,
                          neg_magic=args.neg_magic,
                          output_file_name=prompt[:150],
                          mask_strategy=mask_strategy)

    dist.destroy_process_group()
