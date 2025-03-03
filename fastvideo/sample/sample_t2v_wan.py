# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import os
import sys
import warnings
import types

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from PIL import Image

from fastvideo.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
# from fastvideo.models.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from fastvideo.models.wan.utils.utils import cache_video, cache_image, str2bool
from fastvideo.models.wan import WanT2V

from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
from fastvideo.utils.logging_ import main_print
from fastvideo.models.stepvideo.utils import setup_seed

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}

def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)
    return local_rank, world_size

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    # parser.add_argument(
    #     "--ulysses_size",
    #     type=int,
    #     default=1,
    #     help="The size of the ulysses parallelism in DiT.")
    # parser.add_argument(
    #     "--ring_size",
    #     type=int,
    #     default=1,
    #     help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="en",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        default=False,
        help="Whether to use teacache for inference."
    )
    parser.add_argument(
        "--rel_l1_thresh",
        type=float,
        default=0.0,
        help="Relative L1 threshold for teacache."
    )

    args = parser.parse_args()

    _validate_args(args)

    return args

def teacache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """

    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    if self.enable_teacache:
        e_teacache = e0.clone()
        with FSDP.summon_full_params(self.blocks[0]):
            with amp.autocast(dtype=torch.float32):
                e_teacache = (self.blocks[0].modulation + e_teacache).chunk(6, dim=1)
        assert e_teacache[0].dtype == torch.float32

        x_ = x.clone()
        norm_ = torch.nn.LayerNorm(self.blocks[0].dim, elementwise_affine=False, eps=self.blocks[0].eps)
        modulated_inp = norm_(x_).float() * (1 + e_teacache[1]) + e_teacache[0]
        if self.cnt % 2 == 0:
            self.is_even = True # even->condition odd->uncondition
            if self.cnt == 0 or self.cnt == self.num_steps - 2:
                should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0  
            else: 
                coefficients = [2.02286913e+04, -5.05103368e+03, 4.52376770e+02, -1.62713523e+01, 2.42891155e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_modulated_input_even).abs().mean() / self.previous_modulated_input_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.rel_l1_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_modulated_input_even = modulated_inp.clone()
            self.cnt += 1
        else:
            self.is_even = False
            if self.cnt == 1 or self.cnt == self.num_steps - 1:
                should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0  
            else: 
                coefficients = [2.02286913e+04, -5.05103368e+03, 4.52376770e+02, -1.62713523e+01, 2.42891155e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_modulated_input_odd).abs().mean() / self.previous_modulated_input_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.rel_l1_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_modulated_input_odd = modulated_inp.clone()
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        parallel=self.parallel)
    
    if self.enable_teacache:
        if self.is_even:
            if not should_calc_even:
                print(t)
                x += self.previous_residual_even
            else:
                ori_hidden_states = x.clone()
                x = self.block_forward(x, **kwargs)
                self.previous_residual_even = x - ori_hidden_states
        else:
            if not should_calc_odd:
                print(t)
                x += self.previous_residual_odd
            else:
                ori_hidden_states = x.clone()
                x = self.block_forward(x, **kwargs)
                self.previous_residual_odd = x - ori_hidden_states
    else:
        # --------------------- Pass through DiT blocks ------------------------
        x = self.block_forward(x, **kwargs)
        
    # head
    x, _ = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]

def generate(args):
    rank, world_size = initialize_distributed()
    device = rank
    main_print(f"sequence parallel size: {nccl_info.sp_size}")

    assert "t2v" in args.task or "t2i" in args.task, f"Unsupport task: {args.task}"

    setup_seed(args.base_seed)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        main_print(
            f"offload_model is not specified, set to {args.offload_model}.")

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl=False,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]

    main_print(f"Generation job args: {args}")
    main_print(f"Generation model config: {cfg}")

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    main_print(f"Input prompt: {args.prompt}")
    if args.use_prompt_extend:
        main_print("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                main_print(
                    f"Extending prompt failed: {prompt_output.message}")
                main_print("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        main_print(f"Extended prompt: {args.prompt}")

    main_print("Creating WanT2V pipeline.")
    wan_t2v = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(world_size > 1),
        t5_cpu=args.t5_cpu,
        enable_teacache=args.enable_teacache
    )

    if args.enable_teacache:
        wan_t2v.transformer.num_steps = args.sample_steps * 2
        wan_t2v.transformer.rel_l1_thresh = args.rel_l1_thresh

    main_print(
        f"Generating {'image' if 't2i' in args.task else 'video'} ...")
    video = wan_t2v.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{world_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            main_print(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            main_print(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    main_print("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)