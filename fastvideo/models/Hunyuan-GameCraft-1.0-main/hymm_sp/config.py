import argparse
from hymm_sp.constants import *
import re
import collections.abc

def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")

def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="Hunyuan Multimodal training/inference script")
    parser = add_extra_args(parser)
    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)
    return args

def add_extra_args(parser: argparse.ArgumentParser):
    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_test_args(parser)
    return parser

def add_test_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Test")
    
    group.add_argument("--image-start", action="store_true", help="Use one image from video for training")
    group.add_argument("--use-csv-pose", action="store_true", help="Use one image from video for training")
    group.add_argument("--add-button", action="store_true", help="Use one image from video for training")
    group.add_argument("--action-list", type=str, nargs='+', default=None, help="CSV file for evaluation.")
    group.add_argument("--action-speed-list", type=float, nargs='+', default=None, help="CSV file for evaluation.")
    group.add_argument("--pose", type=str, default=None, help="CSV file for evaluation.")
    
    return parser
    

def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Network")
    group.add_argument("--model", type=str, default="HYVideo-T/2",
                       help="Model architecture to use. It it also used to determine the experiment directory.")
    group.add_argument("--latent-channels", type=str, default=None,
                       help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
                            "it still needs to match the latent channels of the VAE model.")
    group.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")
    return parser

def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Extra Models (VAE, Text Encoder, Tokenizer)")

    # VAE
    group.add_argument("--vae", type=str, default="884-16c-hy0801",  help="Name of the VAE model.")
    group.add_argument("--vae-precision", type=str, default="fp16", 
                       help="Precision mode for the VAE model.")
    group.add_argument("--vae-tiling", action="store_true", default=True, help="Enable tiling for the VAE model.")
    group.add_argument("--text-encoder", type=str, default="llava-llama-3-8b", choices=list(TEXT_ENCODER_PATH),
                       help="Name of the text encoder model.")
    group.add_argument("--text-encoder-precision", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the text encoder model.")
    group.add_argument("--text-states-dim", type=int, default=4096, help="Dimension of the text encoder hidden states.")
    group.add_argument("--text-len", type=int, default=256, help="Maximum length of the text input.")
    group.add_argument("--tokenizer", type=str, default="llava-llama-3-8b", choices=list(TOKENIZER_PATH),
                       help="Name of the tokenizer model.")
    group.add_argument("--text-encoder-infer-mode", type=str, default="encoder", choices=["encoder", "decoder"],
                       help="Inference mode for the text encoder model. It should match the text encoder type. T5 and "
                            "CLIP can only work in 'encoder' mode, while Llava/GLM can work in both modes.")
    
    group.add_argument("--prompt-template-video", type=str, default='li-dit-encode-video', choices=PROMPT_TEMPLATE,
                       help="Video prompt template for the decoder-only text encoder model.")
    group.add_argument("--hidden-state-skip-layer", type=int, default=2,
                       help="Skip layer for hidden states.")
    group.add_argument("--apply-final-norm", action="store_true",
                       help="Apply final normalization to the used text encoder hidden states.")

    # - CLIP
    group.add_argument("--text-encoder-2", type=str, default='clipL', choices=list(TEXT_ENCODER_PATH),
                       help="Name of the second text encoder model.")
    group.add_argument("--text-encoder-precision-2", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the second text encoder model.")
    group.add_argument("--text-states-dim-2", type=int, default=768,
                       help="Dimension of the second text encoder hidden states.")
    group.add_argument("--tokenizer-2", type=str, default='clipL', choices=list(TOKENIZER_PATH),
                       help="Name of the second tokenizer model.")
    group.add_argument("--text-len-2", type=int, default=77, help="Maximum length of the second text input.")
    group.set_defaults(use_attention_mask=True)
    group.add_argument("--text-projection", type=str, default="single_refiner", choices=TEXT_PROJECTION,
                       help="A projection layer for bridging the text encoder hidden states and the diffusion model "
                            "conditions.")
    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule")
    group.add_argument("--flow-shift-eval-video", type=float, default=None, help="Shift factor for flow matching schedulers when using video data.")
    group.add_argument("--flow-reverse", action="store_true", default=True, help="If reverse, learning/sampling from t=1 -> t=0.")
    group.add_argument("--flow-solver", type=str, default="euler", help="Solver for flow matching.")
    group.add_argument("--use-linear-quadratic-schedule", action="store_true", help="Use linear quadratic schedule for flow matching."
                                                    "Follow MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)")
    group.add_argument("--linear-schedule-end", type=int, default=25, help="End step for linear quadratic schedule for flow matching.")
    return parser

def add_evaluation_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Validation Loss Evaluation")
    parser.add_argument("--precision", type=str, default="bf16", choices=PRECISIONS,
                    help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.")
    parser.add_argument("--reproduce", action="store_true",
                       help="Enable reproducibility by setting random seeds and deterministic algorithms.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--load-key", type=str, default="module", choices=["module", "ema"],
                       help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.")
    parser.add_argument("--cpu-offload", action="store_true", help="Use CPU offload for the model load.")
    group.add_argument( "--use-fp8", action="store_true", help="Enable use fp8 for inference acceleration.")
    group.add_argument("--video-size", type=int, nargs='+', default=512,
                        help="Video size for training. If a single value is provided, it will be used for both width "
                            "and height. If two values are provided, they will be used for width and height "
                            "respectively.")
    group.add_argument("--sample-n-frames", type=int, default=33,
                       help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1")
    group.add_argument("--infer-steps", type=int, default=100, help="Number of denoising steps for inference.")
    group.add_argument("--val-disable-autocast", action="store_true",
                       help="Disable autocast for denoising loop and vae decoding in pipeline sampling.")
    group.add_argument("--num-images", type=int, default=1, help="Number of images to generate for each prompt.")
    group.add_argument("--seed", type=int, default=1024, help="Seed for evaluation.")
    group.add_argument("--save-path-suffix", type=str, default="", help="Suffix for the directory of saved samples.")
    group.add_argument("--prompt", type=str, default='', help="Main prompt")
    group.add_argument("--pos-prompt", type=str, default='', help="Prompt for sampling during evaluation.")
    group.add_argument("--neg-prompt", type=str, default='', help="Negative prompt for sampling during evaluation.")
    group.add_argument("--add-pos-prompt", type=str, default='', help="Addition prompt for sampling during evaluation.")
    group.add_argument("--add-neg-prompt", type=str, default='', help="Addition negative prompt for sampling during evaluation.")
    group.add_argument("--pad-face-size", type=float, default=0.7, help="Pad bbox for face align.")
    group.add_argument("--image-path", type=str, default="",  help="")
    group.add_argument("--save-path", type=str, default=None, help="Path to save the generated samples.")
    group.add_argument("--input", type=str, default=None, help="test data.")
    group.add_argument("--item-name", type=str, default=None, help="")
    group.add_argument("--cfg-scale", type=float, default=7.5, help="Classifier free guidance scale.")
    group.add_argument("--ip-cfg-scale", type=float, default=0, help="Classifier free guidance scale.")
    group.add_argument("--use-deepcache", type=int, default=1)
    group.add_argument("--use-sage", action="store_true", help="Use sage attention for speed up.")
    
    return parser

def sanity_check_args(args):
    # VAE channels
    vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
    if not re.match(vae_pattern, args.vae):
        raise ValueError(
            f"Invalid VAE model: {args.vae}. Must be in the format of '{vae_pattern}'."
        )
    vae_channels = int(args.vae.split("-")[1][:-1])
    if args.latent_channels is None:
        args.latent_channels = vae_channels
    if vae_channels != args.latent_channels:
        raise ValueError(
            f"Latent channels ({args.latent_channels}) must match the VAE channels ({vae_channels})."
        )
    return args
