import torch
from pathlib import Path
from loguru import logger
from hymm_sp.constants import PROMPT_TEMPLATE, PRECISION_TO_TYPE
from hymm_sp.vae import load_vae
from hymm_sp.modules import load_model
from hymm_sp.text_encoder import TextEncoder
import torch.distributed
from hymm_sp.modules.parallel_states import (
    initialize_sequence_parallel_state,
    get_sequence_parallel_state,
    nccl_info,
)
from hymm_sp.modules.fp8_optimization import convert_fp8_linear


class Inference(object):
    def __init__(self, 
                 args,
                 vae, 
                 vae_kwargs, 
                 text_encoder, 
                 model, 
                 text_encoder_2=None, 
                 pipeline=None, 
                 cpu_offload=False,
                 device=None, 
                 logger=None):
        self.vae = vae
        self.vae_kwargs = vae_kwargs
        
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        self.model = model
        self.pipeline = pipeline
        self.cpu_offload = cpu_offload
        
        self.args = args
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        if nccl_info.sp_size > 1:
            self.device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        
        self.logger = logger

    @classmethod
    def from_pretrained(cls, 
                        pretrained_model_path,
                        args,
                        device=None,
                        **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, 
            including t2v, text encoder and vae checkpoints.
            device (int): The device for inference. Default is 0.
            logger (logging.Logger): The logger for the inference pipeline. Default is None.
        """
        # ========================================================================
        logger.info(f"Got text-to-video model root path: {pretrained_model_path}")
        
        # ======================== Get the args path =============================
        
        # Set device and disable gradient
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        logger.info("Building model...")
        factor_kwargs = {'device': 'cpu' if args.cpu_offload else device, 'dtype': PRECISION_TO_TYPE[args.precision]}
        in_channels = args.latent_channels
        out_channels = args.latent_channels
        print("="*25, f"build model", "="*25)
        model = load_model(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs
        )
        if args.cpu_offload:
            print(f'='*20, f'load transformer to cpu')
            model = model.to('cpu')
            torch.cuda.empty_cache()
        else:
            model = model.to(device)
        model = Inference.load_state_dict(args, model, pretrained_model_path)
        model.eval()
        
        if args.use_fp8:
            convert_fp8_linear(model)
        
        # ============================= Build extra models ========================
        # VAE
        print("="*25, f"load vae", "="*25)
        vae, _, s_ratio, t_ratio = load_vae(args.vae, 
                                            args.vae_precision, 
                                            logger=logger, 
                                            device='cpu' if args.cpu_offload else device)
        vae_kwargs = {'s_ratio': s_ratio, 't_ratio': t_ratio}
        
        # Parallel VAE
        device_vaes = []
        device_vaes.append(vae)
        if nccl_info.sp_size > 1 and nccl_info.rank_within_group == 0:
            for i in range(1, nccl_info.sp_size):
                cur_device = torch.device(f"cuda:{i}")
                # print("!!!!!!!!!! Load vae for ", cur_device)
                device_vae, _, _, _ = load_vae(args.vae, 
                                               args.vae_precision, 
                                               logger=logger, 
                                               device='cpu' if args.cpu_offload else cur_device)
                device_vaes.append(device_vae)
            vae.device_vaes = device_vaes
        
        # Text encoder
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # prompt_template_video
        prompt_template_video = PROMPT_TEMPLATE[args.prompt_template_video] \
                                if args.prompt_template_video is not None else None
        print("="*25, f"load llava", "="*25)
        text_encoder = TextEncoder(text_encoder_type = args.text_encoder,
                                   max_length = max_length,
                                   text_encoder_precision = args.text_encoder_precision,
                                   tokenizer_type = args.tokenizer,
                                   use_attention_mask = args.use_attention_mask,
                                   prompt_template_video = prompt_template_video,
                                   hidden_state_skip_layer = args.hidden_state_skip_layer,
                                   apply_final_norm = args.apply_final_norm,
                                   reproduce = args.reproduce,
                                   logger = logger,
                                   device = 'cpu' if args.cpu_offload else device ,
                                   )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(text_encoder_type=args.text_encoder_2,
                                         max_length=args.text_len_2,
                                         text_encoder_precision=args.text_encoder_precision_2,
                                         tokenizer_type=args.tokenizer_2,
                                         use_attention_mask=args.use_attention_mask,
                                         reproduce=args.reproduce,
                                         logger=logger,
                                         device='cpu' if args.cpu_offload else device , 
                                         # if not args.use_cpu_offload else 'cpu'
                                         )

        return cls(args=args, 
                   vae=vae, 
                   vae_kwargs=vae_kwargs, 
                   text_encoder=text_encoder,
                   model=model, 
                   text_encoder_2=text_encoder_2, 
                   device=device, 
                   logger=logger)

    @staticmethod
    def load_state_dict(args, model, ckpt_path):
        load_key = args.load_key
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_dir():
            ckpt_path = next(ckpt_path.glob("*_model_states.pt"))
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        elif load_key == ".":
            pass
        else:
            raise KeyError(f"Key '{load_key}' not found in the checkpoint. Existed keys: {state_dict.keys()}")
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_exp_dir_and_ckpt_id(self):
        if self.ckpt is None:
            raise ValueError("The checkpoint path is not provided.")

        ckpt = Path(self.ckpt)
        if ckpt.parents[1].name == "checkpoints":
            # It should be a standard checkpoint path. We use the parent directory as the default save directory.
            exp_dir = ckpt.parents[2]
        else:
            raise ValueError(f"We cannot infer the experiment directory from the checkpoint path: {ckpt}. "
                             f"It seems that the checkpoint path is not standard. Please explicitly provide the "
                             f"save path by --save-path.")
        return exp_dir, ckpt.parent.name

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size
