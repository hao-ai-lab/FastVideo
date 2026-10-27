"""The decoder of the Wan2.2 VAE (Wan2.2-TI2V-5B, 48 latent channels) on sm_100a (B200 / GB200), on one or several GPUs
of a node, built from solution/ on first use.

It decodes latents z [1, 48, T, 44, 78] (bf16) into the video [1, 3, 4T - 3, 704, 1248] (bf16, in [-1, 1]), as
diffusers' AutoencoderKLWan.decode(z).sample computes it (reference.py defines the result), for any T >= 1. Other
latent sizes and batches above one are refused: the kernels are specialised to 704 x 1248.

    decoder = WanVAEDecoder(vae.state_dict(), device="cuda")    # diffusers' AutoencoderKLWan, or its state dict
    video = decoder.decode(latents)                              # [1, 3, 4T - 3, 704, 1248] bf16

Over several GPUs, every rank builds a decoder with the same process group and calls decode with the full latents;
rank 0 gets the video and the other ranks None. Without a process group, decode uses torch.distributed's default
group, and initializes a single-rank NCCL group when there is none (one GPU only). With local=True each process decodes
on its own GPU alone, whatever torch.distributed holds: one rank of one, so the kernels never communicate (comm.cu
returns before any exchange below two ranks) and no group has to be created collectively.

frames_per_pass sets how many latent frames each GPU carries through one pass of the decoder, which trades memory for
speed; the video is the same at any value. On one GB200 at 8 to 20 s of video: 2 frames peak at 14-16 GiB allocated
(x0.85 the speed of 16 frames), 4 (the default) at 21-22 GiB (x0.92), 8 at 34-35 GiB (x0.97), 16 at 60-61 GiB. The
allocator reserves more than it allocates; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True keeps the difference small.
"""

import atexit
import functools
import importlib.util
import os
import sys

import torch
import torch.distributed as dist
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
LATENT_CHANNELS, LATENT_HW, VIDEO_HW = 48, (44, 78), (704, 1248)


def _own(path, name):
    """The module at path, imported under a name of this kernel's own. Other kernels built the same way
    keep tools and references of the same file names; a bare import would hand whichever loaded first to both."""
    name = f"wan22_vae_decode_{name}"
    module = sys.modules.get(name)
    if module is None:
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return module


@functools.lru_cache(maxsize=None)
def _tools():
    """tools/loader.py and tools/spec.py."""
    return (_own(os.path.join(HERE, "tools", "loader.py"), "loader"),
            _own(os.path.join(HERE, "tools", "spec.py"), "spec"))


@functools.lru_cache(maxsize=None)
def _module():
    """The extension built from solution/, once per process. Builds are kept in WAN22_VAE_DECODE_BUILD_DIR (default
    ~/.cache/fastvideo/wan22_vae_decode), one directory per local rank, and reused while the sources and flags match."""
    loader, _ = _tools()
    base = os.environ.get("WAN22_VAE_DECODE_BUILD_DIR", os.path.expanduser("~/.cache/fastvideo/wan22_vae_decode"))
    build_dir = os.path.join(base, f"rank{os.environ.get('LOCAL_RANK', '0')}")
    os.makedirs(base, exist_ok=True)
    return loader.load(os.path.join(HERE, "solution"), build_dir=build_dir, chdir=False).module


def weight_names() -> list[str]:
    """The 110 parameters the decoder takes, under diffusers' AutoencoderKLWan names, in the order it takes them."""
    return list(_tools()[1].WEIGHT_NAMES)


def decoder_weights(state_dict, device=None) -> list[Tensor]:
    """The decoder's weights from an AutoencoderKLWan state dict (or the module itself): its post_quant_conv and
    decoder parameters, in weight_names() order, as contiguous bf16 tensors on `device`."""
    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()
    _, spec = _tools()
    weights = []
    for name, shape in spec.WEIGHTS:
        if name not in state_dict:
            raise KeyError(f"the state dict has no {name}: is it the Wan2.2-TI2V-5B VAE's?")
        w = state_dict[name]
        if tuple(w.shape) != shape:
            raise ValueError(f"{name} is {tuple(w.shape)}, expected {shape}")
        weights.append(w.detach().to(device=device, dtype=torch.bfloat16).contiguous())
    return weights


def _destroy_if_initialized():
    # The group may be gone already (a caller destroyed it), and destroying it twice raises.
    if dist.is_initialized():
        dist.destroy_process_group()


def _default_group():
    if not dist.is_initialized():
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
        atexit.register(_destroy_if_initialized)
    return dist.group.WORLD


class WanVAEDecoder:
    """The decoder with its weights prepared once (the layout and the Winograd transforms the kernels read)."""

    def __init__(self, state_dict, device=None, pg=None, local=False, frames_per_pass=4):
        if frames_per_pass < 1:
            raise ValueError(f"frames_per_pass must be at least 1, not {frames_per_pass}")
        self.frames_per_pass = int(frames_per_pass)
        self.device = torch.device(device if device is not None else "cuda")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(self.device):
            if local:
                if pg is not None:
                    raise ValueError("local=True decodes on this GPU alone; it takes no process group")
                # At world 1 the group is only carried, never used: any initialized group serves.
                self.pg = dist.group.WORLD if dist.is_initialized() else _default_group()
                self.rank, self.world = 0, 1
            else:
                self.pg = pg if pg is not None else _default_group()
                self.rank, self.world = dist.get_rank(self.pg), dist.get_world_size(self.pg)
            if self.world > 8:
                raise ValueError(f"the decoder splits over at most 8 GPUs, not {self.world}")
            self.weights = _module().prepare(decoder_weights(state_dict, self.device), self.rank, self.world,
                                             self.pg)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor | None:
        """The video of latents z [1, 48, T, 44, 78] on rank 0, None on the other ranks."""
        if z.dim() != 5 or z.shape[0] != 1 or z.shape[1] != LATENT_CHANNELS or tuple(z.shape[3:]) != LATENT_HW:
            raise ValueError(f"latents {tuple(z.shape)}: expected [1, 48, T, 44, 78]")
        z = z.to(device=self.device, dtype=torch.bfloat16).contiguous()
        frames = 4 * z.shape[2] - 3
        with torch.cuda.device(self.device):
            out = torch.empty((1, 3, frames, *VIDEO_HW) if self.rank == 0 else (0,), dtype=torch.bfloat16,
                              device=self.device)
            _module().run_with(z, self.weights, self.rank, self.world, self.pg, out, self.frames_per_pass)
        return out if self.rank == 0 else None
