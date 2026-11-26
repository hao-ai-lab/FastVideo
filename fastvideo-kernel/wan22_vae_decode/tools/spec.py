"""The task: decoding Wan2.2 VAE latents into video on one, two or four GB200 GPUs of one node, as bench.py, ci.py and
bounds.py read it. It defines the clips and workloads, the weights' order, the data, the error allowance and the work
of one decode.

What is computed
----------------
reference.py (in the kernel repo) defines the result: diffusers' AutoencoderKLWan._decode of the Wan2.2-TI2V-5B VAE,
latents z [1, 48, T, 44, 78] to a video [1, 3, 4T-3, 704, 1248] in [-1, 1]: post_quant_conv, then the decoder one
latent frame at a time with its causal caches (the first latent frame yields one frame, every later one four),
unpatchify 2 x 2, clamp. The judge compares a solution with the fp32 decode of the same latents, cached as videos in
$DATA (see Correctness).

The interface (as loader.py builds it and bench.py calls it, positional, destination passing)
-------------
    run(z, weights, rank, world, pg, out)
    prepare(weights, rank, world, pg) -> list[Tensor]       optional

- z: [1, 48, T, 44, 78] bf16, contiguous, the full latents on every rank: what a pipeline passes to
  vae.decode(latents), rounded to bf16.
- weights: the 110 tensors of WEIGHTS below, bf16, on this rank's GPU; or what prepare returned.
- rank, world: this process's rank and the number of ranks (1, 2 or 4), rank r on GPU r.
- pg: torch.distributed's NCCL ProcessGroup over the ranks; a C++ entry takes it as
  `const c10::intrusive_ptr<c10d::ProcessGroup>&` (from <torch/csrc/distributed/c10d/ProcessGroup.hpp>).
- out: on rank 0, the video [1, 3, 4T-3, 704, 1248] bf16, contiguous, from torch.empty and not zeroed (every element
  must be written); on the other ranks an empty tensor. When run has returned on every rank, rank 0's out holds the
  video.
- prepare, when the module exports it, is called once per process before any timed call, and is not timed. It sees
  the weights, not the latents; its result replaces `weights` in every run.

Numerics: bf16 weights and activations with fp32 accumulation, and fp32 statistics in the RMS norms; nothing narrower
(no FP8, FP6 or FP4). This is a rule for the reviewers: no check reads it from the code.

Workloads
---------
A workload is a clip decoded at a world size, named <clip>x<world>. A clip is the first T latent frames of one of
three real 20 s clips encoded by the VAE (latents_<source>.pt in $DATA):

    clip   source       latent frames T   video frames 4T-3   worlds    timed
    f1     row_000168          1                   1           1, 4      no
    f5     row_000168          2                   5           1, 4      no
    f17    row_000168          5                  17           1, 4      no
    f121   row_000168         31                 121           1, 2, 4   yes
    f241   row_000129         61                 241           1, 2, 4   yes
    f481   row_000090        121                 481           1, 2, 4   yes

The nine timed workloads (f121x1 ... f481x4) make the objective, their geomean; the six short ones (f1x1 ... f17x4)
are checked for correctness and never timed: a decoder sees every clip length, and the first latent frame is a
special case.

Correctness
-----------
On rank 0's out, against ref_<clip>.pt, the fp32 decode of the clip's latents (fp32 latents, TF32 off), stored as fp16:
every element finite and within [-1, 1]; and with d = out - ref in fp32,

    max |d|  <= min(MAX_ABS_CAP, ERROR_FACTOR x bf16_max_abs)
    rms(d)   <= ERROR_FACTOR x bf16_rms

bf16_max_abs and bf16_rms being the error of the release's bf16 decode (bf16 weights under bf16 autocast, the
latents rounded to bf16) against that fp32 decode on the same clip, from calibration.json. PSNR, 10 log10(4 / mse), is
shown beside them.

The weights
-----------
run() gets the decoder's parameters and post_quant_conv's, 110 tensors, as a list in diffusers' registration order
(that of AutoencoderKLWan.named_parameters(), post_quant_conv first, then the decoder from conv_in to conv_out;
$DATA/weight_names.json lists the same), bf16, in PyTorch's layouts: a conv3d's weight is [out, in, kt, kh, kw], a
conv2d's [out, in, kh, kw], a bias [out], an RMS norm's gain [C, 1, 1, 1] (the attention's [C, 1, 1]).
$DATA/decoder_weights_bf16.pt is that list: the release's fp32 weights rounded to bf16 to nearest even, as diffusers'
from_pretrained(torch_dtype=torch.bfloat16) casts them. The order is not the order of execution: the mid block
registers its attention before its two residual blocks, though it runs resnets.0, the attention, then resnets.1; an
upsampler registers its 3x3 conv (resample.1) before its time conv, though the time conv runs first. The attention's
to_qkv gives q, k and v as its output channels 0-1023, 1024-2047 and 2048-3071. A time conv C -> 2C gives frame 2t of
its output as channels 0..C-1 and frame 2t+1 as channels C..2C-1. conv_out's 12 channels are (colour k, column offset
j, row offset i) of 2 x 2 pixel patches: video[k, f, 2y + i, 2x + j] = channel 4k + 2j + i at (f, y, x).

      #  name                                                shape                  role
      0  post_quant_conv.weight                              [48, 48, 1, 1, 1]      1x1x1 conv 48 -> 48, on the latents
      1  post_quant_conv.bias                                [48]
      2  decoder.conv_in.weight                              [1024, 48, 3, 3, 3]    causal 3x3x3 conv 48 -> 1024
      3  decoder.conv_in.bias                                [1024]
      4  decoder.mid_block.attentions.0.norm.gamma           [1024, 1, 1]           RMS norm gain, before to_qkv
      5  decoder.mid_block.attentions.0.to_qkv.weight        [3072, 1024, 1, 1]     1x1 conv2d 1024 -> 3072 (q, k, v)
      6  decoder.mid_block.attentions.0.to_qkv.bias          [3072]
      7  decoder.mid_block.attentions.0.proj.weight          [1024, 1024, 1, 1]     1x1 conv2d 1024 -> 1024
      8  decoder.mid_block.attentions.0.proj.bias            [1024]
      9  decoder.mid_block.resnets.0.norm1.gamma             [1024, 1, 1, 1]        RMS norm gain, before conv1
     10  decoder.mid_block.resnets.0.conv1.weight            [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     11  decoder.mid_block.resnets.0.conv1.bias              [1024]
     12  decoder.mid_block.resnets.0.norm2.gamma             [1024, 1, 1, 1]        RMS norm gain, before conv2
     13  decoder.mid_block.resnets.0.conv2.weight            [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     14  decoder.mid_block.resnets.0.conv2.bias              [1024]
     15  decoder.mid_block.resnets.1.norm1.gamma             [1024, 1, 1, 1]        RMS norm gain, before conv1
     16  decoder.mid_block.resnets.1.conv1.weight            [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     17  decoder.mid_block.resnets.1.conv1.bias              [1024]
     18  decoder.mid_block.resnets.1.norm2.gamma             [1024, 1, 1, 1]        RMS norm gain, before conv2
     19  decoder.mid_block.resnets.1.conv2.weight            [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     20  decoder.mid_block.resnets.1.conv2.bias              [1024]
     21  decoder.up_blocks.0.resnets.0.norm1.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv1
     22  decoder.up_blocks.0.resnets.0.conv1.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     23  decoder.up_blocks.0.resnets.0.conv1.bias            [1024]
     24  decoder.up_blocks.0.resnets.0.norm2.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv2
     25  decoder.up_blocks.0.resnets.0.conv2.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     26  decoder.up_blocks.0.resnets.0.conv2.bias            [1024]
     27  decoder.up_blocks.0.resnets.1.norm1.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv1
     28  decoder.up_blocks.0.resnets.1.conv1.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     29  decoder.up_blocks.0.resnets.1.conv1.bias            [1024]
     30  decoder.up_blocks.0.resnets.1.norm2.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv2
     31  decoder.up_blocks.0.resnets.1.conv2.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     32  decoder.up_blocks.0.resnets.1.conv2.bias            [1024]
     33  decoder.up_blocks.0.resnets.2.norm1.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv1
     34  decoder.up_blocks.0.resnets.2.conv1.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     35  decoder.up_blocks.0.resnets.2.conv1.bias            [1024]
     36  decoder.up_blocks.0.resnets.2.norm2.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv2
     37  decoder.up_blocks.0.resnets.2.conv2.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     38  decoder.up_blocks.0.resnets.2.conv2.bias            [1024]
     39  decoder.up_blocks.0.upsampler.resample.1.weight     [1024, 1024, 3, 3]     3x3 conv2d 1024 -> 1024, after x2
     40  decoder.up_blocks.0.upsampler.resample.1.bias       [1024]
     41  decoder.up_blocks.0.upsampler.time_conv.weight      [2048, 1024, 3, 1, 1]  causal 3x1x1 conv 1024 -> 2048
     42  decoder.up_blocks.0.upsampler.time_conv.bias        [2048]
     43  decoder.up_blocks.1.resnets.0.norm1.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv1
     44  decoder.up_blocks.1.resnets.0.conv1.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     45  decoder.up_blocks.1.resnets.0.conv1.bias            [1024]
     46  decoder.up_blocks.1.resnets.0.norm2.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv2
     47  decoder.up_blocks.1.resnets.0.conv2.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     48  decoder.up_blocks.1.resnets.0.conv2.bias            [1024]
     49  decoder.up_blocks.1.resnets.1.norm1.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv1
     50  decoder.up_blocks.1.resnets.1.conv1.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     51  decoder.up_blocks.1.resnets.1.conv1.bias            [1024]
     52  decoder.up_blocks.1.resnets.1.norm2.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv2
     53  decoder.up_blocks.1.resnets.1.conv2.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     54  decoder.up_blocks.1.resnets.1.conv2.bias            [1024]
     55  decoder.up_blocks.1.resnets.2.norm1.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv1
     56  decoder.up_blocks.1.resnets.2.conv1.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     57  decoder.up_blocks.1.resnets.2.conv1.bias            [1024]
     58  decoder.up_blocks.1.resnets.2.norm2.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv2
     59  decoder.up_blocks.1.resnets.2.conv2.weight          [1024, 1024, 3, 3, 3]  causal 3x3x3 conv 1024 -> 1024
     60  decoder.up_blocks.1.resnets.2.conv2.bias            [1024]
     61  decoder.up_blocks.1.upsampler.resample.1.weight     [1024, 1024, 3, 3]     3x3 conv2d 1024 -> 1024, after x2
     62  decoder.up_blocks.1.upsampler.resample.1.bias       [1024]
     63  decoder.up_blocks.1.upsampler.time_conv.weight      [2048, 1024, 3, 1, 1]  causal 3x1x1 conv 1024 -> 2048
     64  decoder.up_blocks.1.upsampler.time_conv.bias        [2048]
     65  decoder.up_blocks.2.resnets.0.norm1.gamma           [1024, 1, 1, 1]        RMS norm gain, before conv1
     66  decoder.up_blocks.2.resnets.0.conv1.weight          [512, 1024, 3, 3, 3]   causal 3x3x3 conv 1024 -> 512
     67  decoder.up_blocks.2.resnets.0.conv1.bias            [512]
     68  decoder.up_blocks.2.resnets.0.norm2.gamma           [512, 1, 1, 1]         RMS norm gain, before conv2
     69  decoder.up_blocks.2.resnets.0.conv2.weight          [512, 512, 3, 3, 3]    causal 3x3x3 conv 512 -> 512
     70  decoder.up_blocks.2.resnets.0.conv2.bias            [512]
     71  decoder.up_blocks.2.resnets.0.conv_shortcut.weight  [512, 1024, 1, 1, 1]   1x1x1 conv 1024 -> 512, shortcut
     72  decoder.up_blocks.2.resnets.0.conv_shortcut.bias    [512]
     73  decoder.up_blocks.2.resnets.1.norm1.gamma           [512, 1, 1, 1]         RMS norm gain, before conv1
     74  decoder.up_blocks.2.resnets.1.conv1.weight          [512, 512, 3, 3, 3]    causal 3x3x3 conv 512 -> 512
     75  decoder.up_blocks.2.resnets.1.conv1.bias            [512]
     76  decoder.up_blocks.2.resnets.1.norm2.gamma           [512, 1, 1, 1]         RMS norm gain, before conv2
     77  decoder.up_blocks.2.resnets.1.conv2.weight          [512, 512, 3, 3, 3]    causal 3x3x3 conv 512 -> 512
     78  decoder.up_blocks.2.resnets.1.conv2.bias            [512]
     79  decoder.up_blocks.2.resnets.2.norm1.gamma           [512, 1, 1, 1]         RMS norm gain, before conv1
     80  decoder.up_blocks.2.resnets.2.conv1.weight          [512, 512, 3, 3, 3]    causal 3x3x3 conv 512 -> 512
     81  decoder.up_blocks.2.resnets.2.conv1.bias            [512]
     82  decoder.up_blocks.2.resnets.2.norm2.gamma           [512, 1, 1, 1]         RMS norm gain, before conv2
     83  decoder.up_blocks.2.resnets.2.conv2.weight          [512, 512, 3, 3, 3]    causal 3x3x3 conv 512 -> 512
     84  decoder.up_blocks.2.resnets.2.conv2.bias            [512]
     85  decoder.up_blocks.2.upsampler.resample.1.weight     [512, 512, 3, 3]       3x3 conv2d 512 -> 512, after x2
     86  decoder.up_blocks.2.upsampler.resample.1.bias       [512]
     87  decoder.up_blocks.3.resnets.0.norm1.gamma           [512, 1, 1, 1]         RMS norm gain, before conv1
     88  decoder.up_blocks.3.resnets.0.conv1.weight          [256, 512, 3, 3, 3]    causal 3x3x3 conv 512 -> 256
     89  decoder.up_blocks.3.resnets.0.conv1.bias            [256]
     90  decoder.up_blocks.3.resnets.0.norm2.gamma           [256, 1, 1, 1]         RMS norm gain, before conv2
     91  decoder.up_blocks.3.resnets.0.conv2.weight          [256, 256, 3, 3, 3]    causal 3x3x3 conv 256 -> 256
     92  decoder.up_blocks.3.resnets.0.conv2.bias            [256]
     93  decoder.up_blocks.3.resnets.0.conv_shortcut.weight  [256, 512, 1, 1, 1]    1x1x1 conv 512 -> 256, shortcut
     94  decoder.up_blocks.3.resnets.0.conv_shortcut.bias    [256]
     95  decoder.up_blocks.3.resnets.1.norm1.gamma           [256, 1, 1, 1]         RMS norm gain, before conv1
     96  decoder.up_blocks.3.resnets.1.conv1.weight          [256, 256, 3, 3, 3]    causal 3x3x3 conv 256 -> 256
     97  decoder.up_blocks.3.resnets.1.conv1.bias            [256]
     98  decoder.up_blocks.3.resnets.1.norm2.gamma           [256, 1, 1, 1]         RMS norm gain, before conv2
     99  decoder.up_blocks.3.resnets.1.conv2.weight          [256, 256, 3, 3, 3]    causal 3x3x3 conv 256 -> 256
    100  decoder.up_blocks.3.resnets.1.conv2.bias            [256]
    101  decoder.up_blocks.3.resnets.2.norm1.gamma           [256, 1, 1, 1]         RMS norm gain, before conv1
    102  decoder.up_blocks.3.resnets.2.conv1.weight          [256, 256, 3, 3, 3]    causal 3x3x3 conv 256 -> 256
    103  decoder.up_blocks.3.resnets.2.conv1.bias            [256]
    104  decoder.up_blocks.3.resnets.2.norm2.gamma           [256, 1, 1, 1]         RMS norm gain, before conv2
    105  decoder.up_blocks.3.resnets.2.conv2.weight          [256, 256, 3, 3, 3]    causal 3x3x3 conv 256 -> 256
    106  decoder.up_blocks.3.resnets.2.conv2.bias            [256]
    107  decoder.norm_out.gamma                              [256, 1, 1, 1]         RMS norm gain, before conv_out
    108  decoder.conv_out.weight                             [12, 256, 3, 3, 3]     causal 3x3x3 conv 256 -> 12
    109  decoder.conv_out.bias                               [12]

Up blocks 0 and 1 run at 1024 channels, up block 2 from 1024 to 512, up block 3 from 512 to 256; a residual block's
first conv changes the channel count, and its 1x1x1 conv_shortcut does so on the shortcut.

Data
----
$DATA (default /data), read-only: latents_<source>.pt ({"latents": fp32 [1, 48, 121, 44, 78]}), ref_<clip>.pt
({"video": fp16 [1, 3, 4T-3, 704, 1248], "source", "latent_frames"}), calibration.json (per clip: bf16_max_abs,
bf16_mean_abs, bf16_rms, bf16_psnr_db, fp32_decode_s, bf16_decode_s), weight_names.json and decoder_weights_bf16.pt.

The work of one decode (bounds)
-------------------------------
FLOPs, a multiply-add counting 2: every convolution at its full kernel over its output positions (2 x the weight's
elements x frames x rows x columns, zero padding included, which adds a few percent at 44 x 78 and less than one
beyond), plus the attention's two products per latent frame (2 x 2 x 3432^2 x 1024). The frames at each level, for T
latent frames: T at 44 x 78 (post_quant_conv, conv_in, the mid block, up block 0's residual blocks); T - 1 for up
block 0's time conv, which skips the first latent frame; 2T - 1 at 88 x 156 (up block 0's 3x3 conv, up block 1's
residual blocks); 2T - 2 for up block 1's time conv; 4T - 3 at 176 x 312 (up block 1's 3x3 conv, up block 2's
residual blocks) and at 352 x 624 (up block 2's 3x3 conv, up block 3, conv_out). The FLOPs are divided over the
workload's GPUs. Bytes: the video written (on rank 0) and the weights read once.
"""

import json
import math
import os

LATENT_CHANNELS = 48
LATENT_HW = (44, 78)
VIDEO_HW = (704, 1248)
# How far a solution may err, against what the release's bf16 decode errs by on the same clip (see Correctness).
ERROR_FACTOR = 3.0
MAX_ABS_CAP = 2.0
WEIGHTS_FILE = "decoder_weights_bf16.pt"
NAMES_FILE = "weight_names.json"

CLIPS = {
    "f1": {"source": "row_000168", "latent_frames": 1},
    "f5": {"source": "row_000168", "latent_frames": 2},
    "f17": {"source": "row_000168", "latent_frames": 5},
    "f121": {"source": "row_000168", "latent_frames": 31},
    "f241": {"source": "row_000129", "latent_frames": 61},
    "f481": {"source": "row_000090", "latent_frames": 121},
}
# The short clips are checked at one GPU and at four, never timed; the long ones are checked and timed at 1, 2 and 4.
SHORT_CLIPS, LONG_CLIPS = ("f1", "f5", "f17"), ("f121", "f241", "f481")
WORKLOADS = {
    **{f"{c}x{n}": {"clip": c, "world": n, "timed": False} for c in SHORT_CLIPS for n in (1, 4)},
    **{f"{c}x{n}": {"clip": c, "world": n, "timed": True} for c in LONG_CLIPS for n in (1, 2, 4)},
}

# decoder.up_blocks.{i}: (in channels, out channels, temporal upsampling, spatial upsampling)
UP_BLOCKS = ((1024, 1024, True, True), (1024, 1024, True, True), (1024, 512, False, True), (512, 256, False, False))
RESNETS_PER_UP_BLOCK = 3


def _weights():
    """(name, shape) of run()'s weights in diffusers' registration order (see The weights)."""
    out = []

    def conv(name, shape):
        out.extend([(f"{name}.weight", tuple(shape)), (f"{name}.bias", (shape[0],))])

    def resnet(name, cin, cout):
        out.append((f"{name}.norm1.gamma", (cin, 1, 1, 1)))
        conv(f"{name}.conv1", (cout, cin, 3, 3, 3))
        out.append((f"{name}.norm2.gamma", (cout, 1, 1, 1)))
        conv(f"{name}.conv2", (cout, cout, 3, 3, 3))
        if cin != cout:
            conv(f"{name}.conv_shortcut", (cout, cin, 1, 1, 1))

    conv("post_quant_conv", (48, 48, 1, 1, 1))
    conv("decoder.conv_in", (1024, 48, 3, 3, 3))
    out.append(("decoder.mid_block.attentions.0.norm.gamma", (1024, 1, 1)))
    conv("decoder.mid_block.attentions.0.to_qkv", (3072, 1024, 1, 1))
    conv("decoder.mid_block.attentions.0.proj", (1024, 1024, 1, 1))
    for j in range(2):
        resnet(f"decoder.mid_block.resnets.{j}", 1024, 1024)
    for i, (cin, cout, temporal, spatial) in enumerate(UP_BLOCKS):
        for j in range(RESNETS_PER_UP_BLOCK):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}", cin if j == 0 else cout, cout)
        if spatial:
            conv(f"decoder.up_blocks.{i}.upsampler.resample.1", (cout, cout, 3, 3))
        if temporal:
            conv(f"decoder.up_blocks.{i}.upsampler.time_conv", (2 * cout, cout, 3, 1, 1))
    out.append(("decoder.norm_out.gamma", (256, 1, 1, 1)))
    conv("decoder.conv_out", (12, 256, 3, 3, 3))
    return tuple(out)


WEIGHTS = _weights()
WEIGHT_NAMES = [name for name, _ in WEIGHTS]


def data_dir():
    """The task's data: $DATA, else /data, where the jobs mount it."""
    return os.environ.get("DATA") or "/data"


def frames(latent_frames):
    """The video frames of T latent frames: one for the first, four for every later one."""
    return 4 * latent_frames - 3


def output_shape(workload):
    """The shape of rank 0's out at a workload."""
    return (1, 3, frames(CLIPS[WORKLOADS[workload]["clip"]]["latent_frames"]), *VIDEO_HW)


def _load(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"cannot read {path}: {e}") from e


def make_inputs(workload, device):
    """[z]: the clip's latents, the first T latent frames of its source, rounded to bf16, contiguous on `device`."""
    import torch

    clip = CLIPS[WORKLOADS[workload]["clip"]]
    path = os.path.join(data_dir(), f"latents_{clip['source']}.pt")
    latents = _load(path)["latents"]
    t = clip["latent_frames"]
    if latents.dim() != 5 or tuple(latents.shape[:2]) != (1, LATENT_CHANNELS) or tuple(latents.shape[3:]) != LATENT_HW \
            or latents.shape[2] < t:
        raise ValueError(f"{path}: latents {tuple(latents.shape)}, expected [1, 48, >= {t}, 44, 78]")
    return [latents[:, :, :t].to(device=device, dtype=torch.bfloat16).contiguous()]


def load_weights(device):
    """run()'s weights on `device`: decoder_weights_bf16.pt, checked against WEIGHTS (and weight_names.json where it
    is), so that a solution indexes what this file documents."""
    import torch

    listed = os.path.join(data_dir(), NAMES_FILE)
    if os.path.exists(listed):
        with open(listed) as f:
            names = [(e["name"], tuple(e["shape"])) for e in json.load(f)]
        if names != list(WEIGHTS):
            raise ValueError(f"{listed} does not list spec.WEIGHTS: the weights' order or shapes changed")
    path = os.path.join(data_dir(), WEIGHTS_FILE)
    weights = _load(path)
    if not isinstance(weights, list) or len(weights) != len(WEIGHTS):
        raise ValueError(f"{path}: expected a list of {len(WEIGHTS)} tensors")
    for i, (t, (name, shape)) in enumerate(zip(weights, WEIGHTS)):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16 or tuple(t.shape) != shape:
            got = f"{t.dtype} {tuple(t.shape)}" if isinstance(t, torch.Tensor) else type(t).__name__
            raise ValueError(f"{path}: tensor {i} is {got}, expected {name}, bf16 {shape}")
    return [t.to(device).contiguous() for t in weights]


def load_reference(clip):
    """The clip's fp32 decode, as stored: a CPU fp16 tensor [1, 3, 4T-3, 704, 1248] (memory-mapped)."""
    import torch

    path = os.path.join(data_dir(), f"ref_{clip}.pt")
    ref = _load(path)
    t, source = CLIPS[clip]["latent_frames"], CLIPS[clip]["source"]
    video = ref.get("video") if isinstance(ref, dict) else None
    shape = (1, 3, frames(t), *VIDEO_HW)
    if not isinstance(video, torch.Tensor) or tuple(video.shape) != shape or video.dtype != torch.float16:
        raise ValueError(f"{path}: expected video, fp16 {list(shape)}")
    if ref.get("source") != source or ref.get("latent_frames") != t:
        raise ValueError(f"{path} decodes {ref.get('source')}'s first {ref.get('latent_frames')} latent frames, not "
                         f"{source}'s first {t}")
    return video


def calibration(clip):
    """The release's bf16 decode of the clip against its fp32 decode (calibration.json's entry)."""
    path = os.path.join(data_dir(), "calibration.json")
    try:
        with open(path) as f:
            entry = json.load(f)[clip]
    except (OSError, ValueError, KeyError) as e:
        raise ValueError(f"{path} has no entry for {clip}: {e}") from e
    if entry.get("source") != CLIPS[clip]["source"] or entry.get("latent_frames") != CLIPS[clip]["latent_frames"]:
        raise ValueError(f"{path}: {clip} is calibrated on another clip ({entry.get('source')}, "
                         f"{entry.get('latent_frames')} latent frames)")
    return entry


def allowance(clip):
    """{"max_abs", "rms"}: how far a solution's video may be from the fp32 decode of the clip (see Correctness)."""
    entry = calibration(clip)
    return {"max_abs": min(MAX_ABS_CAP, ERROR_FACTOR * entry["bf16_max_abs"]), "rms": ERROR_FACTOR * entry["bf16_rms"]}


def _conv_level(name, t):
    """(output frames, upsampling factor in space) of the conv whose weight is `name`, decoding t latent frames."""
    if name.startswith(("post_quant_conv.", "decoder.conv_in.", "decoder.mid_block.", "decoder.up_blocks.0.resnets.")):
        return t, 1
    if name.startswith("decoder.up_blocks.0.upsampler.time_conv."):
        return t - 1, 1
    if name.startswith(("decoder.up_blocks.0.upsampler.resample.", "decoder.up_blocks.1.resnets.")):
        return 2 * t - 1, 2
    if name.startswith("decoder.up_blocks.1.upsampler.time_conv."):
        return 2 * t - 2, 2
    if name.startswith(("decoder.up_blocks.1.upsampler.resample.", "decoder.up_blocks.2.resnets.")):
        return 4 * t - 3, 4
    return 4 * t - 3, 8


def decode_flops(t):
    """FLOPs of one decode of t latent frames (see The work of one decode)."""
    h, w = LATENT_HW
    total = 0
    for name, shape in WEIGHTS:
        if name.endswith(".weight") and len(shape) >= 4:
            count, scale = _conv_level(name, t)
            total += 2 * math.prod(shape) * count * (h * scale) * (w * scale)
    return total + 2 * 2 * (h * w) ** 2 * 1024 * t


def bounds(workload):
    """{"flops", "bytes", "gpus"} of one call at the workload, for bounds.py: the decode's FLOPs (spread over the
    world's GPUs), and the bytes that must move: the video written and the weights read."""
    entry = WORKLOADS[workload]
    t = CLIPS[entry["clip"]]["latent_frames"]
    video = math.prod(output_shape(workload)) * 2
    weights = sum(math.prod(shape) for _, shape in WEIGHTS) * 2
    return {"flops": decode_flops(t), "bytes": video + weights, "gpus": entry["world"]}


# The stack rules beside audit.py's: run() reads nothing but its arguments.
RULES = {
    "file-io": {
        "include": r"(fstream|fcntl\.h|sys/mman\.h|filesystem|experimental/filesystem)",
        "text": r"\b(fopen|fopen64|freopen|fdopen|fread|fwrite|ifstream|ofstream|fstream|filebuf|mmap|mmap64|shm_open|"
                r"torch::load|torch::save|torch::pickle_load|torch::pickle_save|torch::jit::load)\b|"
                r"::\s*(open|openat|read|pread|write|pwrite)\s*\(",
        "why": "run() and prepare() read nothing but their arguments and write nothing but out: no files (the "
               "reference videos lie on the job's disk) and nothing that outlives a call"},
}


if __name__ == "__main__":
    for w in WORKLOADS:
        b = bounds(w)
        print(f"{w:<8} {b['flops'] / 1e15:8.4f} PFLOP over {b['gpus']} GPU(s), {b['bytes'] / 1e9:6.2f} GB")
