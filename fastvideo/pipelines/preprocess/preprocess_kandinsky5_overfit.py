# SPDX-License-Identifier: Apache-2.0
"""Preprocess Kandinsky5 overfit data into parquet format.

Encodes videos with Kandinsky5's VAE (shared with HunyuanVideo) and
captions with dual text encoders (Qwen/Reason1 + CLIP) into the t2v
parquet schema expected by the training framework.

Unlike Hunyuan's version of this script, text encoders are loaded via
FastVideo's own ``TextEncoderLoader``/``VAELoader`` (matching
``Kandinsky5Model.ensure_negative_conditioning``) rather than raw
``transformers`` classes -- Kandinsky5's Qwen/Reason1 encoder is a
native FastVideo wrapper around Qwen2.5-VL's language model, not a
plain ``AutoModel.from_pretrained`` load.

The CLIP pooled projection (``[768]``) is zero-padded into a row and
prepended to the Qwen sequence embeddings, then stored as a single
``[seq+1, dim]`` tensor in the existing ``text_embedding`` parquet
field -- the same trick ``preprocess_hunyuan_overfit.py`` uses for
LLaMA+CLIP. No schema change and no separate attention-mask field are
needed: the standard collator derives ``text_attention_mask`` from
padding against the stored (unpadded) row count, so the prepended
pooled row is automatically counted as valid.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m fastvideo.pipelines.preprocess.preprocess_kandinsky5_overfit

Input/output roots default to ``data/kandinsky5_overfit`` /
``data/kandinsky5_overfit_preprocessed`` and can be overridden with the
``KANDINSKY5_OVERFIT_DATA_DIR`` / ``KANDINSKY5_OVERFIT_OUTPUT_DIR`` env
vars (used by the nightly e2e test to keep its disposable roots separate
from a user's real dataset at the defaults).
"""

import json
import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29520")

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from fastvideo.configs.pipelines.base import preprocess_text
from fastvideo.configs.pipelines.kandinsky5 import (
    Kandinsky5T2VConfig,
    kandinsky5_clip_postprocess_text,
    kandinsky5_qwen_postprocess_text,
    kandinsky5_qwen_preprocess_text,
)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TextEncoderLoader, VAELoader
from fastvideo.utils import maybe_download_model

# --- Config ---
# Matches KANDINSKY5_T2V_LITE_5S's native preset (fastvideo/pipelines/basic/
# kandinsky5/presets.py): 121 frames @ 24fps ~= 5.04s, matching the "5s" in
# the checkpoint name. 512/768 fall in the 480p RoPE scale_factor band
# Kandinsky5Model asserts on, and satisfy the real divisor requirement
# (spatial_compression_ratio(8) * patch_size[1](2) = 16 -- not just 8).
NUM_FRAMES = 121  # 4k+1 for temporal compression ratio 4
MAX_HEIGHT = 512
MAX_WIDTH = 768
TRAIN_FPS = 24.0

MODEL_PATH = "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
# Overridable so automation (e.g. the nightly e2e test) can point at its own
# test-owned directories instead of clobbering the documented default paths
# a user may have populated with their real dataset.
DATA_DIR = os.environ.get("KANDINSKY5_OVERFIT_DATA_DIR", "data/kandinsky5_overfit")
OUTPUT_DIR = os.environ.get("KANDINSKY5_OVERFIT_OUTPUT_DIR", "data/kandinsky5_overfit_preprocessed")


def load_video(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Load video as [1, C, T, H, W] in [-1, 1], resized to (height, width).

    Source clips are frequently at native resolution (e.g. 1920x1080);
    encoding that directly through the VAE (instead of at the target
    training resolution) uses far more memory than intended and can OOM.
    """
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if not frames:
        # Without this, the repeat-last-frame fill below raises an opaque
        # IndexError on frames[-1]; cv2.VideoCapture doesn't raise on a
        # missing/corrupt file, it just decodes nothing.
        raise ValueError(f"Could not decode any frames from video {path!r} -- "
                         "the file is missing, empty, or not readable by OpenCV.")

    if len(frames) < num_frames:
        # Repeat last frame to fill
        while len(frames) < num_frames:
            frames.append(frames[-1])

    frames = frames[:num_frames]
    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video).float()
    video = video / 127.5 - 1.0  # [0,255] -> [-1,1]
    video = video.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,T,H,W]
    return video


def get_caption(item: dict) -> str:
    """Extract a caption from one ``videos2caption.json`` entry.

    The documented schema (``docs/training/data_preprocess.md``) stores
    ``cap`` as a plain string; some producers (e.g. this repo's own e2e
    test fixtures, mirroring ``preprocess_hunyuan_overfit.py``) instead
    store a non-empty list of caption variants and use the first one.
    Indexing unconditionally with ``item["cap"][0]`` silently takes the
    first *character* of a string caption instead of erroring, so accept
    and validate both forms explicitly here.
    """
    cap = item.get("cap")
    if isinstance(cap, str):
        if not cap:
            raise ValueError(f"Empty 'cap' string for entry: {item}")
        return cap
    if isinstance(cap, list):
        if not cap or not isinstance(cap[0], str) or not cap[0]:
            raise ValueError(f"'cap' list must be non-empty with a non-empty first string, got: {item}")
        return cap[0]
    raise ValueError(f"'cap' must be a string or a list of strings, got {type(cap).__name__}: {item}")


def main() -> None:
    # FastVideo's native text-encoder layers (e.g. CLIPAttention's
    # QKVParallelLinear) are tensor-parallel-aware and assert the TP process
    # group is initialized, even for a single-process/single-GPU run.
    maybe_init_distributed_environment_and_model_parallel(1, 1)

    device = torch.device("cuda:0")
    model_path = maybe_download_model(MODEL_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load captions. Validate up front (before spending minutes loading the
    # VAE + two text encoders): an empty/malformed manifest would otherwise
    # only surface as an IndexError on records[0] at parquet-write time.
    manifest_path = os.path.join(DATA_DIR, "videos2caption.json")
    with open(manifest_path) as f:
        caption_data = json.load(f)
    if not isinstance(caption_data, list) or not caption_data:
        raise ValueError(f"Manifest {manifest_path} must be a non-empty JSON list of "
                         f"{{'path', 'cap'}} entries, got: {type(caption_data).__name__} "
                         f"with {len(caption_data) if isinstance(caption_data, list) else 'n/a'} entries")

    pipeline_config = Kandinsky5T2VConfig()
    # Kandinsky5T2VConfig.__post_init__ sets load_encoder=False by default --
    # T2V inference only ever decodes generated latents, never encodes real
    # video. Preprocessing needs the encoder to turn real clips into latents.
    pipeline_config.vae_config.load_encoder = True
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pipeline_config=pipeline_config,
    )
    fastvideo_args.device = device

    # --- Load VAE (shared with HunyuanVideo) ---
    print("Loading Kandinsky5 VAE...")
    vae = VAELoader().load(os.path.join(model_path, "vae"), fastvideo_args)
    vae = vae.to(device=device, dtype=torch.float16).eval()
    print(f"VAE loaded ({sum(p.numel() for p in vae.parameters()) / 1e6:.0f}M)")

    # --- Load text encoders ---
    print("Loading Qwen/Reason1 text encoder...")
    qwen_enc = TextEncoderLoader().load(
        os.path.join(model_path, "text_encoder"),
        fastvideo_args,
    ).to(device).eval()
    qwen_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    qwen_tok_kwargs = dict(pipeline_config.text_encoder_configs[0].tokenizer_kwargs)
    # TextEncodingStage overrides max_length from pipeline_config.text_encoder_max_lengths
    # at runtime (fastvideo/pipelines/stages/text_encoding.py) -- Reason1Config's static
    # tokenizer_kwargs default (text_len=512) doesn't account for the 129-token Kandinsky
    # system template ENCODE_START_IDX strips off afterwards, so mirror the runtime value
    # here or training conditions on fewer caption tokens than inference does.
    qwen_tok_kwargs["max_length"] = pipeline_config.text_encoder_max_lengths[0]

    print("Loading CLIP text encoder...")
    clip_enc = TextEncoderLoader().load(
        os.path.join(model_path, "text_encoder_2"),
        fastvideo_args,
    ).to(device).eval()
    clip_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer_2"))
    clip_tok_kwargs = dict(pipeline_config.text_encoder_configs[1].tokenizer_kwargs)
    clip_tok_kwargs["max_length"] = pipeline_config.text_encoder_max_lengths[1]

    # --- Process each video ---
    records = []
    for item in caption_data:
        video_name = item["path"]
        caption = get_caption(item)
        video_path = os.path.join(DATA_DIR, "videos", video_name)

        print(f"\nProcessing: {video_name}")
        print(f"  Caption: {caption[:80]}...")

        # Encode video. Kandinsky5's VAE encode returns channel-first
        # [B, C, T, H, W], matching the storage convention Kandinsky5Model
        # expects (it permutes to channel-last only right before calling
        # the transformer).
        video = load_video(video_path, NUM_FRAMES, MAX_HEIGHT, MAX_WIDTH).to(device=device, dtype=torch.float16)
        print(f"  Video shape: {video.shape}")

        with torch.no_grad():
            latent_dist = vae.encode(video)
            latent = latent_dist.mean.squeeze(0).float().cpu()  # [C, T, H, W]
        print(f"  Latent shape: {latent.shape}")

        # Encode text with Qwen/Reason1.
        qwen_text = kandinsky5_qwen_preprocess_text(caption)
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            qwen_inputs = qwen_tok(qwen_text, **qwen_tok_kwargs).to(device)
            qwen_out = qwen_enc(
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                output_hidden_states=True,
            )
            qwen_embeds, _qwen_mask = kandinsky5_qwen_postprocess_text(qwen_out, qwen_inputs.attention_mask)
            qwen_embeds = qwen_embeds.squeeze(0)  # [seq, dim]

        # Encode text with CLIP.
        clip_text = preprocess_text(caption)
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            clip_inputs = clip_tok(clip_text, **clip_tok_kwargs).to(device)
            clip_out = clip_enc(
                input_ids=clip_inputs.input_ids,
                attention_mask=clip_inputs.attention_mask,
            )
            clip_pooled = kandinsky5_clip_postprocess_text(clip_out).squeeze(0)  # [768]

        # Combine: [pooled_clip_row, qwen_embeds]
        qwen_dim = qwen_embeds.shape[-1]
        pooled_row = torch.zeros(qwen_dim, device=device, dtype=torch.float16)
        pooled_row[:clip_pooled.shape[-1]] = clip_pooled
        text_embedding = torch.cat(
            [pooled_row.unsqueeze(0), qwen_embeds],
            dim=0,
        ).float().cpu()  # [seq+1, dim]
        print(f"  Text embedding shape: {text_embedding.shape}")

        record = {
            "id": video_name,
            "vae_latent_bytes": latent.numpy().tobytes(),
            "vae_latent_shape": list(latent.shape),
            "vae_latent_dtype": str(latent.dtype).replace("torch.", ""),
            "text_embedding_bytes": text_embedding.numpy().tobytes(),
            "text_embedding_shape": list(text_embedding.shape),
            "text_embedding_dtype": str(text_embedding.dtype).replace("torch.", ""),
            "file_name": video_name,
            "caption": caption,
            "media_type": "video",
            "width": MAX_WIDTH,
            "height": MAX_HEIGHT,
            "num_frames": NUM_FRAMES,
            "duration_sec": NUM_FRAMES / TRAIN_FPS,
            "fps": TRAIN_FPS,
        }
        records.append(record)

    # Clean up encoders
    del qwen_enc, qwen_tok, clip_enc, clip_tok, vae

    # Write parquet
    table = pa.table(
        {k: [r[k] for r in records]
         for k in records[0]},
        schema=pyarrow_schema_t2v,
    )
    output_path = os.path.join(OUTPUT_DIR, "data_00000.parquet")
    pq.write_table(table, output_path)
    print(f"\nWrote {len(records)} records to {output_path}")

    # Write validation prompts for callback.
    # Wrap in "data" key -- ValidationDataset expects field="data".
    # Use "caption" field -- ValidationDataset aliases it to "prompt".
    val_prompts = {"data": [{"caption": get_caption(item)} for item in caption_data]}
    val_path = os.path.join(OUTPUT_DIR, "validation_prompts.json")
    with open(val_path, "w") as f:
        json.dump(val_prompts, f, indent=2)
    print(f"Wrote validation prompts to {val_path}")

    print("\nDone! Use data_path: " + OUTPUT_DIR + " in training config.")


if __name__ == "__main__":
    main()
