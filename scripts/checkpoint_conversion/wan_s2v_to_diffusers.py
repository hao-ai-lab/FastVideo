# SPDX-License-Identifier: Apache-2.0
"""Repackage Wan-AI/Wan2.2-S2V-14B into the Diffusers layout FastVideo loads.

The official repo is flat (config.json + shards at the top level, VAE and text
encoder as .pth files), so ``VideoGenerator.from_pretrained`` rejects it: there
is no model_index.json and no transformer/ directory.

This script only moves files around -- it does NOT rename any tensor. The DiT
shards keep their native Wan names (``blocks.0.self_attn.q``); FastVideo's
``WanS2VArchConfig.param_names_mapping`` reads that naming directly. The VAE,
text encoder, tokenizer and scheduler are taken from Wan-AI's own Diffusers
release of Wan2.1 (same published weights, already in the right layout), and
the wav2vec2 audio encoder is copied from the folder bundled inside the S2V
repo itself.

After writing, the script re-checks the result without a GPU: every one of the
1260 transformer tensors must map onto a FastVideo model parameter with the
right shape (the model is built on the meta device, so this costs no memory).

Usage:
    python wan_s2v_to_diffusers.py --output /path/to/Wan2.2-S2V-14B-Diffusers
    # then upload with create_hf_repo.py, and validate on a GPU box first
    # (see AGENTS.md: never push before the smoke test passes).
"""
import argparse
import json
import os
import re
import shutil
import struct

# Files from the official repo we deliberately do not carry over, and why.
SKIPPED_SOURCE_FILES = {
    "assets/": "README imagery, not model weights",
    "Wan2.1_VAE.pth": "replaced by the Diffusers-layout vae/ from Wan2.1-T2V-14B-Diffusers (same weights)",
    "models_t5_umt5-xxl-enc-bf16.pth": "replaced by text_encoder/ from Wan2.1-T2V-14B-Diffusers (same weights)",
    "google/": "replaced by tokenizer/ from Wan2.1-T2V-14B-Diffusers (same tokenizer)",
    "wav2vec2-large-xlsr-53-english/language_model/": "CTC language-model decoding assets; FastVideo only "
                                                      "uses the encoder's hidden states",
    "wav2vec2-large-xlsr-53-english/alphabet.json": "CTC decoding asset, unused",
    "wav2vec2-large-xlsr-53-english/eval.py": "upstream evaluation script, unused",
    "wav2vec2-large-xlsr-53-english/flax_model.msgpack": "Flax duplicate of the torch weights",
}

AUDIO_ENCODER_FILES = ("config.json", "model.safetensors", "pytorch_model.bin", "preprocessor_config.json")
AUX_COMPONENTS = ("vae", "text_encoder", "tokenizer", "scheduler")

MODEL_INDEX = {
    "_class_name": "WanSpeechToVideoPipeline",
    "_diffusers_version": "0.33.0.dev0",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanModel_S2V"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "audio_encoder": ["transformers", "Wav2Vec2Model"],
    "audio_processor": ["transformers", "Wav2Vec2FeatureExtractor"],
}


def _place(src: str, dst: str, link: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build(source: str, aux_source: str, output: str, link: bool) -> None:
    from huggingface_hub import snapshot_download

    if not os.path.isdir(source):
        # Only fetch what the conversion uses. The official repo also carries
        # ~13GB of .pth duplicates of the text encoder and VAE plus demo
        # assets; downloading those wastes half an hour and disk for nothing.
        source = snapshot_download(source,
                                   allow_patterns=[
                                       "diffusion_pytorch_model*",
                                       "config.json",
                                       "wav2vec2-large-xlsr-53-english/*",
                                   ])
    if not os.path.isdir(aux_source):
        aux_source = snapshot_download(aux_source, allow_patterns=[f"{c}/*" for c in AUX_COMPONENTS])

    os.makedirs(output, exist_ok=True)

    # 1. transformer/: shards + weight index verbatim (no tensor renames).
    for name in sorted(os.listdir(source)):
        if name.endswith(".safetensors") or name == "diffusion_pytorch_model.safetensors.index.json":
            _place(os.path.join(source, name), os.path.join(output, "transformer", name), link)

    # transformer/config.json: the official values, plus the class name the
    # FastVideo model registry resolves.
    with open(os.path.join(source, "config.json")) as f:
        transformer_config = json.load(f)
    transformer_config["_class_name"] = "WanModel_S2V"
    transformer_config.setdefault("_diffusers_version", MODEL_INDEX["_diffusers_version"])
    with open(os.path.join(output, "transformer", "config.json"), "w") as f:
        json.dump(transformer_config, f, indent=2, sort_keys=True)

    # 2. vae/ text_encoder/ tokenizer/ scheduler/ from the Wan2.1 Diffusers release.
    for component in AUX_COMPONENTS:
        src_dir = os.path.join(aux_source, component)
        assert os.path.isdir(src_dir), f"{aux_source} has no {component}/ directory"
        for root, _, files in os.walk(src_dir):
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), aux_source)
                _place(os.path.join(root, name), os.path.join(output, rel), link)

    # 3. audio_encoder/ + audio_processor/ from the bundled wav2vec2.
    wav2vec_dir = os.path.join(source, "wav2vec2-large-xlsr-53-english")
    assert os.path.isdir(wav2vec_dir), "official repo is missing its bundled wav2vec2 folder"
    for name in AUDIO_ENCODER_FILES:
        src = os.path.join(wav2vec_dir, name)
        if os.path.isfile(src):
            _place(src, os.path.join(output, "audio_encoder", name), link)
    _place(os.path.join(wav2vec_dir, "preprocessor_config.json"),
           os.path.join(output, "audio_processor", "preprocessor_config.json"), link)

    # 4. The manifest FastVideo's loader reads first.
    with open(os.path.join(output, "model_index.json"), "w") as f:
        json.dump(MODEL_INDEX, f, indent=2, sort_keys=True)


def validate(output: str) -> None:
    """Every transformer tensor must land on a FastVideo parameter, right shape.

    Runs on the meta device: no GPU, no memory cost, catches every naming or
    shape mistake before anything is uploaded.
    """
    import torch

    from fastvideo.configs.models.dits.wan_s2v import WanS2VArchConfig, WanS2VConfig
    from fastvideo.models.dits.wan_s2v import WanS2VTransformer3DModel

    shapes: dict[str, tuple] = {}
    transformer_dir = os.path.join(output, "transformer")
    for name in sorted(os.listdir(transformer_dir)):
        if not name.endswith(".safetensors"):
            continue
        with open(os.path.join(transformer_dir, name), "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for tensor_name, meta in header.items():
            if tensor_name != "__metadata__":
                shapes[tensor_name] = tuple(meta["shape"])

    mapping = WanS2VArchConfig().param_names_mapping

    def to_model_name(name: str) -> str:
        for pattern, repl in mapping.items():
            new, count = re.subn(pattern, repl, name)
            if count:
                return new
        return name

    with torch.device("meta"):
        model = WanS2VTransformer3DModel(config=WanS2VConfig(), hf_config={})
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    mapped = {to_model_name(k): v for k, v in shapes.items()}

    missing = sorted(set(model_shapes) - set(mapped))
    unexpected = sorted(set(mapped) - set(model_shapes))
    mismatched = sorted(k for k in set(mapped) & set(model_shapes) if mapped[k] != model_shapes[k])
    print(f"transformer tensors: {len(shapes)} | model parameters: {len(model_shapes)}")
    print(f"missing: {len(missing)} | unexpected: {len(unexpected)} | shape mismatches: {len(mismatched)}")
    if missing or unexpected or mismatched:
        for label, names in (("MISSING", missing), ("UNEXPECTED", unexpected), ("MISMATCH", mismatched)):
            for n in names[:5]:
                print(f"  {label}: {n}")
        raise SystemExit("validation FAILED -- do not upload this output")

    for required in ("model_index.json", "transformer/config.json", "vae/config.json",
                     "audio_encoder/config.json", "audio_processor/preprocessor_config.json"):
        assert os.path.isfile(os.path.join(output, required)), f"missing {required}"
    print("validation PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="Wan-AI/Wan2.2-S2V-14B",
                        help="Official checkpoint: HF repo id or a local snapshot directory")
    parser.add_argument("--aux-source", default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                        help="Where to take the vae/, text_encoder/, tokenizer/ and scheduler/ from "
                             "(Wan2.1's own Diffusers release ships the identical weights)")
    parser.add_argument("--output", required=True, help="Directory to write the Diffusers-layout repo into")
    parser.add_argument("--link", action="store_true",
                        help="Hardlink instead of copying where possible (saves ~28GB of disk)")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    build(args.source, args.aux_source, args.output, args.link)
    if not args.skip_validation:
        validate(args.output)
    print(f"\nwrote {args.output}")
    print("next: smoke-test a forward pass on a GPU box, then upload with create_hf_repo.py")


if __name__ == "__main__":
    main()
