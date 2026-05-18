# Matrix-Game 3.0 in FastVideo — Onboarding & Resume Guide

> Branch: `feat/kaiqin/add-mg-3` in `/home/hal-kaiqin/FastVideo_mg`.
> Last sync with `origin/main`: merge commit `9146d1eb` (this session, 2026-05-14).
> Status: code-side integration is mostly in. Inference still blocked on a Diffusers-format model directory + conversion script.

## 1. Environment

```bash
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_BIN="$CONDA_ROOT/bin/conda"
ENV_NAME="${ENV_NAME:-FastVideo_kaiqin}"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_NAME"
export PYTHONPATH=/home/hal-kaiqin/FastVideo_mg
```

- 4× NVIDIA GB200, ~188 GB free each.
- Do **not** reinstall the env or rebuild kernels; just `export PYTHONPATH`.

## 2. What this session did

1. **Merged `origin/main` into `feat/kaiqin/add-mg-3`.** Conflicts were:
   - `fastvideo/configs/sample/wan.py` → deleted (followed main; `configs/sample/` is gone).
   - `fastvideo/configs/sample/matrixgame.py` → auto-renamed to `fastvideo/api/matrixgame.py`; import fixed to `from fastvideo.api.sampling_param import SamplingParam`.
   - `fastvideo/configs/pipelines/__init__.py` → switched LTX2 import to `pipelines/basic/ltx2/pipeline_configs`, kept MG imports.
   - `fastvideo/pipelines/stages/__init__.py` → LTX2 imports moved to `pipelines/basic/ltx2/stages`; kept `MatrixGame2CausalDenoisingStage, MatrixGame3DenoisingStage`.
   - `fastvideo/registry.py` → dropped all the dead `configs.sample.*` imports; imports `MatrixGame{2,3}SamplingParam` from `fastvideo.api.matrixgame`; gave the MG2 entry `workload_types=(WorkloadType.I2V,)`/`model_family`/`default_preset` to match main's shape; added MG3 entry (`FastVideo/Matrix-Game-3.0-Diffusers`, `model_family="matrixgame"`, `default_preset="matrixgame_i2v"`).
   - `fastvideo/pipelines/preprocess/matrixgame/matrixgame_preprocess_pipeline_ode_trajectory.py` → renamed `MatrixGameCausalDenoisingStage` → `MatrixGame2CausalDenoisingStage`; import switched to `fastvideo.api.sampling_param`.

   `fastvideo/configs/pipelines/wan.py` auto-merged cleanly; main removed its old `MatrixGameI2V480PConfig` block and we now own that surface in `fastvideo/configs/pipelines/matrixgame.py`.

2. **Sanity checks that pass (no model weights touched):**
  
```python
   from fastvideo.registry import _MODEL_HF_PATH_TO_NAME
   from fastvideo.configs.pipelines.matrixgame import MatrixGame3I2V720PConfig
   from fastvideo.pipelines.basic.matrixgame.matrixgame3_i2v_pipeline import MatrixGame3I2VPipeline
   ```
  
MG3 path registered: `FastVideo/Matrix-Game-3.0-Diffusers`. MG2 paths still registered. No conflict markers anywhere.

1. **Architecture sanity check against the local raw checkpoint** (`/home/hal-kaiqin/models/Matrix-Game-3.0/base_model/diffusion_pytorch_model.safetensors`):
   - With the **5B** MG3 arch values (`num_attention_heads=24, attention_head_dim=128, num_layers=30, ffn_dim=14336, in_channels=48, out_channels=48, text_dim=4096, action_config['img_hidden_size']=3072, action_config['keyboard_dim_in']=6, action_config['blocks']=range(15)`), `MatrixGame3WanModel` instantiates as a 6.47 B-parameter model and **all 1356 checkpoint keys map cleanly** through `param_names_mapping` — 0 unmapped, 0 missing on the model side.
   - **Caveat:** the defaults in `fastvideo/configs/models/dits/matrixgame.py::MatrixGame3WanVideoArchConfig` are still the 28B-MoE values (`num_attention_heads=40, num_layers=40, ffn_dim=13824, img_hidden_size=5120, keyboard_dim_in=4, blocks=range(40)`). `update_model_arch()` won't rescue the load from raw `base_model/config.json` because that JSON uses official-style keys (`dim`, `num_heads`, `in_dim`, `out_dim`) that don't match the diffusers-style field names on the dataclass. See §5.1.

## 3. Where the MG3 code lives

| Concern | File |
|---|---|
| DiT model | `fastvideo/models/dits/matrixgame3/{__init__.py, model.py, action_module.py, utils.py}` (`EntryClass = [MatrixGame3WanModel]`) |
| Arch + pipeline configs | `fastvideo/configs/models/dits/matrixgame.py` (`MatrixGame3WanVideoArchConfig`, `MatrixGame3WanVideoConfig`); `fastvideo/configs/pipelines/matrixgame.py` (`MatrixGame3I2V720PConfig`, subclasses `WanT2V480PConfig`) |
| Sampling param | `fastvideo/api/matrixgame.py` (`MatrixGame3SamplingParam`: 720p, 1280×720, 57 frames, `num_iterations`, `use_base_model`) |
| Inference pipeline | `fastvideo/pipelines/basic/matrixgame/matrixgame3_i2v_pipeline.py` (`MatrixGame3I2VPipeline`) |
| Denoise stage | `fastvideo/pipelines/stages/matrixgame_denoising.py::MatrixGame3DenoisingStage` (autoregressive multi-iteration with `clip_frame=56`, `past_frame=16`, Plücker memory) |
| Image latent stage | `fastvideo/pipelines/stages/image_encoding.py::MatrixGame3ImageVAEEncodingStage` |
| Model registry | `fastvideo/models/registry.py` — `MatrixGame3WanModel` mapped to `("dits", "matrixgame3", "MatrixGame3WanModel")` |
| Config registry | `fastvideo/registry.py::_register_configs` (MG3 entry, `default_preset="matrixgame_i2v"`) |
| Preset | `fastvideo/pipelines/basic/matrixgame/presets.py` (one shared `matrixgame_i2v` preset) |
| ForwardBatch fields | `fastvideo/pipelines/pipeline_batch_info.py:134-135` — `num_iterations`, `use_base_model` already wired |

Reference (official) upstream is **untracked** at `Matrix-Game/` (do not commit). Trust path: `Matrix-Game/Matrix-Game-3/generate.py`, `pipeline/inference_pipeline.py`, `wan/configs/config.py`, `wan/modules/model.py`.

## 4. Local model layout (raw HF download)

```
/home/hal-kaiqin/models/Matrix-Game-3.0/
├── model_index.json                       # only {"_class_name": "MatrixGame3I2VPipeline"} — incomplete for FastVideo's verify_model_config_and_directory
├── base_model/                            # 5B, dim=3072, num_layers=30, num_heads=24, ffn_dim=14336, in_dim=out_dim=48; 13 GB safetensors
│   ├── config.json                        # official-style keys (dim/num_heads/in_dim/...) — not diffusers
│   └── diffusion_pytorch_model.safetensors
├── base_distilled_model/                  # distilled 5B, 26 GB safetensors, same arch
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── Wan2.2_VAE.pth                         # OrderedDict with encoder.*/decoder.* keys (Wan-native, NOT diffusers AutoencoderKLWan)
├── MG-LightVAE.pth, MG-LightVAE_v2.pth    # alt VAE checkpoints
├── models_t5_umt5-xxl-enc-bf16.pth        # T5/UMT5-XXL encoder, .pth (not transformers format)
└── google/umt5-xxl/                       # tokenizer files (special_tokens_map.json, spiece.model, tokenizer.json, tokenizer_config.json)
```

`FastVideo_mg/models/` (sibling to `Matrix-Game/`) was created and is currently empty — this is the intended drop target for the Diffusers-format MG3 dir once conversion lands.

## 5. What's left to finish

### 5.1 Diffusers-format MG3 model dir (blocker for any inference run)

`fastvideo.utils.verify_model_config_and_directory` insists on: `model_index.json` with `_diffusers_version`, a `transformer/` subdir, and a subdir for every non-tokenizer component declared in `model_index.json`. The current raw download has none of that wiring.

Target layout (mirror what `/home/hal-kaiqin/models/Matrix-Game-2.0-Base-Diffusers/` looks like):

```
models/Matrix-Game-3.0-Diffusers/
├── model_index.json                        # _class_name MatrixGame3I2VPipeline, scheduler/transformer/vae/text_encoder entries, _diffusers_version "0.33.1"
├── transformer/
│   ├── config.json                         # _class_name "MatrixGame3WanModel", diffusers-style keys
│   └── diffusion_pytorch_model.safetensors # base_model or base_distilled_model weights (key names already match param_names_mapping; no rewrite needed)
├── vae/
│   ├── config.json                         # _class_name "AutoencoderKLWan", Wan2.2 5B-VAE dims
│   └── diffusion_pytorch_model.safetensors # converted from Wan2.2_VAE.pth
├── scheduler/scheduler_config.json         # FlowUniPCMultistepScheduler (matches MatrixGame3I2VPipeline.initialize_pipeline)
├── text_encoder/
│   ├── config.json                         # _class_name "UMT5EncoderModel"
│   └── model.safetensors                   # converted from models_t5_umt5-xxl-enc-bf16.pth
└── tokenizer/                              # copy of google/umt5-xxl/
```

**Open conversion subtasks** (none of these have been written yet — `scripts/checkpoint_conversion/` has no `matrix_game_3*.py`):

1. **Transformer config translator.** Read `base_model/config.json`, emit `transformer/config.json` with diffusers keys:
   - `dim → hidden_size`, `num_heads → num_attention_heads`, `attention_head_dim = dim // num_heads`, `num_layers → num_layers`, `ffn_dim → ffn_dim`, `in_dim → in_channels`, `out_dim → out_channels`, `text_len`, `patch_size`, plus pass `action_config`, `sigma_theta`, `is_action_model` straight through.
   - Add `"_class_name": "MatrixGame3WanModel"`, `"_diffusers_version": "0.33.1"`.
   - Set `"text_dim": 4096` (UMT5-XXL hidden) — currently missing from the raw config.
   - Also set `"image_dim": 0` (MG3 has no separate image encoder) and `"use_memory": true`, `"camera_embed_in_channels": 1536` if you want memory mode enabled by default.
   - Symlink the safetensors instead of copying: `ln -s /home/hal-kaiqin/models/Matrix-Game-3.0/base_model/diffusion_pytorch_model.safetensors transformer/diffusion_pytorch_model.safetensors`. Key names already pass through `MatrixGame3WanVideoArchConfig.param_names_mapping` — verified this session by mapping all 1356 keys.

2. **Wan2.2 VAE → AutoencoderKLWan.** Existing `scripts/checkpoint_conversion/wan_to_diffusers.py` handles the transformer half. The VAE half is missing for Wan2.2 specifically; the closest reference is the `AutoencoderKLWan` loader in `fastvideo/models/vaes/wanvae.py` (and `fastvideo/configs/models/vaes/wanvae.py` for `WanVAEConfig`). Need to map `encoder.conv1.*`, `encoder.downsamples.*.residual.*`, `decoder.*` etc. to the diffusers `AutoencoderKLWan` state dict.
   - MG3's `MatrixGame3I2V720PConfig.__post_init__` flips both `vae_config.load_encoder` and `load_decoder` on, so both halves of the VAE must be in the safetensors.
   - Light VAE alternatives (`MG-LightVAE.pth`, `MG-LightVAE_v2.pth`) aren't wired in FastVideo yet — defer until after the full Wan VAE path runs.

3. **UMT5-XXL .pth → safetensors.** `models_t5_umt5-xxl-enc-bf16.pth` is the encoder portion only; `_TEXT_ENCODER_MODELS["UMT5EncoderModel"]` resolves to `fastvideo/models/encoders/t5.py::UMT5EncoderModel`. Need to confirm key-name parity then `safetensors.torch.save_file(state_dict, "text_encoder/model.safetensors")` and write a matching `text_encoder/config.json`. Tokenizer is a straight `cp -r google/umt5-xxl/ tokenizer/`.

4. **Scheduler.** Just write `scheduler/scheduler_config.json` with `{"_class_name": "FlowUniPCMultistepScheduler", "_diffusers_version": "0.33.1", "shift": 5.0, ...}`. `MatrixGame3I2VPipeline.initialize_pipeline` re-instantiates the scheduler anyway, but the file has to exist for `verify_model_config_and_directory` to be happy.

5. **`model_index.json`.** Currently a stub. Replace with the multi-component dict (see MG2's `/home/hal-kaiqin/models/Matrix-Game-2.0-Base-Diffusers/model_index.json` for the exact shape — same idea, swap class names).

The cleanest place to land all this is `scripts/checkpoint_conversion/matrix_game_3_to_diffusers.py`. Drop the output under `/home/hal-kaiqin/FastVideo_mg/models/Matrix-Game-3.0-Diffusers/`.

### 5.2 `MatrixGame3WanVideoArchConfig` defaults

`fastvideo/configs/models/dits/matrixgame.py:120-153` currently encodes the 28 B-MoE arch. For the public 5B release this is wrong, and `update_model_arch` won't fix it because diffusers field names don't match the raw config keys.

Two safe options, in order of preference:

- **Translator-only:** keep the dataclass as-is (still useful for the future 28B), and have the conversion script in §5.1 emit `transformer/config.json` with diffusers-style field names (`num_attention_heads`, `attention_head_dim`, `num_layers`, `ffn_dim`, `in_channels`, `out_channels`, `image_dim`, `text_dim`). `update_model_arch` will then overwrite the 28B defaults with the 5B values at load time. **No source change to FastVideo is needed.**
- **Default flip:** change the defaults to 5B and add a second `MatrixGame3_28B_WanVideoArchConfig` if/when a 28B-MoE checkpoint surfaces. Simpler at load time but commits us to the 5B as the canonical default.

Either way, also bump:
- `action_config['blocks']` to `range(15)` for the 5B (the raw `base_model/config.json` lists 15 action blocks, not 40).
- `action_config['keyboard_dim_in']` to `6` (the safetensor `blocks.0.action_model.keyboard_embed.0.weight` is `[128, 6]`).
- `action_config['img_hidden_size']` to the model's `hidden_size`.

### 5.3 Example & CLI scripts

- `examples/inference/basic/basic_matrixgame3.py` — there is **no** MG3 example yet (`basic_matrixgame.py` covers MG2 only). Mirror its structure but with `MatrixGame3SamplingParam` defaults and an `image_path` + `prompt` (MG3 needs text conditioning, MG2 doesn't).
- `examples/inference/basic/basic_matrixgame_streaming.py` is MG2-streaming; an MG3 streaming variant is a stretch goal.

### 5.4 Hardcoded values to revisit

- `fastvideo/pipelines/stages/matrixgame_denoising.py:827` — `clip_frame = 56  # hardcode for now`. Tied to MG3's 57-frame clip window; fine for now but a follow-up should pull from the pipeline config.
- `fastvideo/models/dits/matrixgame3/action_module.py:553` — `## TODO: adding cache here` (action-module KV cache). Not on the critical path for first-run inference.

## 6. Smoke-test cookbook

After the diffusers dir is built (§5.1), this should be enough to verify nothing rotted in the merge:

```python
import torch
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "/home/hal-kaiqin/FastVideo_mg/models/Matrix-Game-3.0-Diffusers",
    num_gpus=1,
    use_fsdp_inference=False,
    dit_cpu_offload=False,
    vae_cpu_offload=False,
    text_encoder_cpu_offload=True,
)

generator.generate_video(
    prompt="A first-person view walking forward through a forest",
    image_path="Matrix-Game/Matrix-Game-3/demo_images/<pick one>.png",
    num_frames=57,
    height=720,
    width=1280,
    num_inference_steps=50,
    guidance_scale=5.0,
    output_path="output_mg3",
    save_video=True,
)
```

Re-run the key-mapping check (no GPU needed) any time the arch config changes:

```bash
python - <<'PY'
import re
from safetensors import safe_open
from fastvideo.configs.models.dits.matrixgame import MatrixGame3WanVideoArchConfig, MatrixGame3WanVideoConfig
from fastvideo.models.dits.matrixgame3.model import MatrixGame3WanModel

arch = MatrixGame3WanVideoArchConfig(num_attention_heads=24, attention_head_dim=128, num_layers=30,
                                     ffn_dim=14336, in_channels=48, out_channels=48, text_len=512, image_dim=0)
arch.text_dim = 4096
ac = dict(arch.action_config); ac.update(dict(blocks=list(range(15)), keyboard_dim_in=6, img_hidden_size=3072,
                                              mouse_qk_dim_list=[8,28,28]))
arch.action_config = ac
arch.__post_init__()
cfg = MatrixGame3WanVideoConfig(); cfg.arch_config = arch
m = MatrixGame3WanModel(config=cfg, hf_config={})
model_keys = set(m.state_dict().keys())

unmapped, missing = [], set(model_keys)
with safe_open("/home/hal-kaiqin/models/Matrix-Game-3.0/base_model/diffusion_pytorch_model.safetensors", framework="pt") as f:
    for k in f.keys():
        nk = k
        for pat, repl in cfg.param_names_mapping.items():
            new = re.sub(pat, repl, nk)
            if new != nk: nk = new; break
        if nk in model_keys: missing.discard(nk)
        else: unmapped.append((k, nk))
print("ckpt unmapped:", len(unmapped), "model unfilled:", len(missing))
PY
```

Expected output: `ckpt unmapped: 0 model unfilled: 0`.

## 7. Gotchas baked in by `origin/main` worth remembering

- `fastvideo/configs/sample/` is gone. Per-model sampling-param dataclasses now live under `fastvideo/api/<model>.py` and subclass `fastvideo.api.sampling_param.SamplingParam`. We followed that convention with `fastvideo/api/matrixgame.py`.
- LTX2 stages and pipeline configs moved to `fastvideo/pipelines/basic/ltx2/`. If you re-pull main, do not "fix" stale `fastvideo/pipelines/stages/ltx2_*` imports — they're meant to be gone.
- Main's MG2 detector used a loose `"matrix-game" in path.lower() or "matrixgame" in path.lower()` lambda which also matched MG3. We tightened it to the explicit `"matrix-game-2"`/`"matrixgame2"`/`"matrix-game-2.0"` triple and gave MG3 its own detector. Keep them in this order in `_register_configs` (MG2 first, then MG3) so detectors stay unambiguous.

## 8. Memory pointers from prior sessions (still relevant)

- `~/.claude/projects/.../memory/project_fastvideo_editing.md` — image-driven video editing context (orthogonal to MG3 but same repo).
- `~/.claude/projects/.../memory/feedback_minimal_changes.md` — preference for smallest possible diffs, native PyTorch hooks over abstractions.
- `~/.claude/projects/.../memory/reference_fastvideo_worker_rpc.md` — for running arbitrary code in the FastVideo worker via `MultiprocExecutor.collective_rpc` without editing FastVideo.

## 9. Suggested resume order

1. Implement `scripts/checkpoint_conversion/matrix_game_3_to_diffusers.py` (§5.1).
2. Build `models/Matrix-Game-3.0-Diffusers/` for the **non-distilled** `base_model` first (fewer surprises than the distilled one).
3. Run the smoke test in §6. Expect a parade of small issues — bad VAE keys, missing `text_dim` in `transformer/config.json`, scheduler args mismatch — fix them one at a time.
4. Once the non-distilled base is producing video, swap to `base_distilled_model` and tune `num_iterations` / `use_base_model`.
5. Add `examples/inference/basic/basic_matrixgame3.py` (§5.3) once the pipeline is reproducible.
6. Only then think about `MG-LightVAE_v2.pth`, INT8 quant, FSDP, or streaming.
