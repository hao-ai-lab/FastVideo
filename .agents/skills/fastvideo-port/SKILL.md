---
name: fastvideo-port
description: End-to-end skill for porting a new video generation model pipeline into FastVideo — from repo recon through numerical alignment tests to a mergeable PR with SSIM coverage.
---

# FastVideo Port

## Purpose

Automate the full pipeline port workflow described in
`docs/contributing/coding_agents.md`. Given an official repo URL, this skill
produces a clean PR that:

1. Implements the model in FastVideo (DiT, VAE, text encoder(s)).
2. Converts or maps checkpoint weights to FastVideo naming.
3. Passes component-level numerical alignment tests (`atol=1e-4`).
4. Passes a pipeline-level SSIM test against official reference videos.
5. Includes user-facing example scripts and docs.

## Prerequisites

- FastVideo repo cloned; `uv pip install -e .[dev]` complete.
- `official_weights/<model_name>/` populated (download manually — large files
  time out in agents; see Step 0).
- Official repo cloned under `FastVideo/<model_name>/` (Step 0).
- GPU available with sufficient VRAM for both official + FastVideo models
  simultaneously (needed for alignment tests).
- `WANDB_API_KEY` set if validation logging is desired.
- `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` set during parity tests to avoid
  backend-specific numerical differences.

## Required credentials

Check these before starting. Missing tokens cause silent download failures
or access errors that are hard to debug mid-port.

| Credential | When needed | How to set |
|-----------|-------------|------------|
| `HF_TOKEN` | Gated HF models (Llama, Gemma, t5gemma, etc.) | `huggingface-cli login` or `export HF_TOKEN=...` |
| `STABILITY_API_KEY` | stable-audio-open-1.0 and other Stability AI models | `export STABILITY_API_KEY=...` |
| HF write access | Uploading converted weights to `fastvideo/` HF org | Ask Will for org access before Step 5 |
| GitHub token | `recon.py` hits GitHub API (60 req/hr unauth vs 5000 auth) | `export GITHUB_TOKEN=...` |

**Check gating status before downloading:**
```bash
python - <<'PY'
from huggingface_hub import model_info
for repo in ["<text_encoder_hf_id>", "<vae_hf_id>"]:
    info = model_info(repo)
    print(repo, "— gated:", getattr(info, "gated", False))
PY
```

If a model is gated and you don't have access, request it on HF before
proceeding — alignment tests cannot run without the weights.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `repo_url` | Yes | GitHub URL of the official model repo |
| `model_name` | No | Short identifier used for paths (e.g. `davinci-magihuman`); inferred from repo name if omitted |
| `tasks` | No | Which tasks to port: `t2v`, `i2v`, `v2v` (default: `t2v` first) |
| `hf_account` | No | HuggingFace org to publish converted weights (e.g. `fastvideo`) |
| `reference_video` | No | Path to reference video from official repo for SSIM comparison |
| `reference_prompt` | No | Text prompt used to generate the reference video |

## Recommended patterns

Before writing any code, note these opinionated defaults. They exist because
the alternatives were tried and caused problems.

| Decision | Recommended | Avoid | Why |
|----------|-------------|-------|-----|
| Training pipeline | Subclass `TrainingPipeline` directly | Wrapping or delegating | Wrapping hides override points; base class hooks (`_sample_timesteps`, `_build_input_kwargs`) only work via subclassing |
| Attention in DiT | `DistributedAttention` for self-attn, `LocalAttention` for cross-attn | Mixing or using raw `nn.MultiheadAttention` | FSDP2 sharding and SP only work with FastVideo attention wrappers |
| Projection layers | `ReplicatedLinear` | `nn.Linear` | `get_lora_layer()` is invisible to `nn.Linear`; LoRA silently has zero effect |
| Alignment backend | `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` | Flash/Triton during tests | Non-SDPA backends give numerically different outputs, making false failures |
| Port order | VAE → text encoder → DiT → pipeline | DiT first | VAE and encoder are simpler; getting them aligned first reduces variables when debugging the DiT |
| dtype during alignment | `float32` first, then `bfloat16` | `bfloat16` from the start | fp32 failures are easier to diagnose; bf16 can pass fp32 bugs through |

## Hard stop conditions

If you hit any of these, **stop immediately and fix the root cause** before
continuing. Going further just buries the problem.

| Symptom | What it means | What to do |
|---------|--------------|------------|
| `load_state_dict` reports >10% unexpected or missing keys | `param_names_mapping` is structurally wrong — probably a prefix mismatch | Re-run `auto_mapper.py`, diff key lists, fix prefix rules before alignment |
| Alignment test fails with dtype error (`expected BFloat16 but found Float`) | Models loaded in different dtypes | Check both `.to(dtype)` calls; don't proceed, you'll get false failures |
| Alignment `max_abs_diff > 0.1` on first layer | Weights didn't load correctly, not a code bug | Print `state_dict` norms on both sides; if they differ, the mapping is wrong |
| SSIM < 0.5 despite passing alignment tests | Pipeline wiring bug — noising/denoising schedule mismatch | Check scheduler shift, sigma format, and timestep range before debugging model code |
| `get_lora_layer()` returns `None` on any projection | `nn.Linear` used instead of `ReplicatedLinear` | Fix before any training run; alignment tests won't catch this |
| Official repo inference crashes in reference video generation | Environment / dependency issue | Fix this first — you need a valid reference video before SSIM means anything |

## Steps

### 0. Reconnaissance — understand the model before writing code

**First, run the prereq check:**
```bash
bash .agents/skills/fastvideo-port/scripts/check_prereqs.sh \
    --model <model_name> \
    --hf_ids "<text_encoder_hf_id> <vae_hf_id>"
```
Fix any hard failures before continuing. Warnings are fine to proceed with.

**Then run `recon.py`** to get a structured JSON summary without manually reading
the repo:

```bash
python .agents/skills/fastvideo-port/scripts/recon.py $ARGUMENTS \
    --output recon_<model_name>.json
cat recon_<model_name>.json
```

**Manually verify the recon output** — several fields are regex-derived and
frequently wrong. For each model file in `recon.model_files`, fetch and read
it (use `gh_raw` in recon.py, or read the GitHub URL directly). Cross-check:

| Field | Common failure | How to verify |
|-------|---------------|---------------|
| `num_layers` | Regex hits a loop variable (`range(1)`) not the config | Search for `num_layers`, `depth`, `n_layers` in config/init args |
| `class_name` | Hits a helper class, not the top-level DiT | Find the class that takes `x`, `timestep`, `encoder_hidden_states` as forward args |
| `text_encoders` | Over-matches (e.g. both "t5" and "t5gemma" as separate entries) | Read the encoder loading code — one model ID is actually used |
| `scheduler.type` | README mentions DDIM in background context | Check the actual noise schedule in the sampling loop |
| `hf_repo` | Matches a HuggingFace Space, not a weights repo | Verify it has model card + weight files, not just a demo |

After reading the source, correct `recon_<model_name>.json` manually and save
it — the corrected JSON is the source of truth for Phase 1.

Answer all questions from `coding_agents.md § "questions to ask yourself"`:

**Checklist:**
- [ ] What is the DiT architecture? (layers, hidden dim, attention type,
      conditioning mechanism)
- [ ] **Is the model timestep-free?** (no `adaln`/`time_embed` in DiT forward) — if so, `DenoisingStage` won't work; create a custom stage. See lesson `2026-04-22_timestep-free-dit-needs-custom-denoising-stage.md`.
- [ ] What text encoder(s) are used? (T5, CLIP, VLM, custom)
- [ ] What VAE is used? (latent channels, z_dim, spatial/temporal compression)
- [ ] **Does the VAE publish per-channel normalization stats?** (`latent_mean`/`latent_std`) — if absent, stub zeros/ones and flag. See lesson `2026-04-22_vae-normalization-stats-may-not-exist-for-new-z-dim.md`.
- [ ] Is there a Diffusers-format HF repo? Run:
      `python scripts/huggingface/download_hf.py --repo_id <hf_id> --local_dir official_weights/<model_name>`
- [ ] If no HF repo: clone official repo and download raw checkpoints manually.
- [ ] Does SGLang already have this model? Check
      `https://github.com/sgl-project/sglang`
- [ ] What tasks are supported? (T2V / I2V / V2V / audio-driven / etc.)
- [ ] Can you generate a reference video with the official repo?
      Save it to `assets/videos/<model_name>_reference.mp4`.

> **Stop here if weights are not available locally.** Flag to the user; this
> step requires manual downloads that can't be automated reliably.

**Create the branch:**
```bash
git checkout -b feat/<model_name>-port
```

**Clone official repo:**
```bash
git clone <repo_url> <model_name>/
```

---

### 1. Implement the model + config mapping

For each component (DiT, VAE, text encoder) in dependency order:

#### 1a. DiT (transformer)

- Create `fastvideo/models/dits/<model_name>.py`.
- Mirror the official architecture as closely as possible.
- Use FastVideo attention layers:
  - `DistributedAttention` for full-sequence self-attention in the DiT.
  - `LocalAttention` for cross-attention and all other attention.
  - `ReplicatedLinear` (not `nn.Linear`) for any projection layer that LoRA
    should be able to target.
- Create `fastvideo/configs/models/dits/<model_name>.py` with:
  - `param_names_mapping`: dict mapping regex patterns to replacement strings
    (Python insertion order = application order) that translate official
    `state_dict` keys → FastVideo keys. Example:
    ```python
    param_names_mapping: dict = field(default_factory=lambda: {
        r"^block\.layers\.(\d+)\.(.*)$": r"layers.\1.\2",
        r"^final_linear\.(.*)$": r"final_proj.\1",
    })
    ```

**Draft param_names_mapping from source (no weights needed):**

First, check if `param_names_mapping = {}` — if the official model already uses
the same attribute names as FastVideo, no rules are needed. Diff a few key names
to confirm before writing any.

Before running `auto_mapper.py` (which requires local weights), derive an
initial mapping by reading the official model source:

1. In the official repo, find the top-level model class. Note every `self.<attr>`
   that holds a sub-module (e.g. `self.block`, `self.layers`, `self.final_norm`).
   These become the key *prefixes* in `official.state_dict()`.
2. Do the same for your FastVideo class — those are the target key prefixes.
3. Write one regex rule per differing prefix, plus any leaf-name differences
   (e.g. `linear_video` → `proj_video`).
4. Common pattern — official uses descriptive container names that FastVideo
   flattens:
   - `block.layers.N.attn` → `layers.N.attn`
   - `video_embedder` → `video_proj`
   - `final_linear` → `final_proj`

This source-code pass typically covers 70–80% of keys. Run `auto_mapper.py`
for the remainder once weights are available.

**Quick key diff workflow:**
```bash
python - <<'PY'
import safetensors.torch as st, torch
official = st.load_file("official_weights/<model_name>/transformer/diffusion_pytorch_model.safetensors")
from fastvideo.models.dits.<model_name> import <ModelClass>
fv = <ModelClass>(<args>)
o_keys = set(official.keys())
fv_keys = set(fv.state_dict().keys())
print("MISSING IN FV:", sorted(o_keys - fv_keys)[:20])
print("EXTRA IN FV:",   sorted(fv_keys - o_keys)[:20])
PY
```

Use `auto_mapper.py` to generate a first-pass `param_names_mapping` (~80%
coverage) before doing manual iteration:

```bash
python .agents/skills/fastvideo-port/scripts/auto_mapper.py \
    --official official_weights/<model_name>/transformer/diffusion_pytorch_model.safetensors \
    --fastvideo_class fastvideo.models.dits.<model_name>.<ModelClass> \
    --fastvideo_args '{"hidden_size": 1024, "num_layers": 28}' \
    --output_py fastvideo/configs/models/dits/<model_name>_mapping_draft.py
```

Iterate on `param_names_mapping` until `load_state_dict(strict=True)` passes.

#### 1b. VAE

- If reusing an existing FastVideo VAE (e.g. `WanVAE`), add a wrapper or
  alias in `fastvideo/models/vaes/<model_name>.py` with the correct
  `param_names_mapping`.
- If it's a new architecture, implement it from scratch following the same
  pattern.

#### 1c. Text encoder(s)

- If using a standard encoder already in FastVideo (T5-XXL, CLIP, Qwen2.5-VL),
  add the config entry in `fastvideo/configs/models/encoders/`.
- If it's a novel encoder, add a class in `fastvideo/models/text_encoders/`.

---

### 1b. HuggingFace repo — resolve before alignment if no Diffusers format exists

If the model has no existing HF Diffusers repo, you need converted weights
before alignment tests can load the model. Do this now, not after pipeline.

```bash
python scripts/checkpoint_conversion/create_hf_repo.py \
  --model_path converted_weights/<model_name>/ \
  --output_dir hf_staging/<model_name>/

# Upload — requires write access to fastvideo/ HF org (ask Will first)
huggingface-cli upload fastvideo/<model_name> hf_staging/<model_name>/
```

If the model is already in Diffusers format, skip and download directly:
```bash
python scripts/huggingface/download_hf.py \
  --repo_id <hf_id> --local_dir official_weights/<model_name>
```

### 2. Numerical alignment tests — one component at a time

Run alignment tests **before** integrating components into the pipeline.
A failing parity test is much easier to debug than a failing end-to-end run.

Use the template at `.agents/skills/fastvideo-port/scripts/alignment_test.py`.

**Per-component test protocol:**

```
For each component (DiT, VAE encoder, VAE decoder, each text encoder):
  1. Load official model weights into official class.
  2. Load same weights into FastVideo class via param_names_mapping.
  3. Generate identical random inputs (fixed seed, same dtype + device).
  4. Run forward pass on both.
  5. Assert torch.testing.assert_close(fv_out, official_out, atol=1e-4, rtol=1e-4).
```

**Settings checklist — verify BEFORE running forward pass:**

| Setting | Required value | Why |
|---------|---------------|-----|
| Attention backend | `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` | Flash/custom backends give different numerics |
| dtype | Both models same dtype (`float32` first, then `bfloat16`) | Mixed dtypes cause false failures |
| `model.eval()` | Both models | Dropout/BN behave differently in train mode |
| `torch.manual_seed(42)` | Before every input generation | Reproducible inputs |
| Rotary/positional encoding | Same `max_seq_len`, `base` frequency | Subtle shape bugs |
| Normalization order | Check if pre-norm vs post-norm matches | Common porting mistake |
| HF token | `HF_TOKEN` set for gated components | Silent 401 errors during weight load |

```bash
# Verify both state_dicts loaded the same number of params
assert sum(p.numel() for p in official.parameters()) == \
       sum(p.numel() for p in fastvideo_model.parameters()), "param count mismatch"
```

**If parity fails:**
1. Add per-layer output logging (`register_forward_hook`) to both models.
2. Run forward pass; compare layer outputs in order.
3. First diverging layer reveals the root cause (wrong weight mapping, missing
   transpose, different normalization order, etc.).
4. See `coding_agents.md § "Common pitfalls"` and
   `.agents/lessons/` for documented failure modes.

Place tests in `tests/local_tests/<model_name>/`:
```
tests/local_tests/<model_name>/
├── README.md
├── test_dit_alignment.py
├── test_vae_alignment.py
└── test_text_encoder_alignment.py
```

---

### 3. Pipeline config + sample defaults

- Add `fastvideo/configs/pipelines/<model_name>.py` with component wiring.
- Add `fastvideo/configs/sample/<model_name>.py` with default sampling params
  (resolution, frames, steps, CFG scale, scheduler shift).
- Register both in `fastvideo/registry.py`:
  ```python
  register_configs(
      pipeline_config_cls=<ModelName>PipelineConfig,
      sample_config_cls=<ModelName>SamplingParam,
  )
  ```
- Add `model_index.json` to the converted weights directory.

---

### 4. Wire pipeline stages

- Create `fastvideo/pipelines/basic/<model_name>/<model_name>_pipeline.py`.
- Compose stages from `fastvideo/pipelines/stages/`:
  - `TextEncodingStage` (or a custom subclass for novel encoders)
  - `LatentPreparationStage` / `LatentDecodingStage`
  - `DenoiseStage`
- Add a `<ModelName>Pipeline.from_pretrained(model_path, ...)` classmethod
  following the pattern in `Cosmos2_5Pipeline` or `WanPipeline`.

---

### 5. HuggingFace repo (if no Diffusers repo exists)

If the official model does not ship in Diffusers format:

```bash
# Stage converted weights
python scripts/checkpoint_conversion/create_hf_repo.py \
  --model_path converted_weights/<model_name>/ \
  --output_dir hf_staging/<model_name>/

# Push to FastVideo HF account
huggingface-cli upload fastvideo/<model_name> hf_staging/<model_name>/
```

Update `model_index.json` to point to `fastvideo/<model_name>` on HF.

---

### 6. Pipeline-level parity test

Add `tests/local_tests/pipelines/test_<model_name>_pipeline.py`.

Minimal scaffold:
```python
def test_pipeline_parity():
    # 1. Generate video with official implementation (save as reference).
    # 2. Generate video with FastVideo pipeline (same prompt, same seed).
    # 3. Assert SSIM >= threshold across frames (see SSIM template below).
```

---

### 7. User-facing example

Add `examples/inference/basic/<model_name>/`:
```
examples/inference/basic/<model_name>/
├── basic.py          # Minimal generate-and-save example
├── lora.py           # LoRA fine-tuning inference (if applicable)
└── README.md
```

Run the example locally to confirm end-to-end behavior.

---

### 8. SSIM tests for CI

Add `fastvideo/tests/ssim/test_<model_name>_ssim.py` using the template at
`.agents/skills/fastvideo-port/scripts/ssim_test.py`.

**Protocol:**
1. Generate reference video with official repo at a fixed seed and prompt.
   Save to `assets/videos/<model_name>_reference.mp4`.
2. FastVideo generates video at same prompt + seed.
3. Compute per-frame SSIM; assert mean SSIM ≥ 0.85 (or model-specific threshold).
4. Document GPU requirement in the test docstring.

---

### 9. Pre-commit + PR

```bash
pre-commit run --all-files
pytest tests/local_tests/<model_name>/ -v
pytest fastvideo/tests/ssim/test_<model_name>_ssim.py -vs
```

Open a **DRAFT PR** after Step 1 (first component aligned), promote to
ready-to-review after Steps 8–9 pass.

PR description must include:
- Architecture summary (model family, params, tasks supported).
- Component alignment test results (atol, dtypes, any known divergences).
- SSIM result (mean ± std across test frames).
- Sample generated video (attach or link W&B run).
- Any open questions for the maintainer.

---

## Outputs

| Artifact | Path |
|----------|------|
| DiT implementation | `fastvideo/models/dits/<model_name>.py` |
| VAE implementation | `fastvideo/models/vaes/<model_name>.py` |
| Arch configs | `fastvideo/configs/models/dits/<model_name>.py` |
| Pipeline config | `fastvideo/configs/pipelines/<model_name>.py` |
| Sample defaults | `fastvideo/configs/sample/<model_name>.py` |
| Pipeline class | `fastvideo/pipelines/basic/<model_name>/<model_name>_pipeline.py` |
| Alignment tests | `tests/local_tests/<model_name>/` |
| SSIM test | `fastvideo/tests/ssim/test_<model_name>_ssim.py` |
| Example script | `examples/inference/basic/<model_name>/basic.py` |
| Reference video | `assets/videos/<model_name>_reference.mp4` |
| HF repo (optional) | `fastvideo/<model_name>` on HuggingFace |

---

## Worked examples

See `.agents/skills/fastvideo-port/examples/` for full worked examples.
- `examples/cosmos2_5.md` — standard cross-attention DiT; `net.*` prefix
  stripping, AdaLN-LoRA, Reason1 VLM encoder with layer-concat postprocessing.
- `examples/davinci-magihuman.md` — 15B unified Transformer with audio
  conditioning, sandwich weight sharing, MoELinear stacked weights, and
  two-stage inference.

## Known pitfalls

Read all files in `.agents/lessons/` before starting. Key ones:

| Lesson file | TL;DR |
|-------------|-------|
| `2026-04-09_lora-requires-replicated-linear.md` | `nn.Linear` is invisible to LoRA |
| `2026-04-09_fp32-bf16-dtype-mismatch-at-projection.md` | cast at projection when `--dit_precision fp32` |
| `2026-04-09_hardcoded-text-encoder-shape-in-utils.md` | hardcoded (512, 4096) breaks non-T5-XXL |
| `2026-04-09_multimodal-processor-needs-inner-tokenizer.md` | unwrap `processor.tokenizer` |
| `2026-04-09_bf16-embeddings-need-float-before-numpy.md` | `.cpu().float().numpy()` not `.cpu().numpy()` |
| `2026-04-09_preprocessing-entrypoint-not-always-ported.md` | check `v1_preprocessing_new.py` |
| `2026-04-09_scheduler-shift-not-wired-to-training.md` | `train()` overwrites scheduler |
| `2026-04-20_moe-stacked-weights-must-match-official-layout.md` | mirror stacked `[N*out, in]` for checkpoint compat |
| `2026-04-22_timestep-free-dit-needs-custom-denoising-stage.md` | timestep-free DiT → custom denoising stage required |
| `2026-04-22_vae-normalization-stats-may-not-exist-for-new-z-dim.md` | stub zeros/ones when VAE stats not published |

## Eval harness

To improve this skill after completing a port, see `eval-harness.md`.

## References

- `docs/contributing/coding_agents.md` — full porting workflow
- `docs/design/overview.md` — pipeline architecture, config system
- `.agents/skills/fastvideo-port/scripts/alignment_test.py` — parity test template
- `.agents/skills/fastvideo-port/scripts/ssim_test.py` — SSIM test template
- `.agents/lessons/` — known pitfalls and fixes
- `fastvideo/training/cosmos2_5_training_pipeline.py` — recent port example
- `tests/local_tests/` — existing alignment test examples
- `scripts/checkpoint_conversion/create_hf_repo.py` — HF repo creation helper
- `scripts/huggingface/download_hf.py` — HF download helper

<!-- changelog in git log -->
