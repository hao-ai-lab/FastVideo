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

## Steps

### 0. Reconnaissance — understand the model before writing code

Run `recon.py` first to get a structured JSON summary without manually reading
the repo:

```bash
python .agents/skills/fastvideo-port/scripts/recon.py $ARGUMENTS \
    --output recon_<model_name>.json
cat recon_<model_name>.json
```

Then verify/supplement the output by reading the README and key model files.
Answer all questions from `coding_agents.md § "questions to ask yourself"`:

**Checklist:**
- [ ] What is the DiT architecture? (layers, hidden dim, attention type,
      conditioning mechanism)
- [ ] What text encoder(s) are used? (T5, CLIP, VLM, custom)
- [ ] What VAE is used? (latent channels, spatial compression factor)
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
  - `param_names_mapping`: ordered list of `(regex_pattern, replacement)`
    tuples that translate official `state_dict` keys → FastVideo keys.

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

## Worked Example: daVinci-MagiHuman

**Repo:** https://github.com/GAIR-NLP/daVinci-MagiHuman  
**Architecture:** 15B unified single-stream Transformer (not a standard DiT).
Text, video, and audio processed jointly via self-attention only. No
cross-attention.

Key architecture details:
- 40 layers total; first 4 and last 4 use modality-specific projections;
  middle 32 layers share parameters across modalities (sandwich design).
- Per-head scalar gating (sigmoid) for stability.
- Two-stage inference: low-res generation → latent-space super-resolution.
- Text encoder: `t5gemma-9b` (Google).
- Audio model: `stable-audio-open-1.0` (Stability AI).
- VAE: `Wan2.2-TI2V-5B` latent space + custom lightweight turbo VAE decoder.

**Non-standard aspects requiring attention:**
- Unified sequence model — `DistributedAttention` applies to the full
  joint token sequence, not just video patches.
- Sandwich weight sharing requires explicit module aliasing in FastVideo
  (middle 32 layers share the same `nn.Module` instance).
- Audio conditioning is a new modality type; may require a new pipeline
  stage (`AudioEncodingStage`).
- Two-stage pipeline requires two `DenoiseStage` calls or a custom stage.
- `stable-audio-open-1.0` weights are Apache-licensed; confirm redistribution
  rights before uploading a converted HF repo under `fastvideo/`.

**Recommended port order:**
1. VAE (reuse `WanVAE` with Wan2.2-TI2V-5B weights — likely same arch).
2. Text encoder (T5Gemma-9B — check if FastVideo has T5 support already).
3. DiT forward pass, T2V only (skip audio conditioning first).
4. Audio encoder (second pass once T2V parity passes).
5. Super-resolution stage.

---

## Known pitfalls (from Cosmos 2.5 port)

Read `.agents/lessons/` before starting. Key lessons from the first real port:

- **ReplicatedLinear required for LoRA** — `nn.Linear` is invisible to
  `get_lora_layer()`; alignment tests pass but LoRA has zero effect.
  See `2026-04-09_lora-requires-replicated-linear.md`.
- **dtype mismatch fp32/bf16** — cast inputs at projection layers when
  `--dit_precision fp32` is used. See `2026-04-09_fp32-bf16-dtype-mismatch-at-projection.md`.
- **Hardcoded (512, 4096) in utils.py** — breaks any non-T5-XXL encoder.
  See `2026-04-09_hardcoded-text-encoder-shape-in-utils.md`.
- **VLM processor needs inner tokenizer** — unwrap `processor.tokenizer` for
  text-only encoding. See `2026-04-09_multimodal-processor-needs-inner-tokenizer.md`.
- **bf16 embeddings crash numpy** — `.cpu().float().numpy()` not `.cpu().numpy()`.
  See `2026-04-09_bf16-embeddings-need-float-before-numpy.md`.
- **Preprocessing entrypoint** — check whether model is ported to
  `v1_preprocessing_new.py` before writing example scripts.
  See `2026-04-09_preprocessing-entrypoint-not-always-ported.md`.
- **Scheduler shift ignored during training** — `train()` overwrites
  `self.noise_scheduler`; see `2026-04-09_scheduler-shift-not-wired-to-training.md`.

## Eval harness (training loop)

To improve the skill, run the eval harness against a known-good port:

```bash
# Step 1: create eval branch from pre-port commit
python .agents/skills/fastvideo-port/scripts/eval_harness.py \
    --ground_truth_branch feat/cosmos25-training \
    --pre_port_commit <hash_before_cosmos_work> \
    --model_name cosmos2_5

# Step 2: run the port skill on the eval branch, then diff
python .agents/skills/fastvideo-port/scripts/eval_harness.py \
    --ground_truth_branch feat/cosmos25-training \
    --agent_branch eval/cosmos25-port-test \
    --diff_only --model_name cosmos2_5 \
    --output_dir eval_results/cosmos25/

# Step 3: review lesson candidates in eval_results/cosmos25/lesson_candidates/
# Promote good ones to .agents/lessons/
```

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

## Changelog

| Date | Change |
|------|--------|
| 2026-04-17 | Added recon.py + auto_mapper.py to Steps 0 and 1; added 8 lessons from Cosmos 2.5 port; added eval harness section |
| 2026-04-16 | Initial version; daVinci-MagiHuman as worked example |
