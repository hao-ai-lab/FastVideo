---
name: add-model
description: Use when adding a new model (or new model variant) to FastVideo — a text/image/video diffusion pipeline under `fastvideo/pipelines/basic/`. Walks through the component-by-component integration sequence, pointing at exact files, base classes, and registration hooks. FastVideo has one pipeline architecture (stage-based composition), so this skill does not cover style choices — only the canonical flow.
---

# Add a Model to FastVideo

## Purpose

Port a new diffusion model into FastVideo so it runs end-to-end through
`VideoGenerator.from_pretrained(...)` with the same public API as existing
models (Wan, LTX-2, Hunyuan, Cosmos, etc.).

This skill is the **procedural index**. For the narrative version — why each
step exists, prompting examples, and a Wan2.1 case study — read
`docs/contributing/coding_agents.md` first, then return here.

## When to use

- A new model family is being ported (e.g., "add FooVideo").
- A new variant of an existing family needs its own pipeline config / preset
  (e.g., a distilled or I2V variant that doesn't fit the existing wiring).
- The official code or HF repo already exists and you have weights locally.

## When not to use

- Tweaking sampling defaults only → edit the relevant preset in
  `fastvideo/pipelines/basic/<family>/presets.py`. No new pipeline needed.
- Reconfiguring an existing pipeline's CLI → use the API skill track
  (`fastvideo/api/*` + `fastvideo/configs/pipelines/*`), not this.
- Seeding SSIM reference videos for a *just-added* test → that is a separate
  skill, `.agents/skills/seed-ssim-references/`.

## Prerequisites

**Do not start any work until you have explicitly collected all three of
the following from the user.** If any is missing, stop and ask — don't
guess, don't scrape, don't assume.

1. **Official reference repository.** GitHub URL or Diffusers pipeline
   source. This is the numerical ground-truth you'll parity-test against.
2. **Hugging Face weight path.** The HF model ID (or local directory) where
   the weights live. Confirm whether the repo is **Diffusers-format** (has
   `model_index.json` at the root with `_class_name`, plus per-component
   subdirs like `transformer/`, `vae/`, `text_encoder/`) or **raw official
   format** (monolithic `.safetensors` / `.pt`, no `model_index.json`).
   This dictates whether step 4 (weight conversion) is needed.
3. **Hugging Face write-enabled API key** — always ask at the start, even
   if you think conversion/publishing won't be needed. Must have **write
   scope** (a read-only token will not pass
   `create_hf_repo.py --push_to_hub` nor the reference-video upload used
   by `seed-ssim-references`). Ask the user to export it as one of
   `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` / `HF_API_KEY` in the shell where
   the skill will run. **Never** accept the raw token pasted in chat and
   **never** echo it back, log it, or commit it. If no env var is set,
   stop and wait.

Ask exactly: *"Before we start: (1) what's the official reference
repo/pipeline URL, (2) what's the HF model path or local weights
directory, and is it already in Diffusers format, and (3) is a
write-enabled HF API key exported in the environment (which env var —
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`)?"*

Additional environment prerequisites:

- Editable install working (`uv pip install -e .[dev]`) and pre-commit
  hooks installed.
- Ability to run single-GPU inference (for parity + smoke tests). Multi-GPU
  optional.
- If weights aren't yet downloaded, stage them under
  `official_weights/<model_name>/` (see the
  `scripts/huggingface/download_hf.py` helper).

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_family` | Yes | snake_case identifier for the family (e.g. `ltx2`, `wan`, `foovideo`). Used for directory, registry, and preset names. |
| `pipeline_class` | Yes | PascalCase pipeline class name (e.g. `LTX2Pipeline`). Must match the Diffusers `model_index.json` `_class_name` when one exists. |
| `hf_model_paths` | Yes | List of HF model IDs this pipeline should claim (e.g. `["Org/Model-Diffusers"]`). |
| `workload_types` | Yes | Tuple of `WorkloadType` values the pipeline supports (`T2V`, `I2V`, `V2V`, `T2I`, ...). |
| `official_ref` | Yes | Pointer to the reference implementation (repo URL, pipeline file, or HF model id). **Must be collected from the user before any code action.** |
| `hf_weights_path` | Yes | HF model ID or local directory where weights live, plus whether it is already in Diffusers format. **Must be collected from the user before any code action.** Drives whether step 4 (conversion) is needed. |
| `hf_write_api_key` | Yes | Confirmation that a write-enabled HF token is exported (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` / `HF_API_KEY`). **Must be collected at the start of the skill**, before any work. Used for publishing converted Diffusers repos and for SSIM reference uploads. Never accept the raw token in chat. |
| `reuse_candidates` | No | Existing families whose DiT/VAE/encoders you expect to reuse — surface these up front to avoid duplicate implementations. |

## FastVideo's single architecture

Every pipeline in FastVideo is a **stage-based composition** that inherits
from `ComposedPipelineBase`. There is **no** hybrid-vs-modular choice like
other frameworks have — all models use the same pattern. Variation happens
three ways:

1. **Reuse standard stages** when the model fits them (most common).
2. **Subclass a standard stage** when one step diverges (e.g.
   `LTX2DenoisingStage(DenoisingStage)` for audio+video fusion).
3. **Add a model-specific stage** when no standard stage fits.

The canonical stage order is:

```
# T2V (verified against fastvideo/pipelines/basic/wan/wan_pipeline.py):
InputValidationStage
  → TextEncodingStage
  → ConditioningStage            # present in Wan T2V + I2V; treat as default, not optional
  → TimestepPreparationStage
  → LatentPreparationStage
  → DenoisingStage               # receives transformer (+ transformer_2 for MoE) + scheduler + vae
  → DecodingStage

# I2V (verified against fastvideo/pipelines/basic/wan/wan_i2v_pipeline.py):
InputValidationStage
  → TextEncodingStage
  → ImageEncodingStage           # NEW — CLIP-style image embedding; guarded on image_encoder being loaded
  → ConditioningStage
  → TimestepPreparationStage
  → LatentPreparationStage
  → ImageVAEEncodingStage        # NEW — encodes the reference image into VAE latent space
  → DenoisingStage               # no vae kwarg here; it was consumed by ImageVAEEncodingStage
  → DecodingStage
```

Reference: `fastvideo/pipelines/basic/wan/wan_pipeline.py` is the cleanest
"standard" example. Model-specific stages and overrides live in
`fastvideo/pipelines/basic/<family>/stages/`, not in the shared
`fastvideo/pipelines/stages/` tree.

## Files you will create or touch

Minimum set for a new model family. Paths are relative to the repo root.

| # | Path | Role |
|---|------|------|
| 1 | `fastvideo/models/dits/<family>.py` | DiT implementation (reuses FastVideo layers + attention). |
| 2 | `fastvideo/configs/models/dits/<family>.py` | DiT arch config + `param_names_mapping`. |
| 3 | `fastvideo/configs/models/dits/__init__.py` | Export the new DiT config. |
| 4 | `fastvideo/models/vaes/<family>vae.py` | VAE — skip if reusing an existing one. |
| 5 | `fastvideo/configs/models/vaes/<family>vae.py` | VAE config — skip if reusing. |
| 6 | `fastvideo/configs/models/vaes/__init__.py` | Export the new VAE config (if added). |
| 7 | `fastvideo/configs/pipelines/<family>.py` **or** `fastvideo/pipelines/basic/<family>/pipeline_configs.py` | `PipelineConfig` subclass wiring DiT + VAE + encoders. |
| 8 | `fastvideo/pipelines/basic/<family>/<family>_pipeline.py` | Pipeline class with `EntryClass = <PipelineClass>`. **I2V / V2V / DMD variants get their own sibling file** (e.g. `wan_i2v_pipeline.py`, `wan_dmd_pipeline.py`, `wan_v2v_pipeline.py`), each with its own `EntryClass`. See [Adding an I2V variant](#adding-an-i2v-variant). |
| 9 | `fastvideo/pipelines/basic/<family>/presets.py` | `InferencePreset`s + `ALL_PRESETS` tuple. |
| 10 | `fastvideo/pipelines/basic/<family>/stages/` (optional) | Model-specific stage subclasses. If added, import them in `fastvideo/pipelines/stages/__init__.py`. |
| 11 | `fastvideo/registry.py` | Add `register_configs(...)` call inside `_register_configs()` **and** import + add to the group tuple inside `_register_presets()`. |
| 12 | `tests/local_tests/pipelines/test_<family>_pipeline_smoke.py` | End-to-end smoke test (in-process, fast). |
| 13 | `tests/local_tests/pipelines/test_<family>_pipeline_parity.py` | **Mandatory** end-to-end parity test against the cloned official pipeline. Must be green before handing off to the user. Template: `tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py`. |
| 14 | `tests/local_tests/<component>/test_<family>_*.py` | Per-component numerical parity tests (DiT, VAE, encoder). |
| 15 | `fastvideo/tests/ssim/test_<family>_similarity.py` | SSIM regression test (reference videos seeded via `.agents/skills/seed-ssim-references/`). |
| 16 | `scripts/checkpoint_conversion/<family>_to_diffusers.py` (or `convert_<family>_to_diffusers.py`) | **Only if source weights are not Diffusers-format.** Rewrites keys + splits by component into `converted_weights/<family>/`. See the [Weight conversion](#weight-conversion) section. |
| 17 | `examples/inference/basic/basic_<family>.py` | User-runnable example script for manual quality verification. Mirrors existing `basic_<family>.py` examples (e.g. `basic_ltx2.py`, `basic_wan2_2.py`). |

## Steps

1. **Gather inputs from the user (blocking).**
   - Before any file read, grep, or implementation, ask the user for
     **all three** of the following and wait for answers:
     1. **Official reference repo/pipeline URL.**
     2. **HF model path or local weights directory**, plus whether it is
        already Diffusers-format (has `model_index.json`).
     3. **Write-enabled HF API key** — confirmation that one of `HF_TOKEN`
        / `HUGGINGFACE_HUB_TOKEN` / `HF_API_KEY` is exported in the shell.
        You need this for (a) publishing the converted Diffusers repo via
        `create_hf_repo.py --push_to_hub`, and (b) uploading SSIM reference
        videos later. Do not accept the raw token in chat. Verify it's set
        with `[ -n "$HF_TOKEN$HUGGINGFACE_HUB_TOKEN$HF_API_KEY" ] && echo ok`.
   - Also confirm: the target `model_family` name (snake_case), supported
     workload types (`T2V`/`I2V`/`V2V`/`T2I`/…), and whether the user
     wants LoRA support.
   - If any of the three are missing, stop and let the user resolve. A
     wrong HF path silently cascades into wrong `param_names_mapping` and
     wasted parity debugging; a missing write token fails step 4 and
     step 14 after the costly work is already done.

2. **Study the reference implementation.**
   - Open the official repo / Diffusers pipeline source. Locate
     `model_index.json` (or the equivalent config) and list the required
     modules (`text_encoder`, `tokenizer`, `vae`, `transformer`,
     `scheduler`, plus any extras like `audio_vae`, `vocoder`,
     `image_encoder`).
   - Read `pipeline.__call__` end-to-end. Note: text encoding flow, latent
     shape/dtype/scaling, timestep schedule, DiT forward kwargs,
     CFG/guidance mechanism, VAE scale/shift, any post-decoding processing.

3. **Decide what to reuse.** Before creating any file, grep existing pipelines:
   - Does an existing family use the same text encoder? (CLIP, T5, Gemma, ...)
   - Is the VAE identical to an existing one? Reuse `AutoencoderKL` /
     `WanVAE` / `LTX2VAE` / etc. where possible.
   - Does an existing `DenoisingStage` (or its subclasses:
     `CosmosDenoisingStage`, `LTX2DenoisingStage`, `DmdDenoisingStage`,
     `CausalDenoisingStage`, ...) match the denoising math? If 80%+ of a
     step matches, subclass rather than duplicate.
   - If the reference implementation is already in
     [SGLang's multimodal stack](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen),
     much of the logic can be ported (interfaces align with FastVideo's).
   - Record the reuse decisions in the PR description.

4. **Convert weights to Diffusers format — only if the HF path is not
   already Diffusers-format.** See the
   [Weight conversion](#weight-conversion) section below for the detailed
   recipe. Summary decision tree:
   - HF repo has `model_index.json` and per-component subdirs →
     **skip conversion**, download via `scripts/huggingface/download_hf.py`.
     You still need `param_names_mapping` in the DiT config.
   - HF repo / official release has a monolithic `.safetensors` (or `.pt`)
     with no `model_index.json` → **write a conversion script** under
     `scripts/checkpoint_conversion/convert_<family>_to_diffusers.py` (or
     `<family>_to_diffusers.py`), modeled on the closest existing script.
     Stage the output under `converted_weights/<family>/`. Optionally
     publish to HF via `create_hf_repo.py`.
   - Unsure → inspect the HF path's file listing before deciding:

     ```bash
     python - <<'PY'
     from huggingface_hub import list_repo_files
     for f in sorted(list_repo_files("Org/Model")):
         print(f)
     PY
     ```

     Presence of `model_index.json` at the root means Diffusers-format.

5. **Clone the official reference repo locally (for parity testing).**
   - Existing parity tests under `tests/local_tests/` (see e.g.
     `tests/local_tests/transformers/test_ltx2.py:13-16`) expect the
     reference repo to be checked out **at the FastVideo repo root**
     (`FastVideo/LTX-2/`, `FastVideo/gen3c/`, …). They push its module
     path onto `sys.path` and skip if the directory is absent.
   - Clone under the repo root, for example:

     ```bash
     git clone --depth 1 <official_repo_url> <family-upper-or-canonical>
     ```

   - Add the clone to `.gitignore` so it's not accidentally committed:

     ```bash
     echo "/<FamilyDir>/" >> .gitignore
     ```

   - Install its dependencies **inside the same FastVideo environment** —
     do not create a new venv, do not let it override FastVideo-pinned
     versions. Typical recipe:

     ```bash
     uv pip install --no-deps -e ./<FamilyDir>           # if editable
     uv pip install -r ./<FamilyDir>/requirements.txt --no-deps
     ```

     Use `--no-deps` first so you can review what extra transitive deps
     the official repo wants. If any conflict with FastVideo's pins (torch,
     transformers, diffusers, flash-attn, triton), stop and ask the user
     before changing versions — a mismatched torch is the single most
     common source of phantom parity failures.
   - If the official implementation is already reachable on PyPI (e.g. the
     Diffusers library itself), skip the clone and `pip install` the pinned
     version instead.
   - This directory is **temporary**. Step 14 removes it after SSIM refs
     are seeded.

6. **Port each component in parallel using subagents.** This is where the
   bulk of the implementation happens. Launch one subagent per component
   (DiT, VAE, text encoder(s), any extras — audio_vae, vocoder,
   image_encoder, upsampler, …) in a **single message with multiple Agent
   tool calls** so they run concurrently. See the
   [Parallel component porting](#parallel-component-porting) section below
   for the exact subagent prompt template and the
   [Parity test pattern](#parity-test-pattern) section for the test
   template each subagent must produce.

   Each subagent is responsible for producing, for its one component:
   - The FastVideo model class under `fastvideo/models/<bucket>/<family>.py`
     (where `<bucket>` is `dits`, `vaes`, `encoders`, `schedulers`, or
     `upsamplers`).
   - The config class under `fastvideo/configs/models/<bucket>/<family>.py`,
     subclassing the bucket's base config and defining
     `param_names_mapping` (regex map from official keys → FastVideo keys).
   - An export entry in the bucket's `__init__.py`.
   - A component parity test under `tests/local_tests/<bucket>/test_<family>_*.py`
     that loads FastVideo and official models in-process, runs them on
     identical inputs, and asserts closeness with `torch.testing.assert_close`.
   - Hard rule: **reuse `fastvideo/layers/` and `fastvideo/attention/`
     primitives.** Do not introduce raw `nn.Linear`, raw SDPA, or hand-rolled
     RMSNorm/RoPE. See the [FastVideo layers and attention](#fastvideo-layers-and-attention)
     section for the catalog + selection rules.

   The main agent's job during this step is:
   - Draft the per-component subagent prompts (one each), reference the
     closest existing component file, and dispatch them in parallel.
   - As each subagent returns, inspect the diff (do not trust the
     summary). Verify the parity test actually compares forward outputs
     (not just weight shapes) and that `assert_close` tolerances are
     sensible (`atol=1e-4, rtol=1e-4` default; loosen deliberately, never
     silently).
   - Collect unresolved `param_names_mapping` mismatches; resolve them
     centrally before moving on.
   - Open a **DRAFT PR** once the DiT is parity-clean. Remaining
     components can land incrementally.

7. **Create the `PipelineConfig`.**
   - For one-off families, the file can live in
     `fastvideo/configs/pipelines/<family>.py`. For families that introduce
     multiple related configs (like LTX-2), use
     `fastvideo/pipelines/basic/<family>/pipeline_configs.py`. Both
     patterns exist — pick the one that matches the family you are closest
     to structurally.
   - Subclass `PipelineConfig` (see
     `fastvideo/configs/pipelines/base.py`). Wire `dit_config`,
     `vae_config`, `text_encoder_configs`, precisions, `flow_shift`,
     `boundary_ratio`, `vae_tiling`, etc. Override `__post_init__` only to
     toggle `load_encoder` / `load_decoder` or adjust encoder
     `output_hidden_states` for downstream needs.

8. **Build or pick the stages.**
   - Start by trying to compose only standard stages from
     `fastvideo/pipelines/stages/`. If that works, skip to step 9.
   - If a step diverges, subclass the matching standard stage and put it
     under `fastvideo/pipelines/basic/<family>/stages/<step>.py`.
     - Keep overrides small: a new stage usually just wraps a different
       scheduler math path, an extra conditioning kwarg, or a different
       latent packing.
     - If you add new stage classes, re-export them from
       `fastvideo/pipelines/basic/<family>/stages/__init__.py` and add the
       imports to `fastvideo/pipelines/stages/__init__.py` so downstream
       code can still do `from fastvideo.pipelines.stages import ...`.

9. **Write the pipeline class.**

   ```python
   # fastvideo/pipelines/basic/<family>/<family>_pipeline.py
   from fastvideo.fastvideo_args import FastVideoArgs
   from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
   from fastvideo.pipelines.stages import (
       DecodingStage, DenoisingStage, InputValidationStage,
       LatentPreparationStage, TextEncodingStage, TimestepPreparationStage,
   )


   class FooVideoPipeline(LoRAPipeline, ComposedPipelineBase):  # drop LoRAPipeline if unsupported
       _required_config_modules = [
           "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
       ]

       def initialize_pipeline(self, args: FastVideoArgs) -> None:
           # Optional hook: swap scheduler, preload custom modules, etc.
           ...

       def create_pipeline_stages(self, args: FastVideoArgs) -> None:
           self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
           self.add_stage(
               stage_name="prompt_encoding_stage",
               stage=TextEncodingStage(
                   text_encoders=[self.get_module("text_encoder")],
                   tokenizers=[self.get_module("tokenizer")],
               ),
           )
           self.add_stage(stage_name="timestep_preparation_stage",
                          stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))
           self.add_stage(stage_name="latent_preparation_stage",
                          stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                       transformer=self.get_module("transformer", None)))
           self.add_stage(stage_name="denoising_stage",
                          stage=DenoisingStage(transformer=self.get_module("transformer"),
                                               scheduler=self.get_module("scheduler"),
                                               vae=self.get_module("vae"),
                                               pipeline=self))
           self.add_stage(stage_name="decoding_stage",
                          stage=DecodingStage(vae=self.get_module("vae"), pipeline=self))


   EntryClass = FooVideoPipeline  # single class, NOT a list
   ```

   - Module-level `EntryClass` is how `fastvideo/pipelines/pipeline_registry.py`
     discovers pipelines. No manual pipeline registration required.
   - `_required_config_modules` must list every key the loader will read from
     `model_index.json`. Missing keys cause silent degradation.

10. **Define presets.**

   ```python
   # fastvideo/pipelines/basic/<family>/presets.py
   from fastvideo.api.presets import InferencePreset, PresetStageSpec

   _DENOISE_STAGE = PresetStageSpec(
       name="denoise",
       kind="denoising",
       description="Main denoising pass",
       allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
   )

   FOO_BASE = InferencePreset(
       name="foovideo_base",
       version=1,
       model_family="foovideo",
       description="FooVideo base at 512x768",
       workload_type="t2v",
       stage_schemas=(_DENOISE_STAGE,),
       defaults={
           "seed": 0,
           "height": 512,
           "width": 768,
           "num_frames": 121,
           "fps": 24,
           "guidance_scale": 3.0,
           "num_inference_steps": 40,
           "negative_prompt": "",
       },
   )

   ALL_PRESETS = (FOO_BASE,)
   ```

   - `ALL_PRESETS` is the export expected by `_register_presets()`. Every
     preset listed there becomes reachable via
     `SamplingParam.from_pretrained(model_path)` or the public preset API.
   - Multi-stage pipelines (e.g. LTX-2's two-stage refine) declare multiple
     `PresetStageSpec` entries and provide `stage_defaults`.

11. **Register in `fastvideo/registry.py`.** Two edits required:

    a. Add a `register_configs(...)` block inside `_register_configs()`:

    ```python
    register_configs(
        sampling_param_cls=None,        # use the default SamplingParam unless you subclass it
        pipeline_config_cls=FooVideoT2VConfig,
        workload_types=(WorkloadType.T2V,),
        hf_model_paths=[
            "Org/FooVideo-Diffusers",
        ],
        model_detectors=[
            lambda path: "foovideo" in path.lower(),
        ],
        model_family="foovideo",
        default_preset="foovideo_base",
    )
    ```

    b. Inside `_register_presets()`, import the family's presets and add
       them to `all_preset_groups`:

    ```python
    from fastvideo.pipelines.basic.foovideo.presets import (
        ALL_PRESETS as FOOVIDEO_PRESETS,
    )
    # ...
    all_preset_groups = (
        ...,
        FOOVIDEO_PRESETS,
    )
    ```

    Discovery order during `_get_config_info`: exact HF path → short name →
    detector fallback (which also matches against the `_class_name` pulled
    from `model_index.json`). Make sure your detector accepts both the HF
    path and the pipeline class name.

12. **Smoke-test the pipeline.**

    ```python
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    gen = VideoGenerator.from_pretrained("Org/FooVideo-Diffusers", num_gpus=1)
    sp = SamplingParam.from_pretrained("Org/FooVideo-Diffusers")
    video = gen.generate_video(
        "A vibrant city street at sunset.",
        sampling_param=sp,
        output_path="video_samples",
        save_video=True,
    )
    ```

    Add this as `tests/local_tests/pipelines/test_<family>_pipeline_smoke.py`
    mirroring `tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py`.

13. **Full-pipeline parity + user-runnable example (gated; iterate until
    green before any user hand-off).** This is the final verification
    gate. Do **not** tell the user the port is ready until (a) passes.

    (a) **Pipeline parity test against the official implementation —
        mandatory, iterate until green.** Write
        `tests/local_tests/pipelines/test_<family>_pipeline_parity.py`
        modeled on
        `tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py`.
        The test must:
        - Push the cloned official repo onto `sys.path` and import its
          pipeline, matching the pattern used by the component parity
          tests (see [Parity test pattern](#parity-test-pattern)).
        - Instantiate both pipelines **in the same process** with
          matched weights, matched prompts, matched seed, matched
          resolution, matched `num_inference_steps`.
        - Compare **denoised latents** (faster than decoded video) with
          `torch.testing.assert_close`. Start at `atol=1e-3, rtol=1e-3`;
          if the denoising loop accumulates more drift than that, log
          per-step latent sums from both pipelines and chase down the
          first divergent step before loosening bounds.
        - Set `DISABLE_SP=1` / `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
          in env defaults to match the official backend.
        - `pytest.skip` cleanly when the official clone or converted
          weights are missing — never fail.

        Run: `DISABLE_SP=1 pytest -v -s tests/local_tests/pipelines/test_<family>_pipeline_parity.py`.
        If it fails, **iterate** — the most common causes are:
        - Sigma / timestep schedule drift (first denoising step diverges).
        - VAE scale/shift wrong in `PipelineConfig` (latents come out
          wrong magnitude but shape right).
        - CFG kwargs not matching the DiT's `forward()` signature.
        - Rotary embedding style (neox vs interleaved) flipped.
        - Prompt embeds passed as tensor where the DiT expects a list,
          or vice versa.
        - `vae_precision` causing overflow on long sequences.

        Do not move to (b) until this test passes with a non-skip pass.

    (b) **Create the user-facing example script.** Once (a) is green,
        add `examples/inference/basic/basic_<family>.py` modeled on the
        closest existing example (e.g. `basic_ltx2.py`, `basic_wan2_2.py`,
        `basic_longcat_t2v.py`). Keep it minimal and deterministic:

        ```python
        from fastvideo import VideoGenerator


        PROMPT = "<one-line representative prompt>"


        def main() -> None:
            generator = VideoGenerator.from_pretrained(
                "<Org>/<Family>-Diffusers",
                num_gpus=1,
            )
            output_path = "outputs_video/<family>_basic/output_<family>_t2v.mp4"
            generator.generate_video(
                prompt=PROMPT,
                output_path=output_path,
                save_video=True,
                # Leave sampling params blank to use the registered preset.
                # Only override here if a specific resolution / frame count
                # is meaningful for manual QA.
            )
            generator.shutdown()


        if __name__ == "__main__":
            main()
        ```

        Run it once yourself to confirm it writes a non-corrupted mp4.
        Do not tune sampling params beyond the preset unless the preset
        defaults are clearly wrong — in that case, fix the preset, not
        the example.

    (c) **Only now, ask the user to manually verify quality.** Point
        them at `examples/inference/basic/basic_<family>.py` and the
        passing pipeline parity test output. The pipeline parity test
        is the objective gate; the example script is for the user's
        subjective eyeball check.

14. **Add SSIM regression + seed references.**
    - Copy `fastvideo/tests/ssim/test_ltx2_similarity.py` and adapt for the
      new model: parametrize prompts, attention backends, and model_ids;
      set `min_acceptable_ssim` (start at 0.60 unless the model is known
      to be stable).
    - The first run will fail on missing reference videos — that is
      expected. Hand off to the
      [`seed-ssim-references`](../seed-ssim-references/SKILL.md) skill to
      generate the refs on Modal L40S and upload them to
      `FastVideo/ssim-reference-videos`.

15. **Clean up the locally cloned official repo.** Once the DiT, VAE,
    encoder, and pipeline parity tests are all green (or have been
    converted into permanent, skip-on-missing tests), remove the
    `FastVideo/<FamilyDir>/` clone introduced in step 5:

    ```bash
    rm -rf <FamilyDir>
    git check-ignore -v <FamilyDir> || true  # confirm it was gitignored
    ```

    Leave the `.gitignore` line in place so future parity runs (which
    re-clone on demand) stay untracked. If your parity tests need the
    clone permanently, instead document the expected directory name in
    `tests/local_tests/README.md` and leave the clone untouched — but
    still never commit it.

16. **Ask about tests and performance data.** Once quality is verified and
    SSIM is green, ask the user whether to add:
    - Additional unit/integration tests (per-component parity tests under
      `tests/local_tests/<component>/`).
    - Performance benchmarks (single-GPU latency, TP/SP scaling numbers).

    Do not skip — the scope varies per model and per contributor goal.

## Standard stages — subclass targets

All live under `fastvideo/pipelines/stages/` unless noted. When an existing
subclass is listed, that is a strong signal others have hit the same
divergence — read it before writing your own.

| Stage | File | Existing subclasses |
|-------|------|---------------------|
| `PipelineStage` (base) | `base.py` | — |
| `InputValidationStage` | `input_validation.py` | (shared, rarely subclassed) |
| `TextEncodingStage` | `text_encoding.py` | `Cosmos25TextEncodingStage`, `LTX2TextEncodingStage` |
| `ImageEncodingStage` et al. | `image_encoding.py`, `gamecraft_image_encoding.py` | `MatrixGame...`, `Hy15...`, `HYWorld...`, `RefImage...`, `ImageVAE...`, `VideoVAE...`, `GameCraftImageVAE...` |
| `ConditioningStage` | `conditioning.py` | `Gen3CConditioningStage`, `SD35ConditioningStage` |
| `TimestepPreparationStage` | `timestep_preparation.py` | `Cosmos25TimestepPreparationStage` |
| `LatentPreparationStage` | `latent_preparation.py` | `CosmosLatentPreparationStage`, `Cosmos25...`, `Gen3CLatentPreparationStage`, `LTX2LatentPreparationStage` |
| `DenoisingStage` | `denoising.py` | `CosmosDenoisingStage`, `Cosmos25(...)DenoisingStage`, `DmdDenoisingStage`, plus causal (`causal_denoising.py`), `MatrixGameCausalDenoisingStage`, `HYWorldDenoisingStage`, `GameCraftDenoisingStage`, `Gen3CDenoisingStage`, `LTX2DenoisingStage`, `SRDenoisingStage`, `LongCatVCDenoisingStage` |
| `DecodingStage` | `decoding.py` | `LTX2AudioDecodingStage` |
| `EncodingStage` | `encoding.py` | — |

## FastVideo layers and attention

Every component you port **must** build on the primitives in
`fastvideo/layers/` and `fastvideo/attention/`. Do not introduce raw
`torch.nn.Linear` in hot paths, hand-rolled RMSNorm, custom SDPA calls,
or ad-hoc RoPE implementations. The shared primitives handle: attention
backend selection, TP/SP sharding, torch.compile compatibility,
bf16/fp32 dispatch, and quantization. Bypassing them breaks all of those
invariants silently.

### Linear layers (`fastvideo/layers/linear.py`)

FastVideo has two linear regimes, and mixing them up is a common porting
mistake. **Verified across every DiT in `fastvideo/models/dits/` as of
2026-04-24:**

- **DiTs and VAEs use `ReplicatedLinear` everywhere** — QKV projections,
  output projection, MLP up/down projections, adaLN modulation heads,
  timestep/context embedders, patch projection. There is **zero** usage
  of `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` /
  `MergedColumnParallelLinear` in any DiT or VAE. Multi-GPU scaling in
  DiTs comes from **sequence parallelism via `DistributedAttention`**,
  not from TP-sharding the linear weights.
- **LLM-style text encoders use TP linears.** Files like
  `fastvideo/models/encoders/llama.py`, `qwen2_5.py`, `siglip.py`, and
  `t5.py` shard their QKV / MLP across TP ranks using
  `QKVParallelLinear`, `MergedColumnParallelLinear`, and
  `RowParallelLinear`. Copy the pattern from these files when your new
  text encoder is LLM-structured.

Summary rule:

| Component bucket | Default linear | Why |
|------------------|---------------|-----|
| DiT (`fastvideo/models/dits/`) | `ReplicatedLinear` everywhere. | DiTs scale via SP on the long spatio-temporal sequence, not via TP on linear weights. `DistributedAttention` handles the sequence redistribution. |
| VAE (`fastvideo/models/vaes/`) | `ReplicatedLinear` (plus `nn.Conv*` / conv primitives). | Same reason — no TP in VAEs. |
| LLM-style text encoder (`fastvideo/models/encoders/`) | `QKVParallelLinear` for QKV, `MergedColumnParallelLinear` for gate/up MLP, `RowParallelLinear` for output/down projections. | Proven win on transformer encoders; existing encoders (llama, qwen2_5, siglip, t5) already do this. |
| Small pre/post heads inside any bucket | `ReplicatedLinear`. | Too small for TP to help. |

Why `ReplicatedLinear` instead of raw `nn.Linear` in DiTs/VAEs? It plugs
into the quantization and weight-loading stacks the same way the TP
variants do, so swapping it in later (if TP ever makes sense for a
specific DiT) is a one-line change rather than a rewrite. A handful of
legacy `nn.Linear` calls remain in some DiT files (e.g. `proj_out`) —
treat those as legacy, not as a pattern to copy. For new code, default
to `ReplicatedLinear`.

Reference reading before writing any linear layer:

| Reading for… | Open |
|--------------|------|
| DiT linear pattern | `fastvideo/models/dits/wanvideo.py`, `fastvideo/models/dits/ltx2.py`, `fastvideo/models/dits/hunyuanvideo.py` — all `ReplicatedLinear` throughout. |
| LLM text encoder TP pattern | `fastvideo/models/encoders/llama.py:39-141`, `fastvideo/models/encoders/qwen2_5.py`. |
| SigLIP vision-encoder TP pattern | `fastvideo/models/encoders/siglip.py:96-168`. |
| T5 encoder TP pattern | `fastvideo/models/encoders/t5.py`. |

### Attention layers (`fastvideo/attention/layer.py`)

There are **three** attention layers. Picking the wrong one is one of
the most common porting bugs — it parity-passes on single GPU and then
breaks silently under SP.

| Class | Use for | Do not use for |
|-------|---------|----------------|
| `DistributedAttention` | The DiT's **full-sequence self-attention** layers — the long spatial/temporal sequences that get SP-sharded. Handles all-to-all head/seq redistribution, RoPE application, padding trim/pad. | Cross-attention. Short-sequence attention inside encoders. |
| `DistributedAttention_VSA` | Video Sparse Attention variant of the above — only when the reference model uses VSA-style attention. | Dense attention. |
| `LocalAttention` | Cross-attention blocks, short-seq attention inside text/image encoders, and any attention that should **not** be SP-sharded. | Full-sequence DiT self-attention (it will recompute full attention on every rank). |

Rule of thumb:
- "Does this attention need to see the whole spatio-temporal sequence,
  and will that sequence be sharded across SP ranks?" → `DistributedAttention`.
- Everything else (cross-attn, encoder self-attn on small token counts,
  fusion heads) → `LocalAttention`.

See `fastvideo/models/dits/wanvideo.py` and
`fastvideo/models/dits/ltx2.py` for canonical usage. Do not reach down
into the backend directly via `fastvideo/attention/selector.py` from a
model file — always go through `DistributedAttention` / `LocalAttention`.

### Other layer primitives

| Module | Use for |
|--------|---------|
| `fastvideo/layers/layernorm.py` — `RMSNorm` | Canonical RMSNorm with optional fused scale/shift. |
| `fastvideo/layers/layernorm.py` — `FP32LayerNorm` | When the reference model forces fp32 LayerNorm (common at modulation boundaries). |
| `fastvideo/layers/layernorm.py` — `ScaleResidual`, `ScaleResidualLayerNormScaleShift`, `LayerNormScaleShift` | Fused scale/residual/shift combinations used by most DiT blocks. Prefer over manual `x = x + alpha * residual; norm(x)`. |
| `fastvideo/layers/activation.py` — `SiluAndMul`, `GeluAndMul`, `NewGELU`, `QuickGELU` | Fused activation + multiply for gated MLPs. |
| `fastvideo/layers/mlp.py` — `MLP` | Generic 2-layer MLP; prefer this over writing one inline. |
| `fastvideo/layers/rotary_embedding.py` — `RotaryEmbedding`, `_apply_rotary_emb` | Standard RoPE (1D). Respect the `is_neox_style` flag — neox = split-half, non-neox = interleaved. |
| `fastvideo/layers/rotary_embedding_3d.py` — `RotaryPositionalEmbedding3D` | 3D RoPE used by video DiTs. |
| `fastvideo/layers/visual_embedding.py` — `PatchEmbed`, `TimestepEmbedder`, `ModulateProjection`, `Timesteps` | Standard patch + timestep + modulation embeds. |
| `fastvideo/layers/vocab_parallel_embedding.py` — `VocabParallelEmbedding` | Shardable embedding tables (text encoders). |

### Quick reuse audit before any new layer

When a subagent (or you) is about to write a new nn.Module, check in
this order:

1. Is it an attention? → pick from the attention table above.
2. Is it a Linear? → pick from the linear table above.
3. Is it a norm, activation, RoPE, or embedding? → check the layer
   primitive table; re-export if needed.
4. Is it a block/module composition (e.g. a DiT transformer block)? →
   grep existing DiTs (`wanvideo.py`, `ltx2.py`, `hunyuanvideo.py`,
   `cosmos.py`) for the same pattern and subclass/inline adapt.
5. None of the above? Only then, implement custom and document why.

## Parity test pattern

Every component subagent's output must include a parity test under
`tests/local_tests/<bucket>/` that compares FastVideo against the
locally-cloned official reference. The pattern below is distilled from
`tests/local_tests/transformers/test_ltx2.py`,
`tests/local_tests/vaes/test_ltx2_vae.py`, and
`tests/local_tests/encoders/test_ltx2_gemma_parity.py`.

### Conventions (all parity tests share these)

- **Repo-root resolution**:

  ```python
  from pathlib import Path
  import sys
  repo_root = Path(__file__).resolve().parents[3]
  official_src = repo_root / "<FamilyDir>" / "src"   # adjust to the repo's layout
  if official_src.exists() and str(official_src) not in sys.path:
      sys.path.insert(0, str(official_src))
  ```

  Tests under `tests/local_tests/<bucket>/test_*.py` live three levels
  below repo root — `.parents[3]` resolves correctly. Push the reference
  package's source path onto `sys.path` *before* importing from it.

- **Skip-on-missing, never fail-on-missing.** If the clone, converted
  weights, or official safetensors are absent, `pytest.skip(...)` so
  CI does not have to carry the reference code:

  ```python
  if not official_path.exists():
      pytest.skip(f"Official weights not found at {official_path}")
  ```

- **Paths via env vars with sensible defaults** — lets different
  contributors run the same test with different local layouts:

  ```python
  official_path = Path(os.getenv(
      "FOOVIDEO_OFFICIAL_PATH",
      "official_weights/foovideo/foovideo-v1.safetensors",
  ))
  converted_path = Path(os.getenv(
      "FOOVIDEO_DIFFUSERS_PATH",
      "converted_weights/foovideo",
  ))
  ```

- **Load matched weights into both models.** Use the conversion
  script's mapping so the same tensor content lands in both models; a
  freshly-initialized FastVideo model compared against a pre-trained
  official model is not a parity test.

- **Identical inputs**, fixed seed:

  ```python
  torch.manual_seed(0)
  x = torch.randn(batch, ..., dtype=precision, device=device)
  ```

- **`torch.testing.assert_close(fastvideo_out, official_out, atol=1e-4, rtol=1e-4)`**
  as the default tolerance. If the test needs looser bounds, write a
  comment explaining *why* (e.g. "flash-attn fp16 accumulation drift —
  aligned with TORCH_SDPA shows 1e-5"). Never loosen silently.

- **Align attention backend** to avoid ghost drift:

  ```python
  os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
  ```

- **Debug logging hooks** — `tests/local_tests/transformers/test_ltx2.py`
  shows the per-block forward-hook pattern that logs activation sums to
  a file. Copy it when the scalar assertion fails and you need to locate
  the first divergent layer.

### Pipeline-level parity test (the gate for user handoff)

Per step 13(a), every ported family must ship a pipeline-level parity
test at `tests/local_tests/pipelines/test_<family>_pipeline_parity.py`.
This is the **final objective gate** before telling the user to eyeball
the example script — if this test is red, the port is not done.

Use `tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py` as
the template. Key differences from component tests:

- Compares on **denoised latents**, not decoded video — decoding is
  expensive and adds VAE error on top. Run the denoising loop in both
  pipelines with matched `num_inference_steps`, snapshot the latent at
  the end of the loop, and compare.
- Tolerances are looser by construction (the denoising loop compounds
  tiny per-step deltas). Start at `atol=1e-3, rtol=1e-3`; if you need
  looser, first log per-step latent sums from both pipelines to prove
  divergence is bounded and steady (not runaway).
- Set `DISABLE_SP=1` in env defaults to avoid cross-pipeline SP
  disagreement when running on multi-GPU hosts.
- `pytest.skip` if the official clone is missing, if converted weights
  are missing, or if GPU memory is below what the pipeline needs.

### Component-specific template skeleton

```python
# tests/local_tests/<bucket>/test_<family>_parity.py
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

repo_root = Path(__file__).resolve().parents[3]
official_src = repo_root / "<FamilyDir>"
if official_src.exists() and str(official_src) not in sys.path:
    sys.path.insert(0, str(official_src))

from fastvideo.models.<bucket>.<family> import FooVideoComponent
from fastvideo.configs.models.<bucket> import FooVideoComponentConfig


def test_foovideo_<component>_parity():
    official_weights = Path(os.getenv(
        "FOOVIDEO_OFFICIAL_PATH",
        "official_weights/foovideo/foovideo-v1.safetensors",
    ))
    converted_weights = Path(os.getenv(
        "FOOVIDEO_DIFFUSERS_PATH",
        "converted_weights/foovideo",
    ))
    if not official_weights.exists():
        pytest.skip(f"Official weights missing: {official_weights}")
    if not converted_weights.exists():
        pytest.skip(f"Converted weights missing: {converted_weights}")
    try:
        from foovideo_official.<component> import OfficialFooComponent
    except ImportError as exc:
        pytest.skip(f"Official reference not importable: {exc}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Build both models
    config = FooVideoComponentConfig()
    fv_model = FooVideoComponent(config).to(device=device, dtype=dtype).eval()
    ref_model = OfficialFooComponent(<ref_args>).to(device=device, dtype=dtype).eval()

    # Load matched weights
    # ... use the conversion mapping to route tensors into both models ...

    # Identical inputs
    torch.manual_seed(0)
    inputs = <build representative forward inputs>

    with torch.no_grad():
        fv_out = fv_model(**inputs)
        ref_out = ref_model(**inputs)

    assert_close(fv_out, ref_out, atol=1e-4, rtol=1e-4)
```

## Parallel component porting

Step 6 launches one subagent per component in parallel. They have no
shared state, so each prompt must be fully self-contained — the agent
doesn't see this conversation.

### Dispatch pattern

Send a **single message with multiple `Agent` tool calls**. Typical
component list:

- DiT (`bucket=dits`)
- VAE (`bucket=vaes`)
- Text encoder(s) (`bucket=encoders`) — one per distinct encoder
- Any extras: audio VAE, vocoder, image encoder, upsampler, scheduler

Skip components that step 3 ("decide what to reuse") marked as reused.

### Subagent prompt template

```
Port the <component_name> component of <family> into FastVideo.

Context:
  - Official reference repo is cloned at FastVideo/<FamilyDir>/.
    Relevant source: <official_module_path>.
  - Converted (Diffusers-format) weights are at
    converted_weights/<family>/<component_dir>/.
  - Closest existing FastVideo component to model on:
    fastvideo/models/<bucket>/<closest_family>.py
    fastvideo/configs/models/<bucket>/<closest_family>.py
  - Closest existing parity test to model on:
    tests/local_tests/<bucket>/test_<closest_family>_parity.py

Hard requirements:
  - Reuse primitives from fastvideo/layers/ and fastvideo/attention/.
    Do NOT introduce raw nn.Linear in hot paths, hand-rolled RMSNorm,
    custom SDPA, or ad-hoc RoPE. Selection rules (verified against
    every existing DiT / VAE / encoder in the repo):
      * Attention:
        - DiT full-sequence self-attention → DistributedAttention
        - DiT Video-Sparse-Attention variant → DistributedAttention_VSA
        - Cross-attention, encoder self-attention, any short-seq attn
          → LocalAttention
      * Linear layers:
        - If this is a DiT or a VAE → ReplicatedLinear everywhere
          (QKV, output proj, MLP, adaLN heads, embedders). No
          ColumnParallel/RowParallel/QKVParallel here — DiTs scale via
          SP inside DistributedAttention, not TP.
        - If this is an LLM-style text encoder → QKVParallelLinear for
          QKV, MergedColumnParallelLinear for gate/up MLP,
          RowParallelLinear for output/down projections (mirror
          fastvideo/models/encoders/llama.py, qwen2_5.py, siglip.py,
          t5.py).
      * Use fused norms/activations from fastvideo/layers/{layernorm,
        activation,mlp,visual_embedding,rotary_embedding{,_3d}}.py.
  - Name parameters to minimize the regex map in
    fastvideo/configs/models/<bucket>/<family>.py :: param_names_mapping.
  - Keep torch.compile-compatible: no Python-side control flow that
    branches on tensor values.

Deliverables (in order):
  1. fastvideo/models/<bucket>/<family>.py — the model class.
  2. fastvideo/configs/models/<bucket>/<family>.py — config + regex map.
  3. Add export in fastvideo/configs/models/<bucket>/__init__.py.
  4. tests/local_tests/<bucket>/test_<family>_parity.py — loads
     FastVideo and official in-process, asserts closeness with
     torch.testing.assert_close (atol=1e-4, rtol=1e-4 default).

Validation before you return:
  - Run: `pytest tests/local_tests/<bucket>/test_<family>_parity.py -vs`
  - The test must produce a non-skip pass (not xfail, not skip). If it
    legitimately must skip (e.g. weights missing), explain which env var
    must be set.
  - Report any unresolved param_names_mapping mismatches verbatim.

Do NOT:
  - Touch the pipeline class, presets, registry, or any other component.
  - Create new files outside the four listed above.
  - Loosen atol/rtol silently. If a looser tolerance is needed, add a
    comment explaining why and quantifying the drift.
```

Notes for the main agent:

- Give each subagent only its component's scope. Don't bundle (e.g. "do
  the DiT and the VAE") — subagents perform worse as scope expands.
- Run them concurrently — **one message with N Agent calls** — they
  have no dependencies on each other.
- When each returns, verify by reading the diff, not by trusting the
  summary.
- Consolidate shared issues centrally (e.g. if every subagent hit a
  missing `param_names_mapping` rule that comes from the conversion
  script, fix the conversion, not each component).

## Weight conversion

When the source weights are **not** in Diffusers format (no
`model_index.json`, single monolithic checkpoint, or custom layout from the
official repo), you must convert them before FastVideo can load the model.
All conversion scripts live under `scripts/checkpoint_conversion/`.

### Decide whether conversion is needed

| Source layout | Action |
|---------------|--------|
| HF repo with `model_index.json` + `transformer/`, `vae/`, `text_encoder/` subdirs, each with `config.json` + `.safetensors` | **No conversion.** Download via `scripts/huggingface/download_hf.py`, rely on `param_names_mapping` in the DiT config to reconcile any module-name drift. |
| HF repo or release with one big `.safetensors` (all components fused, e.g. LTX-2 `ltx-2-19b-dev.safetensors`) | **Write a conversion script.** Split by component prefix, write per-component directories, add a synthetic `model_index.json`. |
| Official `.pt` / `.ckpt` files | First convert to `.safetensors` with `scripts/checkpoint_conversion/pt_to_safetensors.py`, then treat as the previous row. |
| Component weights scattered across separate files with non-Diffusers prefixes | Write a conversion script that loads each, remaps, and lays them out in Diffusers form. See `convert_gen3c_to_fastvideo.py` for a dense example. |

### Conversion script conventions

All scripts in `scripts/checkpoint_conversion/` follow the same shape:

- A CLI with `argparse`, typically:
  - `--source` — path to the official `.safetensors` file or directory.
  - `--output` — target directory, by convention
    `converted_weights/<family>/`.
  - `--class-name` — the Diffusers DiT class name that will appear in
    `model_index.json` (e.g. `"LTX2Transformer3DModel"`).
  - `--pipeline-class-name` — the pipeline class name that FastVideo will
    match on (e.g. `"LTX2Pipeline"`).
  - `--diffusers-version` — version string baked into the output config.
  - Component-specific flags (e.g. `--gemma-path` for LTX-2, image-encoder
    paths, VAE paths, etc.).
- A `PARAM_NAME_MAP: dict[str, str]` of regex patterns: official key →
  FastVideo/Diffusers key. Applied in-order via `re.sub`.
- A `COMPONENT_PREFIXES: dict[str, tuple[str, ...]]` that groups keys by
  destination subdir (`transformer`, `vae`, `audio_vae`, `vocoder`,
  `text_encoder`, …). Each key in the final state dict is routed into
  `<output>/<component>/diffusion_pytorch_model.safetensors`.
- Per-component `config.json` generation so `from_pretrained` can load each
  piece without the original repo.
- A top-level `model_index.json` that pins the `_class_name`,
  `_diffusers_version`, and the list of component sub-paths.

### Reference scripts — pick the closest one and copy its shape

| Script | When to model on it |
|--------|--------------------|
| `convert_ltx2_weights.py` | **Default starting point** for a new model. Modern pattern: shard-safe loading, regex mapping, component splitting (`transformer`/`vae`/`audio_vae`/`vocoder`), synthesized `model_index.json`, optional extra text-encoder side-load. 484 lines, clean. |
| `wan_to_diffusers.py` | **Legacy pattern** — no argparse, just a module of regex dicts (`_param_names_mapping`, `_self_forcing_to_diffusers_param_names_mapping`). Useful as a reference for *how to write regex rules* for self-/cross-attention, FFN, modulation, and image-embedding projections, and for handling variant checkpoints (`self_forcing_`, etc.). **Do not copy the module shape** — follow the `convert_ltx2_weights.py` CLI template instead. |
| `convert_turbodiffusion_to_diffusers.py` / `convert_turbodiffusion_i2v_to_diffusers.py` | Shows T2V vs I2V variants sharing most logic with small deltas — copy when you're adding I2V support for an existing family. |
| `convert_gamecraft_weights.py` + `convert_gamecraft_vae.py` + `convert_gamecraft_full.py` | Pattern for families where the DiT, VAE, and end-to-end conversion live in separate scripts. Use when a single family needs piecewise conversion (different source formats per component). |
| `convert_gen3c_to_fastvideo.py` | Heaviest example (724 lines). Use as reference when the model has many components and non-trivial per-component logic; skim rather than copy wholesale. |
| `longcat_to_fastvideo.py` | Similar: large, component-heavy. Pair with `validate_longcat_weights.py` for the sanity-check pattern. |
| `extract_llava_text_encoder.py` | Template for pulling a single component out of a larger multimodal checkpoint (here, the text encoder half of a LLaVA checkpoint). |
| `pt_to_safetensors.py` | Utility — convert legacy `.pt` / `.ckpt` to `.safetensors` before running any of the above. |

### Conversion recipe

1. **Inspect source keys** before touching code:

    ```bash
    python - <<'PY'
    import safetensors.torch as st
    keys = list(st.load_file(
        "official_weights/<family>/<file>.safetensors"
    ).keys())
    print(len(keys), "keys")
    for k in keys[:40]:
        print(k)
    PY
    ```

2. **Pick the closest reference script** from the table above.
3. **Copy it to** `scripts/checkpoint_conversion/<family>_to_diffusers.py`
   and rename the family identifier throughout.
4. **Edit `PARAM_NAME_MAP`** — replace with the regex rules that translate
   the official keys to FastVideo / Diffusers naming. Rules apply in
   iteration order; put more-specific patterns first.
5. **Edit `COMPONENT_PREFIXES`** — list every component your pipeline needs
   (transformer, vae, text_encoder, audio_vae, vocoder, upsampler, …) with
   the set of official prefixes that route into each.
6. **Run the conversion** into `converted_weights/<family>/`:

    ```bash
    python scripts/checkpoint_conversion/<family>_to_diffusers.py \
        --source official_weights/<family>/<file>.safetensors \
        --output converted_weights/<family> \
        --class-name <FamilyTransformer3DModel> \
        --pipeline-class-name <FamilyPipeline> \
        --diffusers-version "0.33.0"
    ```

7. **Validate** the output layout:

    ```bash
    ls converted_weights/<family>
    #   model_index.json
    #   transformer/{config.json, diffusion_pytorch_model.safetensors}
    #   vae/{config.json, diffusion_pytorch_model.safetensors}
    #   text_encoder/{config.json, ...}
    ```

8. **Parity-sanity-check converted weights.** Load one component both ways
   and diff a tensor sum. See `validate_longcat_weights.py` for a
   worked-through validator. Do this before writing any FastVideo model
   code — if conversion silently drops keys, all downstream parity work
   will fail mysteriously.
9. **Feed the converted directory into `from_pretrained`** during parity
   testing:

    ```python
    gen = VideoGenerator.from_pretrained("converted_weights/<family>", num_gpus=1)
    ```

10. **(Optional) Publish to HF** so teammates and CI don't need to rerun
    conversion. Use:

    ```bash
    python scripts/checkpoint_conversion/create_hf_repo.py \
        --repo_id "FastVideo/<Family>-Diffusers" \
        --local_dir "converted_weights/<family>" \
        --checkpoint_dir "converted_weights/<family>/transformer" \
        --component_name "transformer" \
        --push_to_hub \
        --upload_repo_id "FastVideo/<Family>-Diffusers"
    ```

    `create_hf_repo.py` downloads the base diffusers repo, injects the
    converted component weights, and (with `--push_to_hub`) uploads the
    result. Once the HF repo exists, add its ID to the `hf_model_paths`
    list in your `register_configs(...)` call — future users skip
    conversion entirely.

### Conversion gotchas

- **Regex iteration order matters.** A loose pattern placed before a
  specific one swallows the specific one. Always test on a dry-run dict.
- **Check for unmapped keys.** After applying `PARAM_NAME_MAP`, diff the
  post-mapping key set against the FastVideo module's `state_dict().keys()`.
  Unmapped keys silently fall through to `load_state_dict(strict=False)`
  and produce noisy output.
- **Sharded checkpoints.** Use `*.safetensors.index.json` + per-shard
  loading (see `_find_shards` in `convert_ltx2_weights.py`) when the
  source is split across multiple files. Don't assume a single file.
- **Dtype preservation.** Many conversion scripts call `.to(torch.bfloat16)`
  on save. Match the DiT's configured `dit_precision` in the pipeline
  config, or you'll get silent upcasts during load.
- **Shape checks.** Some regex rewrites hide shape-mismatch bugs (e.g. a
  fused QKV reshaped wrong). After conversion, load into the FastVideo
  model with `strict=True` at least once and inspect the error message
  carefully — missing/unexpected keys are the only reliable conversion
  verification.

## `register_configs` cheatsheet

Required fields:

- `pipeline_config_cls` — your `PipelineConfig` subclass.
- `workload_types` — tuple of `WorkloadType` values (`T2V`, `I2V`, `V2V`,
  `T2I`, etc.). Use `()` for configs that should not appear as a public
  workload.

Optional but strongly recommended:

- `hf_model_paths` — exact HF IDs this config claims. First match wins.
- `model_detectors` — list of `Callable[[str], bool]` run against the HF
  path AND the `_class_name` from `model_index.json`. Use these for
  prefix/substring patterns when you don't want to enumerate every variant.
- `model_family` — groups presets together.
- `default_preset` — name of the preset used by default when the user does
  not specify one. Must exist in your `ALL_PRESETS`.
- `sampling_param_cls` — only when the family adds new sampling fields.
  Default is the base `SamplingParam` in `fastvideo/api/sampling_param.py`.

## Adding an I2V variant

This section covers adding image-to-video (I2V) support to a family,
based directly on how Wan I2V is structured in the repo. The same
pattern also underlies V2V variants (`WANV2VConfig` just swaps the
image encoder).

**Decision point: same family or new family?**

I2V is a *variant of an existing T2V family*, not a new family. Add it
on top of the T2V port — do not start from scratch. If you have not
yet completed the T2V port per steps 1–16 above, do that first.

**Verified against** (as of 2026-04-24):
- `fastvideo/configs/pipelines/wan.py:73-94` (config inheritance).
- `fastvideo/pipelines/basic/wan/wan_i2v_pipeline.py` (stage wiring).
- `fastvideo/models/dits/wanvideo.py:246-296, 406-457, 83-98, 595`
  (DiT branching on `added_kv_proj_dim`).
- `fastvideo/pipelines/basic/wan/presets.py:87-103`
  (`WAN_I2V_14B_480P` preset).
- `fastvideo/registry.py` (I2V registrations).
- `examples/inference/basic/basic_wan2_2_i2v.py` (user-facing example).

### Files touched for I2V (delta on top of the T2V port)

| Path | Delta |
|------|-------|
| `fastvideo/pipelines/basic/<family>/<family>_i2v_pipeline.py` | **New file** with its own `EntryClass`. Wires the I2V-specific stage order. |
| `fastvideo/configs/pipelines/<family>.py` | **Add** a new `<Family>I2V...Config` class that **inherits from** the corresponding T2V config. |
| `fastvideo/pipelines/basic/<family>/presets.py` | **Add** one or more `InferencePreset`s with `workload_type="i2v"`. Append to `ALL_PRESETS`. |
| `fastvideo/registry.py` | **Add** a second `register_configs(...)` call with `workload_types=(WorkloadType.I2V,)` and the I2V `pipeline_config_cls`. |
| `examples/inference/basic/basic_<family>_i2v.py` | **New example** that passes `image_path=...` to `generate_video(...)`. |
| `fastvideo/models/dits/<family>.py` | **Usually no new file** — the T2V DiT is reused by branching on the config. See "DiT branching" below. |
| `fastvideo/models/vaes/...` | No change. Same VAE serves T2V and I2V. |
| `fastvideo/models/encoders/...` | No new text encoder. Image encoder is typically stock CLIP (`CLIPVisionConfig`) already present in the repo. |

### Config delta — inherit, add image encoder, flip VAE encoder flag

```python
# fastvideo/configs/pipelines/<family>.py
from fastvideo.configs.models.encoders import CLIPVisionConfig

@dataclass
class <Family>I2V480PConfig(<Family>T2V480PConfig):
    """I2V variant: inherits every T2V field, adds CLIP image encoder,
    flips VAE load_encoder on so we can encode the reference image."""

    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True   # I2V NEEDS the encoder
        self.vae_config.load_decoder = True
```

Why `load_encoder=True`: T2V only decodes generated latents at the end,
so it saves memory by skipping the VAE encoder. I2V must encode the
**reference image** into latent space for conditioning, so the encoder
is required. Setting this wrong is a common silent failure — the
pipeline will still run but the image conditioning is empty.

For non-standard image encoders (e.g. Wan's V2V control variant), swap
the field's default:

```python
@dataclass
class <Family>V2VConfig(<Family>I2V480PConfig):
    image_encoder_config: EncoderConfig = field(default_factory=<SpecialVisionConfig>)
    image_encoder_precision: str = "bf16"
```

### Pipeline class delta — new file, two extra stages

Create `fastvideo/pipelines/basic/<family>/<family>_i2v_pipeline.py`.
Copy the T2V pipeline file and apply these deltas:

1. `_required_config_modules` adds `"image_encoder"` and
   `"image_processor"`:

   ```python
   _required_config_modules = [
       "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
       "image_encoder", "image_processor",   # NEW
   ]
   ```

2. Add `ImageEncodingStage` after `TextEncodingStage`, guarded on both
   modules being loaded (so the same pipeline class can still run if
   the user uses a no-image-encoder variant of the family):

   ```python
   if (self.get_module("image_encoder") is not None
           and self.get_module("image_processor") is not None):
       self.add_stage(
           stage_name="image_encoding_stage",
           stage=ImageEncodingStage(
               image_encoder=self.get_module("image_encoder"),
               image_processor=self.get_module("image_processor"),
           ),
       )
   ```

3. Add `ImageVAEEncodingStage` **between** `LatentPreparationStage`
   and `DenoisingStage`:

   ```python
   self.add_stage(
       stage_name="image_latent_preparation_stage",
       stage=ImageVAEEncodingStage(vae=self.get_module("vae")),
   )
   ```

4. **Remove** the `vae=...` kwarg from `DenoisingStage` — the VAE was
   consumed upstream by `ImageVAEEncodingStage`:

   ```python
   # T2V version (keep vae here):
   # DenoisingStage(transformer=..., scheduler=..., vae=...)
   # I2V version (drop vae):
   DenoisingStage(transformer=self.get_module("transformer"),
                  transformer_2=self.get_module("transformer_2", None),
                  scheduler=self.get_module("scheduler"))
   ```

5. `DecodingStage` is unchanged.

6. `EntryClass = <Family>ImageToVideoPipeline` at module level.

### DiT branching — one DiT, two cross-attention branches

In most families, a single DiT file serves both T2V and I2V. The split
happens in the DiT block constructor based on an arch-config flag:

```python
# fastvideo/models/dits/<family>.py (pattern from wanvideo.py:288-297)
if added_kv_proj_dim is not None:
    self.attn2 = <Family>I2VCrossAttention(dim, ...)   # concatenates image K/V
else:
    self.attn2 = <Family>T2VCrossAttention(dim, ...)   # text-only cross-attn
```

The DiT's `forward()` gains an optional `encoder_hidden_states_image`
kwarg; it stays `None` for T2V runs and carries the image embedding
tensor for I2V runs (see `wanvideo.py:83-98` for the canonical
`condition_embedder` pattern that fuses timestep + text + image).

Flag: make `added_kv_proj_dim: int | None = None` default in the DiT
arch config; the I2V variant's DiT config sets it to the CLIP hidden
size (e.g. 1024). This keeps T2V and I2V loading from the same model
class with no code fork.

### Preset — `workload_type="i2v"`, fewer default steps

```python
# fastvideo/pipelines/basic/<family>/presets.py
<FAMILY>_I2V_14B_480P = InferencePreset(
    name="<family>_i2v_14b_480p",
    version=1,
    model_family="<family>",
    description="<Family> I2V 14B at 480p",
    workload_type="i2v",                 # NOT "t2v"
    stage_schemas=(_DENOISE_STAGE,),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 5.0,
        "num_inference_steps": 40,        # Wan I2V defaults to 40; T2V uses 50
        "negative_prompt": _NEGATIVE_PROMPT,
        # Do NOT put image_path in defaults — it's a per-call input.
    },
)
```

Add the new preset(s) to the `ALL_PRESETS` tuple at the bottom of the
file.

### Registry — second `register_configs` call with I2V workload

```python
# fastvideo/registry.py — inside _register_configs()
register_configs(
    sampling_param_cls=None,
    pipeline_config_cls=<Family>I2V480PConfig,
    workload_types=(WorkloadType.I2V,),   # NOT T2V
    hf_model_paths=["Org/<Family>-I2V-Diffusers"],
    model_detectors=[
        lambda path: "<family>" in path.lower() and "i2v" in path.lower(),
    ],
    model_family="<family>",
    default_preset="<family>_i2v_14b_480p",
)
```

If the HF repo's `model_index.json` `_class_name` already uniquely
distinguishes I2V (e.g. `"WanImageToVideoPipeline"`), prefer a detector
that matches on that class name — it's robust against repo-renaming.

Unified T2V+I2V variants (e.g. `Wan2_2_TI2V_5B`) pass both workload
types:

```python
workload_types=(WorkloadType.T2V, WorkloadType.I2V),
```

### User-facing example — `image_path` kwarg

```python
# examples/inference/basic/basic_<family>_i2v.py
from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_<family>_i2v"


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "Org/<Family>-I2V-Diffusers",
        num_gpus=1,
    )

    prompt = "<representative I2V prompt>"
    image_path = "<http URL or local path to a reference image>"

    generator.generate_video(
        prompt,
        image_path=image_path,          # <-- the only API delta vs T2V
        output_path=OUTPUT_PATH,
        save_video=True,
        height=480,
        width=832,
        num_frames=81,
    )


if __name__ == "__main__":
    main()
```

The `image_path` kwarg is resolved to a PIL image internally and
flows through the stage chain (`ImageEncodingStage` →
`ImageVAEEncodingStage`).

### Pipeline parity test for I2V

Follow the [Pipeline-level parity test](#pipeline-level-parity-test-the-gate-for-user-handoff)
pattern with the I2V-specific extras:

- Feed both pipelines the same `(prompt, image_path, seed, height,
  width, num_frames, num_inference_steps)`.
- Compare denoised latents as usual; the image-encoding path will
  contribute its own drift budget, so start at `atol=1e-3` and loosen
  deliberately if needed (log per-step latent sums to prove bounded
  drift first).
- Gate user handoff on a green I2V parity test *and* a green T2V
  parity test — porting I2V without re-running T2V is how silent T2V
  regressions slip in.

## Distributed support (SP for DiTs, TP for LLM encoders)

FastVideo splits the distributed story by component, and the skill's
advice reflects what actually ships in the codebase:

- **DiTs: sequence parallelism (SP) only.** Every DiT in
  `fastvideo/models/dits/` scales across multiple GPUs via SP on the
  spatio-temporal sequence dimension. The SP wiring is already inside
  `DistributedAttention`; your DiT just needs to assert head count is
  divisible by `get_sp_world_size()` and to shard the input sequence
  before the first block / all-gather it after the last. Reference:
  `fastvideo/models/dits/wanvideo.py` (see `sequence_model_parallel_shard`
  near line 657 and `sequence_model_parallel_all_gather_with_unpad` near
  line 718). No `ColumnParallelLinear` / `RowParallelLinear` in DiTs.
- **LLM-style text encoders: tensor parallelism (TP).** `llama.py`,
  `qwen2_5.py`, `siglip.py`, and `t5.py` all use `QKVParallelLinear` +
  `MergedColumnParallelLinear` + `RowParallelLinear`. Copy that
  pattern when your new text encoder is LLM-structured.
- **VAEs: neither, typically.** Parallelism comes from VAE tiling
  (`vae_tiling=True` in `PipelineConfig`) plus optional VAE sequence
  parallelism (`vae_sp=True`). No TP linears, no sequence sharding in
  the VAE model class itself.
- **Ship single-GPU first.** If the DiT is >1-GPU capable, verify SP
  parity in a follow-up; don't block the initial port on it.

Key imports (used by all DiTs):

```python
from fastvideo.distributed import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size
```

## Common pitfalls

1. **`EntryClass` missing or wrong.** Must be a single class at module
   level (not a list). Without it, pipeline discovery silently skips the
   file.
2. **`_required_config_modules` drift.** If it doesn't match
   `model_index.json`, the loader either misses a module or errors on load.
3. **`param_names_mapping` incomplete.** Silent weight-load misses mean
   the model runs but outputs noise. Always diff `state_dict().keys()`.
4. **Scheduler / sigma schedule off-by-one.** Produces subtly wrong videos,
   not obvious errors. Parity-test the schedule itself, not just the DiT.
5. **Attention backend mismatch.** Official repos often use SDPA; align
   backends (`FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`) before chasing
   numerical drift.
6. **Wrong `DistributedAttention` vs `LocalAttention` choice.**
   `DistributedAttention` only for full-sequence self-attention in the
   DiT; everything else (cross-attn, short-seq) is `LocalAttention`.
7. **Forgetting to add the preset import to `_register_presets`.** The
   preset file exists, the pipeline loads, but `SamplingParam.from_pretrained`
   falls back to a generic default — hard to notice without a test.
8. **VAE precision.** Many VAEs need fp32 or bf16 for numerical stability.
   Set `vae_precision` in `PipelineConfig` explicitly.
9. **Detector regex too broad.** `lambda p: "video" in p.lower()` will
   claim every model. Anchor on the family prefix.
10. **Missing `__init__.py` exports.** A new DiT/VAE config that isn't
    re-exported from `fastvideo/configs/models/dits/__init__.py` (or the
    VAE equivalent) breaks imports elsewhere in the codebase.
11. **Raw `torch.nn.Linear` in a hot path.** Skips the quantization /
    weight-loading hooks and drifts from the codebase convention. Use
    `ReplicatedLinear` in DiTs and VAEs, and the TP variants
    (`QKVParallelLinear` / `MergedColumnParallelLinear` /
    `RowParallelLinear`) in LLM-style text encoders — see the
    [Linear layers](#linear-layers-fastvideolayerslinearpy) section for
    the bucket-specific rule. Same "use the fastvideo primitive"
    discipline applies to RMSNorm, GELU, SiLU-and-mul, MLP, RoPE.
12. **Using TP linears in a DiT or VAE.** The opposite mistake: reaching
    for `ColumnParallelLinear` / `QKVParallelLinear` in a DiT because
    it sounded fancier. No DiT or VAE in the repo uses TP linears;
    doing so breaks weight loading for the existing converted HF repos.
    `ReplicatedLinear` is the DiT/VAE default — period.
13. **Official-repo deps overriding FastVideo's pins.** When `pip install`-ing
    the cloned reference repo, a stray `torch==` or `transformers==`
    requirement can silently downgrade FastVideo's environment and produce
    phantom parity failures. Use `--no-deps` and review each extra
    dependency before accepting it.
14. **Subagents widening their scope.** A DiT subagent that also edits the
    pipeline class or the registry breaks step 6's parallelism and
    creates merge headaches. Check every subagent's diff and reject any
    out-of-scope edit.

## Outputs

- A new pipeline reachable via `VideoGenerator.from_pretrained(<hf_path>)`.
- Registry entries (config + preset) committed to `fastvideo/registry.py`.
- Per-component parity tests under `tests/local_tests/<bucket>/` (DiT,
  VAE, encoder(s)).
- **Pipeline-level parity test** at
  `tests/local_tests/pipelines/test_<family>_pipeline_parity.py`, passing
  green (non-skip) against the cloned official repo.
- User-runnable example at `examples/inference/basic/basic_<family>.py`.
- SSIM regression test + seeded reference videos on HF.
- PR description summarizing: reuse decisions, parity deltas, GPU
  requirements, known caveats.

## Example prompt snippet

```
Add FooVideo to FastVideo as a T2V family.

Inputs (required before any work):
  - Official reference repo: github.com/Example/FooVideo
  - HF weights path: Org/FooVideo-Release   (NOT Diffusers-format — single
    foovideo-v1.safetensors at root, no model_index.json. Conversion
    required.)
  - Target model_family: foovideo
  - Workload types: (T2V,)
  - LoRA support: no

The DiT is a flow-matching transformer close to Wan's structure; the VAE
is a custom 3D VAE (unique to this model). Reuse T5 text encoder.

Reference files:
  - Closest existing pipeline: fastvideo/pipelines/basic/wan/wan_pipeline.py
  - Closest existing DiT: fastvideo/models/dits/wanvideo.py
  - Closest existing conversion script: scripts/checkpoint_conversion/convert_ltx2_weights.py
  - SGLang port (if helpful): <url>

Acceptance:
  - Conversion produces converted_weights/foovideo/ that loads via
    VideoGenerator.from_pretrained without missing keys.
  - Per-component parity tests under tests/local_tests/{transformers,
    vaes,encoders}/ pass at atol=1e-4.
  - Pipeline parity test
    tests/local_tests/pipelines/test_foovideo_pipeline_parity.py passes
    (non-skip) against the cloned official repo.
  - examples/inference/basic/basic_foovideo.py runs end-to-end and
    produces a clean mp4.
  - SSIM test registered (references seeded separately via
    seed-ssim-references).
```

## References

- `docs/contributing/coding_agents.md` — narrative guide, LTX-2 case study,
  Wan2.1 worked example. Read this alongside the skill.
- `docs/design/overview.md` — pipeline architecture, configuration system,
  Diffusers weight layout.
- `fastvideo/pipelines/basic/wan/wan_pipeline.py` — minimal standard pipeline.
- `fastvideo/pipelines/basic/ltx2/` — non-standard family with custom
  stages, audio synthesis, multi-stage presets; copy patterns here when
  your model diverges from standard.
- `fastvideo/registry.py` — registration entry points
  (`_register_configs`, `_register_presets`).
- `fastvideo/configs/pipelines/base.py` — `PipelineConfig` base class.
- `fastvideo/pipelines/composed_pipeline_base.py` — pipeline base.
- `fastvideo/pipelines/stages/__init__.py` — catalog of standard stages.
- `fastvideo/layers/` — linears, norms, activations, MLPs, RoPE, visual
  embeddings. See the [FastVideo layers and attention](#fastvideo-layers-and-attention)
  section for selection rules.
- `fastvideo/attention/layer.py` — `DistributedAttention`,
  `DistributedAttention_VSA`, `LocalAttention`. Selection rules in the
  section above.
- `fastvideo/attention/selector.py` — backend selection (never call
  directly from a model file; go through `DistributedAttention` /
  `LocalAttention`).
- `tests/local_tests/transformers/test_ltx2.py`,
  `tests/local_tests/vaes/test_ltx2_vae.py`,
  `tests/local_tests/encoders/test_ltx2_gemma_parity.py` — canonical
  component parity-test templates; see the [Parity test pattern](#parity-test-pattern)
  section.
- `tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py` —
  canonical **pipeline-level** parity-test template (the gate in step 13).
- `tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py` — smoke-test
  template for step 12.
- `tests/local_tests/README.md` — existing convention doc for local parity tests.
- `examples/inference/basic/` — directory of user-runnable example
  scripts. Model on `basic_ltx2.py`, `basic_wan2_2.py`, or the closest
  existing family when creating `basic_<family>.py` in step 13(b).
- `scripts/checkpoint_conversion/` — every conversion script. Start with
  `convert_ltx2_weights.py` as the template; see the
  [Weight conversion](#weight-conversion) section for the full map.
- `scripts/checkpoint_conversion/create_hf_repo.py` — publish a converted
  directory as a Diffusers-format HF repo.
- `scripts/huggingface/download_hf.py` — stage existing HF repos locally
  when no conversion is needed.
- `.agents/skills/seed-ssim-references/SKILL.md` — sibling skill for
  seeding SSIM reference videos once the test file exists.

## Changelog

| Date | Change |
|------|--------|
| 2026-04-24 | Initial draft. Adapted from SGLang's `sglang-diffusion-add-model` skill, rewritten around FastVideo's single stage-based architecture and its registry + preset model. |
| 2026-04-24 | Require official repo URL + HF weights path as explicit gather-inputs step 1. Add "Weight conversion" section mapping `scripts/checkpoint_conversion/` and the conversion recipe. |
| 2026-04-24 | Add HF write-enabled API key as required input (asked at start, never pasted in chat). New step 5 "Clone official repo locally". New step 6 "Port components in parallel via subagents", consolidating former DiT + VAE + encoder steps. New reference sections: "FastVideo layers and attention" (enforces reuse of `fastvideo/layers/` + correct attention-layer selection), "Parity test pattern" (distilled from `tests/local_tests/<bucket>/test_ltx2_*`), "Parallel component porting" (subagent prompt template). Added step 15 cleanup for the temporary clone. Added pitfalls for raw `nn.Linear`, dep overrides, and subagent scope creep. |
| 2026-04-24 | Step 13 is now a gated "full-pipeline parity + example script" checkpoint: (a) mandatory pipeline parity test against the cloned official repo — iterate until green, (b) create user-runnable `examples/inference/basic/basic_<family>.py`, (c) only then hand off to the user. Added pipeline parity test template section. Added example-script + pipeline-parity-test rows to the Files table. |
| 2026-04-24 | **Correction** based on code audit: DiTs and VAEs use `ReplicatedLinear` exclusively; TP linears (`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` / `MergedColumnParallelLinear`) appear only in LLM-style text encoders (`llama.py`, `qwen2_5.py`, `siglip.py`, `t5.py`). Prior draft incorrectly recommended TP linears for DiT QKV/MLP. Linear-layers table rewritten with a per-bucket rule, "Distributed support" section rewritten to match (SP for DiTs, TP for LLM encoders), subagent prompt template and Common pitfalls updated, added pitfall #12 for the inverse mistake (using TP linears in a DiT/VAE). |
| 2026-04-24 | Full-pass verification against Wan2.1 / Wan2.2 code. Canonical stage-order diagram now shows verified T2V and I2V orders side-by-side; ConditioningStage reframed as default-present (not optional). Files-table row 8 notes I2V/V2V/DMD variants get sibling pipeline files. Conversion reference table flags `wan_to_diffusers.py` as a legacy pattern (use `convert_ltx2_weights.py` as the template instead). New "Adding an I2V variant" section added — covers config inheritance, pipeline-file delta (`ImageEncodingStage` + `ImageVAEEncodingStage`, drop `vae` from DenoisingStage), DiT branching on `added_kv_proj_dim`, preset `workload_type="i2v"`, registry delta, example with `image_path`, and I2V pipeline parity test. Uncertainties captured in sibling `REVIEW.md`. |
