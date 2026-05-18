# `fastvideo/pipelines/` — Pipeline Composition

**Generated:** 2026-05-02

Diffusion pipelines are **compositions of `PipelineStage` objects**. Each stage owns one verb (validate / encode / schedule / denoise / decode). Adding a model means assembling stages, not subclassing a megapipeline.

## Layout

```
pipelines/
├── pipeline_batch_info.py      # ForwardBatch — the dict passed between stages
├── lora_pipeline.py            # LoRA-aware base
├── composed_pipeline_base.py   # Base for stage-composed pipelines
├── stages/                     # Reusable stage implementations (~30 files)
│   ├── base.py                 #   PipelineStage ABC + StageVerificationError
│   ├── input_validation.py     #   Validates ForwardBatch shape/keys
│   ├── text_encoding.py        #   Generic prompt encoder stage
│   ├── image_encoding.py       #   Image conditioning
│   ├── latent_preparation.py   #   Init noise + scheduler
│   ├── conditioning.py         #   CFG / negative prompt fan-out
│   ├── denoising.py            #   Standard diffusion loop
│   ├── sd35_conditioning.py    #   Per-model overrides (named by family)
│   ├── longcat_*.py            #   LongCat I2V/V2V/refine variants
│   ├── gen3c_stages.py         #   Gen3C-specific stages
│   ├── gamecraft_denoising.py  #   GameCraft-specific
│   └── matrixgame2_denoising.py #  Matrix-Game 2.0-specific
├── basic/                      # Per-model end-to-end pipelines
│   ├── hunyuan/, hunyuan15/, hyworld/, gamecraft/, gen3c/, cosmos/
│   ├── wan/, longcat/, ltx2/, lingbotworld/, magi_human/, matrixgame2/
│   ├── sd35/, stable_audio/, turbodiffusion/
│   └── <model>/{<model>_pipeline.py, presets.py, __init__.py}
├── preprocess/                 # Data preprocessing pipelines (ltx2, wan, matrixgame2)
└── training/                   # Training-time pipeline glue
```

## Stage Authoring Rules

- Subclass `PipelineStage` from `stages/base.py`. Implement `forward(batch, args) -> ForwardBatch`.
- Implement `verify_input` / `verify_output` — both return `VerificationResult`. Failures raise `StageVerificationError`.
- Mutate `ForwardBatch` only by reassigning fields you declared in `pipeline_batch_info.py`. New keys → add to the dataclass first.
- Stages must be **deterministic given the same `ForwardBatch + FastVideoArgs`**. Side effects (logging, profiling) only.
- Read all knobs from the passed-in `FastVideoArgs` / `PipelineConfig`. Never `os.getenv` directly.

## Per-Model Pipeline Pattern (`basic/<model>/`)

Every model directory has the same skeleton:

```
basic/<model>/
├── __init__.py
├── <model>_pipeline.py      # Composes stages list
├── presets.py               # Default PipelineConfig + SamplingParam combos
└── (optional) stage_overrides.py, continuation.py, ...
```

`presets.py` is the entry point that `registry.py` imports — it must export the named preset constants used elsewhere in the codebase.

## Forking vs Reusing a Stage

Reuse `stages/text_encoding.py` if your model takes text → embeddings via a standard encoder. Fork only when:

- The model needs a **different ForwardBatch shape** (extra inputs, different output keys).
- The denoising loop has structural differences (causal, refine-then-denoise, multi-stream).

When forking, keep the file name model-prefixed (`longcat_*`, `gamecraft_*`) so the registry stays grep-able.

## Per-Package `AGENTS.md` and `JOURNAL.md` (optional)

Pipelines with non-trivial parity invariants, lazy-loaded shared components,
or cross-component coordination MAY ship a per-package `AGENTS.md` and
`JOURNAL.md` next to the pipeline files. The canonical example is
`basic/magi_human/AGENTS.md` (umbrella HF repo, four lazy-loaded shared
components, channel-major packing + dtype-boundary invariants).

When present:

- `basic/<model>/AGENTS.md` — manifest table of every file in scope, parity
  invariants, "if you change X re-run Y" cross-refs, run book, open
  questions, provenance (PR numbers + source SHAs).
- `basic/<model>/JOURNAL.md` — port-state log for the original port; useful
  for future maintainers to understand why specific decisions were made.

These files are **not** required for simple ports that share stages and have
no special parity invariants.

## Anti-Patterns

- Putting a full pipeline in a single file under `basic/<model>/` instead of composing stages.
- Reading config from globals or env vars inside a stage.
- Adding cross-stage state via module-level dicts. Use `ForwardBatch`.
