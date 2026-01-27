# FastVideo + Coding Agents

Coding agents are now strong at navigating large codebases and iterating fast
with parity tests and examples. This guide shows how to use them to add new
model pipelines and ship PRs in a production-grade video diffusion framework.

FastVideo is a great project to contribute to, with production-grade
infrastructure, active collaborations (including NVIDIA), and a pipeline design
and inference architecture that has been forked by [SGLang’s
multimodal generation stack](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen).

Goal: run the new pipeline with a minimal script like
`examples/inference/basic/basic.py`. In production, FastVideo can download
models automatically via `HF_HOME`; for development, use local directories so
agents can run scripts and tests deterministically. We standardize local paths
as:

- `official_weights/<model_name>/` for official checkpoints
- `converted_weights/<model_name>/` if conversion is required

## Tips when prompting the agent

When prompting the agent, include:

- This guide and the [FastVideo design overview](../design/overview.md).
- Exact file paths to edit.
- A closest reference example file in FastVideo.
- Expected behavior and acceptance criteria.
- Repro steps (command, inputs, logs).
- Constraints (performance, memory, compatibility).
- Local paths (e.g., `official_weights/<model_name>/` or
  `converted_weights/<model_name>/`) for parity tests.

## FastVideo structure at a glance

Before diving in, scan these references:

- [Contributing overview](overview.md) for environment/setup context.
- [FastVideo design overview](../design/overview.md) for pipeline architecture, configs, and HF layout.

FastVideo maps a Diffusers-style repo into a pipeline like:

- `fastvideo/models/*`: model implementations (DiT, VAE, encoders, upsamplers).
- `fastvideo/configs/models/*`: arch configs and `param_names_mapping` for
  weight name translation.
- `fastvideo/configs/pipelines/*`: pipeline wiring (component classes + names).
- `fastvideo/configs/sample/*`: default runtime sampling parameters.
- `fastvideo/pipelines/basic/*`: end-to-end pipeline logic built from stages.
- `model_index.json`: the HF repo entrypoint that maps component names to
  classes and weight files.
- Component loading happens in `VideoGenerator.from_pretrained`, which reads
  `model_index.json`, resolves configs, and loads weights.

Minimal usage example (based on `examples/inference/basic/basic.py`):

```python
from fastvideo import VideoGenerator
from fastvideo.configs.sample import SamplingParam

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # or official_weights/<model_name>/
generator = VideoGenerator.from_pretrained(model_id, num_gpus=1)

sampling = SamplingParam.from_pretrained(model_id)
sampling.num_frames = 45
video = generator.generate_video(
    "A vibrant city street at sunset.",
    sampling_param=sampling,
    output_path="video_samples",
    save_video=True,
)
```

## Some questions to ask yourself before starting

Answering these upfront clarifies the work and speeds up implementation.

### Is the model already supported by SGLang's multimodal generation stack?
If yes, you can port many components from SGLang. It is a FastVideo fork, so
interfaces line up, but you still need to swap layers/modules to match
FastVideo's architecture and attention stack.

If not, implement the model directly in FastVideo.

### Is there an official implementation of the model you are adding?

If yes, use it as the numerical reference. For example, LTX‑2 has an official
implementation here: https://github.com/Lightricks/LTX-2. Prefer official code
even if Diffusers also has one.

### Is there a HuggingFace repo for the model you are adding? Is it in Diffusers format?

If yes, load it directly in FastVideo after setting tensor mapping rules in the
config. Otherwise, convert the weights to Diffusers format. See [Weights and
Diffusers format](../design/overview.md#weights-and-diffusers-format) for details.

### Do I have official weights + local paths ready?

Standardize local paths as:
- `official_weights/<model_name>/` for official checkpoints
- `converted_weights/<model_name>/` if conversion is required (can be created later)

### What pipeline components are required for the model you are adding?

Usually you need a transformer (DiT), VAE, text encoder, and tokenizer. Some
models add extra components.

### What tasks does the model support?

Usually a video diffusion model supports text‑to‑video (T2V),
image‑to‑video (I2V), and video‑to‑video (V2V). Some add extra tasks (two‑stage
generation, keyframe interpolation), which require extra components.

It's usually easiest to start with a T2V pipeline and add the other tasks later.

You can refer to the [Pipeline system](../design/overview.md#pipeline-system)
section for more details.

### Am I able to generate videos with the official implementation?

These videos and prompts are your reference. Once the FastVideo pipeline works,
compare outputs to the official implementation. Due to seeding and other
factors, outputs may not match exactly, but they should be comparable.

## Workflow: adding a full pipeline

This is an example workflow for adding a full model pipeline (model + configs +
examples + tests). This guide is in active development; feedback is welcome.

!!! note
    If you get stuck, refer to existing models/pipelines in FastVideo or ask in Slack.

### 0) Fetch official model's code and weights

Purpose:

- Keep official checkpoints and source code local so conversion, parity tests,
  and reference runs are reproducible.
- Clone the official repo so you can use it as a numerical reference.

Action:

- Download official weights into `official_weights/<model_name>/`
  (Diffusers format or not).
- Clone the official repo under the project root (e.g., `FastVideo/LTX-2/`).
- If a Diffusers-format HF repo already exists, you can skip manual weight
  handling and download it directly with
  `scripts/huggingface/download_hf.py`.

!!! note
    This step is best done manually because large downloads can time out.
    Example:
    ```bash
    python scripts/huggingface/download_hf.py \
      --repo_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
      --local_dir official_weights/Wan2.1-T2V-1.3B-Diffusers \
      --repo_type model
    ```
  
### 1) Implement the model + config mapping

Purpose:

- Model weights are a dictionary of named tensors (`state_dict`). If the names
  don’t line up with FastVideo’s module names, weights won’t load correctly (or
  will silently load into the wrong layer).
- Official checkpoints often use different prefixes or module layouts than
  FastVideo, so we translate names via the mapping (during load or conversion).
- Mapping aligns three things:
  1. the official implementation’s module names,
  2. the checkpoint `state_dict` keys,
  3. FastVideo’s model classes and layer naming conventions.
- If names don’t align, weights won’t load; implement the FastVideo model and
  define mapping rules first.

Action:

- Implement the FastVideo model + config mapping.
  - Add/extend the model in `fastvideo/models/...` and config in
    `fastvideo/configs/models/...` (including `param_names_mapping`).
  - Reuse existing FastVideo layers/modules where possible.
  - Use FastVideo’s attention layers:
    - `DistributedAttention` only for full‑sequence self‑attention in the DiT.
    - `LocalAttention` for cross‑attention and other attention layers.
  - See the “Configuration System” and “Weights and Diffusers format” sections
    in `docs/design/overview.md` for how these pieces connect.
  - If you are using an agent, ask it to implement the model, config mapping,
    and a parity test together so you can validate numerics immediately.

!!! note
    After the first component is aligned and parity‑tested, open a **DRAFT PR**
    on FastVideo so the rest of the pipeline work can build on top of it.

!!! note
    If a Diffusers-format HF repo already exists and loads correctly, you can
    skip conversion entirely (no conversion script needed) and just download it
    with `scripts/huggingface/download_hf.py`. Otherwise, you may need a
    conversion script + a `converted_weights/<model>/` staging directory.

Example (key renaming via arch config mapping, Wan2.1‑style):

```python
# Official model (simplified) in the upstream repo.
class OfficialWanTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Conv3d(16, 1536, kernel_size=2, padding=0)

    def forward(self, x):
        return self.patch_embedding(x)

# FastVideo model (simplified) in fastvideo/models/dits/wanvideo.py
class PatchEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv3d(16, 1536, kernel_size=2, padding=0)

    def forward(self, x):
        return self.proj(x)

class WanTransformer3DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbed()

    def forward(self, x):
        return self.patch_embedding(x)

# Mapping defined in a config (simplified; see the real mapping in
# fastvideo/configs/models/dits/wanvideo.py)
param_names_mapping = {
    r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
}

def apply_regex_map(state_dict, mapping):
    # Pseudocode: apply regex substitutions in order
    ...

# Official checkpoint keys (example)
official = {
    "patch_embedding.weight": ...,
    "blocks.0.attn1.to_q.weight": ...,
}

# Apply mapping so keys match FastVideo modules
converted = apply_regex_map(official, param_names_mapping)

```

Optional helper (print a few checkpoint keys quickly):

```bash
python - <<'PY'
import safetensors.torch as st
keys = list(st.load_file("official_weights/<model>/transformer/diffusion_pytorch_model.safetensors").keys())
print(keys[:20])
PY
```

Example agent prompt (task request):

```
Please add the Wan2.1 T2V 1.3B Diffusers pipeline to FastVideo:
- Add a FastVideo native Wan2.1 DiT implementation + config mapping.
- Make sure to use the existing FastVideo layers and attention modules where possible.
- Add a parity test that loads the official model alongside the FastVideo model and compares outputs numerically with fixed seeds and inputs.

Paths:
  - Official repo: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  - Local download: official_weights/Wan2.1-T2V-1.3B-Diffusers
Mapping steps:
  - Load the official DiT weights from
    official_weights/Wan2.1-T2V-1.3B-Diffusers/transformer/diffusion_pytorch_model.safetensors.
  - Instantiate the FastVideo DiT (`WanTransformer3DModel`) and compare
    its `state_dict().keys()` to the official keys.
  - Update `param_names_mapping` in
    fastvideo/configs/models/dits/wanvideo.py to resolve missing/unexpected keys.
  - Use `load_state_dict(strict=False)` during iteration to surface mismatches.
```

External examples of the same pattern:
- SGLang uses prefix-based routing in its weight loader to map checkpoint keys
  into internal submodules (e.g., stripping a top-level prefix before delegating).
- vLLM includes model-specific renamers for certain checkpoints that adjust
  key prefixes so weights match its internal naming.

### 2) Test numerical alignment with the official implementation

Purpose:

- Verify that the FastVideo component is numerically aligned with the official
  implementation.

Action:

- Add or reuse a numerical parity test that loads the official model and the
  FastVideo model and compares outputs.
- See examples in `tests/local_tests/` (e.g., `tests/local_tests/upsamplers/`)
  and the commands in `tests/local_tests/README.md`.
- If there are discrepancies, add opt‑in logging to both models and compare
  activation summaries (layer output sums, per‑stage logs).
- First align the loaded weights (validate `param_names_mapping`).
- Then align forward outputs using fixed seeds and inputs.
  - Start with `atol=1e-4, rtol=1e-4` in `assert_close`.
  - Keep dtype consistent (bf16 if available; otherwise fp32).
  - If attention parity is unstable, align backends (e.g.,
    `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`).

### 3) Repeat the process for each component

If the model requires additional components, repeat Steps 1–2 for each one.
For example, implement the VAE in `fastvideo/models/vaes/` and its config in
`fastvideo/configs/models/vaes/`, then add parity coverage for it.

### 4) Add a pipeline config + sample defaults

Purpose:

- `fastvideo/configs/pipelines/` describes pipeline wiring and model module
  names.
- `fastvideo/configs/sample/` defines default runtime parameters.

Action:

- Add a new pipeline config + sampling params.
- Register them in `fastvideo/configs/pipelines/registry.py` and
  `fastvideo/configs/sample/registry.py`.

### 5) Wire pipeline stages

Purpose:

- `fastvideo/pipelines/basic/<pipeline>/` contains the actual pipeline logic.
- `fastvideo/pipelines/stages/` holds reusable, testable stages.

Action:

- Build the pipeline using stages; keep new stages isolated and documented.
- Prefer opt‑in flags for expensive or optional steps.

### 6) Add pipeline‑level tests

Purpose:

- Ensure the end‑to‑end pipeline works and stays aligned as pieces evolve.

Action:

- Add a pipeline parity test under `tests/local_tests/pipelines/`.
- See the [Testing Guide](testing.md) for test conventions.

### 7) Add user‑facing examples

Purpose:

- `examples/inference/basic/` is the entry point for simple, runnable scripts.

Action:

- Provide a minimal “hello world” example plus advanced variations.
- Use fixed seeds and stable prompts.
- Run the example locally to confirm end‑to‑end behavior.

### 8) Add SSIM tests for CI checks

Purpose:

- Ensure visual similarity stays within expected bounds for regression testing.
- SSIM tests act as a higher‑level guardrail beyond unit/parity tests.

Action:

- Add SSIM tests under `fastvideo/tests/ssim/` and include reference videos
  (see the structure in the Testing Guide).
- Use stable prompts/seeds and document any GPU‑specific requirements.
- Follow the [Testing Guide](testing.md) for reference video placement and
  execution details.

### 9) Document it

Purpose:

- `docs/` is where users find the new pipeline usage and limitations.

Action:

- Add a short doc page or update an existing one.
- Mention any caveats (memory, speed, constraints).

## Common pitfalls when porting models

- **Attention backend mismatch**: parity can fail if the official model uses a
  different attention backend (e.g., SDPA vs custom). Align backends before
  debugging deeper issues.
- **Patchifier shape mistakes**: wrong patchification or reshape lengths can
  silently corrupt outputs. Validate patch shapes early.
- **Mask handling**: attention masks must match the official behavior (padding,
  causal masks, and broadcast shapes).
- **Scheduler / sigma schedule mismatch**: even small differences in schedules
  or timestep shapes can cause noticeable drift.

## Diffusers vs manual conversion

If a model already ships in Diffusers format (with a proper `model_index.json`),
prefer downloading it directly and loading it via FastVideo. In that case:

- You usually **do not need** a conversion script.
- You still need a correct `param_names_mapping` if the internal module names
  differ from FastVideo’s implementation.

If the model does **not** have a Diffusers-format repo:

- You will need a conversion script to rewrite `state_dict` keys into FastVideo
  naming and stage the result (e.g., under `converted_weights/<model>/`).
- You may still use the official repo for reference parity and debugging.

In both cases, parity testing is required to validate correctness.

If you want to publish a Diffusers‑style repo after conversion, use
`scripts/checkpoint_conversion/create_hf_repo.py` to assemble a HuggingFace‑ready
directory before uploading.

## FAQ

**Q: Why do we implement the FastVideo model before conversion?**  
A: You can’t define the key‑mapping rules until the FastVideo module names are
known. The implementation determines the target `state_dict` schema.

**Q: Do we always need a conversion script?**  
A: No. If a Diffusers‑format repo exists and loads correctly, download it and
skip conversion.

**Q: How do I figure out `param_names_mapping` quickly?**  
A: Load the official weights, instantiate the FastVideo model, and diff
`state_dict().keys()` on both sides. Add regex rules until missing/unexpected
keys are resolved. Agents can help you with this.

**Q: What if parity fails even after mapping?**  
A: Align attention backends, sigma schedules, and timestep shapes first. Then
add opt‑in activation logging to locate the first divergent layer.

## Case study: LTX‑2 port (from PLAN.md)

The LTX‑2 port in `PLAN.md` shows the real sequence of steps and backtracking
that happened during integration. Use it as a reference for how parity work
actually unfolds:

- Ported components first (transformer, VAE, audio, text encoder).
- Added parity tests per component; used SDPA for reference parity.
- Added debug logging to compare per‑block activations and isolate divergence.
- Fixed cross‑attention reshape and patch grid bounds issues after logging.
- Aligned sigma schedule and masking behavior to match the official pipeline.

Recommendation:

- Keep raw step‑by‑step logs in your own local `PLAN.md` for large ports.

## Worked example: Wan2.1 T2V 1.3B pipeline

The Wan2.1 T2V 1.3B Diffusers pipeline is a good “standard” example for
FastVideo integration.

1. Verify model config + mapping.
   - DiT mapping: `fastvideo/configs/models/dits/wanvideo.py`
   - VAE: `fastvideo/models/vaes/wanvae.py`
   - Text encoder: `fastvideo/models/encoders/t5.py`

2. Parity test the core components.
   - Example tests: `fastvideo/tests/transformers/test_wanvideo.py`,
     `fastvideo/tests/vaes/test_wan_vae.py`,
     `fastvideo/tests/encoders/test_t5_encoder.py`

3. Pipeline wiring.
   - Pipeline: `fastvideo/pipelines/basic/wan/wan_pipeline.py`
   - Pipeline config: `fastvideo/configs/pipelines/wan.py`
   - Sampling defaults: `fastvideo/configs/sample/wan.py`

4. Minimal example.
   - Script: `examples/inference/basic/basic.py`
