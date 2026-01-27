# FastVideo + Coding Agents

Coding agents have become extremely capable at navigating large codebases and
iterating quickly using parity tests and examples.  This tutorial is a step by
step guide for using coding agents to add new model pipelines and ship
meaningful PRs in a production-grade video diffusion inference and training
framework.

FastVideo itself is a great project to contribute to, with production-grade
infrastructure and CI/CD, active collaborations (including NVIDIA), and a
pipeline design and inference architecture that has been forked by [SGLang’s
multimodal generation stack](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen).

Goal: after completing this workflow, you should be able to run the new pipeline
with a minimal script similar to `examples/inference/basic/basic.py`.

## Tips when prompting the agent

When prompting the agent, include:

- The exact file paths to edit.
- A reference to an existing example file for another model in FastVideo. This drastically improves the agent's ability to understand the model and the pipeline requirements.
- Expected behavior and any acceptance criteria.
- How to reproduce the issue (command, inputs, logs).
- Constraints (performance, memory, compatibility).

## FastVideo architecture context

Before diving into the workflow, read the following documents:

- [Contributing overview](overview.md) for environment/setup context.
- [FastVideo design overview](../design/overview.md) for pipeline architecture, configs, and HF layout.

## Some questions to ask yourself before starting
Below are some questions that you should answer before starting the workflow. It will help you understand the model and the pipeline requirements better and significantly make the workflow easier.

- Is the model you are adding supported by SGLang's multimodal generation stack already?
If so, you can refer to SGLang's codebase and port many of the components to FastVideo. Note that SGLang's multimodal generation stack is a fork of FastVideo and has many of the same components and features. You will need to change the layers and modules to match FastVideo's architecture.

If not, you can start from scratch and implement the model from scratch.

- Is there an official implementation of the model you are adding? If so, you can use it as a reference to implement the model in FastVideo. For example LTX-2 has an official implementation here: https://github.com/Lightricks/LTX-2. We prefer to align numerically against the official implementation even if Diffusers also has an implementation of the model.

- Is there a HuggingFace repo for the model you are adding? Is it in Diffusers format?
If so, you can directly use it to load the model in FastVideo, after setting the appropriate tensor mapping rules in the config. Otherwise you will need to convert the weights to the Diffusers format. More details on this in the [Weights and Diffusers format](../design/overview.md#weights-and-diffusers-format) section.

- What pipeline components are required for the model you are adding?
Usually a video diffusion model pipeline requires a transformer model (DiT), a VAE, a text encoder, and a tokenizer. But specific models may require additional components.

- What tasks does the model support?
Usually a video diffusion model supports text-to-video generation (T2V), image-to-video generation (I2V), and video-to-video generation (V2V). But specific models may support additional tasks (e.g., 2-stage generation, keyframe interpolation, etc.). Each of these tasks may require additional components.

It's usually easiest to start with a T2V pipeline and then add the other tasks later.

You can refer to the [Pipeline architecture](../design/overview.md#pipeline-architecture) section for more details.

- Am I able to generate videos with the official implementation?
These videos and prompts are a good reference to check if the model is working correctly.
And once the FastVideo pipeline is working, you can compare the outputs with the official implementation to ensure that they are similar in quality. Due to seeding and other factors, the outputs may not be exactly the same, but they should be similar in quality.

## Workflow: adding a full pipeline

This is an example workflow for adding a full model pipeline (model +
configs + examples + tests) to FastVideo. This guide is in active development so
any suggestions or improvements are welcome.

!!! note
    Remember if you have any doubts about the implementation, you can always refer to existing models and pipelines in FastVideo. You can also ask for help from the community or the maintainers in our Slack channel.

### 0) Fetch official model's code and weights

Purpose:

- Keep official checkpoints and source code local so conversion, parity tests,
  and reference runs are reproducible.
- By cloning the official repo, we can use the official implementation to verify
  the conversion is correct.

Action:

- Download official weights (Diffusers format or not) into `official_ltx_weights/` (or a model-specific
  folder under the project root).
- Clone the official repo under the project root (e.g., `FastVideo/LTX-2/`).
- If a Diffusers-format HF repo already exists, you can skip manual weight
  handling and download it directly with
  `scripts/huggingface/download_hf.py`.

!!! note
    This step is most easily done manually as some downloads could take a long
    time and cause timeouts.
  
### 1) Convert and place weights

Purpose:

- Model weights are just a big dictionary of named tensors (`state_dict`).
  If the names don’t line up with FastVideo’s module names, the weights won’t
  load correctly (or will silently load into the wrong layer).
- Official checkpoints often use different prefixes or module layouts than
  FastVideo, so we translate the names during conversion.
- Conversion aligns three things:
  1) the official implementation’s module names,
  2) the checkpoint `state_dict` keys,
  3) FastVideo’s model classes and layer naming conventions.
- `converted/` is the local staging area for the aligned components in
  diffusers-style folders (`config.json` + `model.safetensors`).
- However we need to know the correct tensor names to use, which means we
  must implement the FastVideo model first and define its mapping rules.

Action (recommended order):

1) Implement the FastVideo model wrapper + config mapping first.
   - Add/extend the model definition in `fastvideo/models/...` and its config in
     `fastvideo/configs/models/...` (including any rename map for keys).
   - Remember to reuse existing layers and modules from FastVideo where possible
     and only add new ones if necessary.
   - Use FastVideo’s attention layers:
     - `DistributedAttention` only for full‑sequence self‑attention in the DiT.
     - `LocalAttention` for cross‑attention and other attention layers
       (including text encoders).
   - See the “Configuration System” and “Weights and Diffusers format” sections
     in `docs/design/overview.md` for how these pieces connect.
2) Write a parity test that loads the official model + FastVideo model and
   compares outputs numerically (ideally with fixed seeds).
   - See examples in `tests/local_tests/` (e.g., `tests/local_tests/upsamplers/`).
3) If needed, add the conversion script (or update an existing one) to rewrite
   `state_dict` keys to the FastVideo naming, then save into `converted/`.

!!! note
    Note that the converted weights are temporary and eventually we can create a
    new HuggingFace repo for the converted model, in Diffusers format, and upload
    it to the HuggingFace Hub.
    If a Diffusers-format HF repo already exists and loads correctly, you can
    skip conversion entirely (no conversion script needed) and just download it
    with `scripts/huggingface/download_hf.py`.

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
class WanTransformer3DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Conv3d(16, 1536, kernel_size=2, padding=0)

    def forward(self, x):
        return self.patch_embedding(x)

# Mapping defined in a config (simplified)
param_names_mapping = {
    r"^model\.(.*)$": r"\1",
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

Example agent prompt (task request):

```
Please add the Wan2.1 T2V 1.3B Diffusers pipeline to FastVideo:
- Add a FastVideo native Wan2.1 DiT implementation + config mapping.
- Make sure to use the existing FastVideo layers and attention modules where possible.
- Add a parity test that loads the official model alongside the FastVideo model and compares outputs numerically with fixed seeds and inputs.

Paths:
  - Official repo: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  - Local download: weights/Wan2.1-T2V-1.3B-Diffusers
Mapping steps:
  - Load the official DiT weights from
    weights/Wan2.1-T2V-1.3B-Diffusers/transformer/diffusion_pytorch_model.safetensors.
  - Instantiate the FastVideo DiT (`WanTransformer3DModel`) and compare
    its `state_dict().keys()` to the official keys.
  - Update `param_names_mapping` in
    fastvideo/configs/models/dits/wanvideo.py to resolve missing/unexpected keys.
  - Use `load_state_dict(strict=False)` during iteration to surface mismatches.
```

External examples of the same pattern:
- SGLang shows `load_weights` routing by prefix, stripping `llm.` before
  forwarding to a submodule loader. This is a runtime mapping of checkpoint
  names to SGLang’s internal module layout.
- vLLM’s `mllama4` model includes a rename helper that rewrites
  `model.*` → `language_model.model.*` and remaps a few scale parameter
  names, so ModelOpt checkpoints match vLLM’s internal naming.

### 2) Test numerical alignment with the official implementation
Purpose:
- Verify that the FastVideo model is numerically aligned with the official implementation.

Action:
- Add or use the existing numerical parity test that loads the official model + FastVideo model and
  compares outputs numerically.
- See examples in `tests/local_tests/` (e.g., `tests/local_tests/upsamplers/`).
- If the component has discrepancies, detailed logging in both the FastVideo model and the official model to debug the issue.
- First align the loaded weights numerically, making sure the `param_names_mapping` in the config is correct.
- Then align the forward pass outputs numerically. Print the sum of the model activations after each layer to debug the issue. Log the activations of the FastVideo model and the official model side by side in two files and have the agents continuously run and debug the issue.

### 3) Repeat the process for each component
If the model requires additional components, you can repeat the process for each component.
For example, if the model requires a VAE, you can implement the VAE in `fastvideo/models/vaes/` and its config in `fastvideo/configs/models/vaes/`.
You can then repeat the process for the other components.

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

### 6) Add tests and parity checks

Purpose:
- `tests/local_tests/` is where we keep local parity tests and component checks.

Action:
- Add a minimal component parity test (weights + output match).
- Add a pipeline parity test if applicable (stage sums or output mean).
- Gate tests via env vars so they can be skipped without weights.

### 7) Add user‑facing examples

Purpose:
- `examples/inference/basic/` is the entry point for simple, runnable scripts.

Action:
- Provide a minimal “hello world” example plus advanced variations.
- Use fixed seeds and stable prompts.

### 8) Document it

Purpose:
- `docs/` is where users find the new pipeline usage and limitations.

Action:
- Add a short doc page or update an existing one.
- Mention any caveats (memory, speed, constraints).

## Worked example: Wan2.1 T2V 1.3B pipeline

The Wan2.1 T2V 1.3B Diffusers pipeline is a good “standard” example for
FastVideo integration.

1) Verify model config + mapping.
   - DiT mapping: `fastvideo/configs/models/dits/wanvideo.py`
   - VAE: `fastvideo/models/vaes/wanvae.py`
   - Text encoder: `fastvideo/models/encoders/t5.py`

2) Parity test the core components.
   - Example tests: `fastvideo/tests/transformers/test_wanvideo.py`,
     `fastvideo/tests/vaes/test_wan_vae.py`,
     `fastvideo/tests/encoders/test_t5_encoder.py`

3) Pipeline wiring.
   - Pipeline: `fastvideo/pipelines/basic/wan/wan_pipeline.py`
   - Pipeline config: `fastvideo/configs/pipelines/wan.py`
   - Sampling defaults: `fastvideo/configs/sample/wan.py`

4) Minimal example.
   - Script: `examples/inference/basic/basic.py`

## Review hygiene

- Summarize what changed and why.
- Call out any remaining risk, missing coverage, or known limitations.
- If the change was forced by a limitation, note it explicitly.

## Safety

- Never delete data or reset history unless explicitly instructed.
- When uncertain about a requested change, ask for clarification.
