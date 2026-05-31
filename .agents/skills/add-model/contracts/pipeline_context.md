# Pipeline Context Contract

Canonical packet passed from `/add-model` to `add-model-09-pipeline` for pipeline
definition and pipeline parity-debug work.

```text
pipeline_context:
  model_family: <snake_case>
  mode: pipeline-definition | pipeline-parity-debug
  workload_types:
    - <T2V|I2V|V2V|T2I|compatibility-shim-with-rationale>
  modalities:
    inputs: <text/image/video/audio/pose/depth/mask/etc.>
    outputs: <video/image/audio/joint-av/latents/etc.>
  official_ref_dir: <path or import path>
  official_pipeline_files:
    - path: <repo-relative or absolute path in official repo>
      symbols: <pipeline/factory/sample functions>
      notes: <stage order, mutable state, output contract>
  official_call:
    command_or_api: <official CLI, Python call, or package entrypoint>
    args_and_defaults: <height, width, frames, fps, duration, steps, CFG, scheduler, seeds, etc.>
    preset_source: <model card, config file, official script, or unknown>
    scheduler_and_rng: <timestep/sigma/noise/generator behavior>
    output_contract: <decoded media, denoised latents, waveform, dict keys, etc.>
  model_index:
    class_name: <FastVideo pipeline class name to emit in model_index.json>
    entry_class_names: <registered EntryClass.__name__ values that must include class_name>
    required_modules: <text_encoder, tokenizer, vae, transformer, scheduler, etc.>
    passthrough_modules: <tokenizer, scheduler, processor, external HF dirs, or none>
  sampling_param:
    new_fields: <none or list of public kwargs/preset defaults to add to SamplingParam>
    cli_fields: <none or list of fields that need CLI args>
    placeholder_fields: <none or video-shaped compatibility placeholders with rationale>
  components:
    - name: <component>
      component_type: <dit|vae|encoder|scheduler|conditioner|upsampler|vocoder|generic>
      parity_test: tests/local_tests/<bucket>/test_<family>_<component>_parity.py
      parity_status: non_skip_pass
      fastvideo_target_files: <model/config/export files>
      converted_component_dir: converted_weights/<family>/<component>
  conversion:
    converted_weights_dir: converted_weights/<family>
    source_layout: <diffusers|raw_official|monolithic|separate_components|mixed|custom>
    model_index_path: converted_weights/<family>/model_index.json
  fastvideo_targets:
    pipeline_files:
      - fastvideo/pipelines/basic/<family>/<family>_pipeline.py
    stage_files:
      - fastvideo/pipelines/basic/<family>/stages/<stage>.py
    pipeline_config_files:
      - fastvideo/configs/pipelines/<family>.py
    preset_file: fastvideo/pipelines/basic/<family>/presets.py
    registry_file: fastvideo/registry.py
    example_files:
      - examples/inference/basic/basic_<family>.py
    smoke_test: tests/local_tests/pipelines/test_<family>_pipeline_smoke.py
    parity_test: tests/local_tests/pipelines/test_<family>_pipeline_parity.py
  local_tests_readme: tests/local_tests/<model_family>/README.md
  port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
  concerns_or_unknowns:
    - <pipeline branch, unsupported workload, output head, preset ambiguity, etc.>
```

Rules:

- Start only after every required component, reused or ported, has
  `parity_status=non_skip_pass`. If any row is missing or skipped, return to
  `/add-model` Phase 6.
- Record official call arguments and default sources before writing FastVideo
  presets. Do not invent inference defaults from memory.
- `model_index.class_name` must match a registered pipeline `EntryClass.__name__`;
  registry detectors are not sufficient for executable pipeline resolution.
- Every public generation kwarg or preset default must be represented in
  `SamplingParam`, or documented as an intentional internal-only field.
- `T2A`, `A2A`, and `AV` may be used only after `WorkloadType` supports them;
  otherwise record the compatibility shim and rationale explicitly.
- Keep token values out of the packet. Use only token environment variable names.
- If a target path is unknown, use `unknown` plus the exact search already
  performed.
- Update `local_tests_readme` and `port_state_file` whenever pipeline smoke,
  parity, presets, registry, examples, or blockers change.
