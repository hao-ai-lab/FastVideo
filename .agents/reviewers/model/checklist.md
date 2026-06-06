# Model reviewer checklist

Each item resolves to ✅ / ❌ / ⚠️ / N/A in the final report.

## New model (`[new-model]` or adds a new DiT/VAE/encoder)

- [ ] DiT under `fastvideo/models/dits/<model>.py` inherits from
      `fastvideo/models/dits/base.py`.
- [ ] Arch config under `fastvideo/configs/models/dits/<model>.py` includes
      `param_names_mapping`.
- [ ] `param_names_mapping` covers every HF key referenced in the linked
      model card / safetensors index.
- [ ] Pipeline under `fastvideo/pipelines/basic/<model>/` composed from
      stages in `fastvideo/pipelines/stages/`.
- [ ] Pipeline + config registered in `fastvideo/registry.py`.
- [ ] Loader logic in `fastvideo/models/loader/` handles the model's
      safetensors layout (or reuses an existing loader).
- [ ] SSIM test at `fastvideo/tests/ssim/test_<model>_similarity.py`.
- [ ] Reference videos uploaded to `FastVideo/ssim-reference-videos` (via the
      `seed-ssim-references` skill).
- [ ] Minimal example at `examples/inference/basic/<model>.py`.
- [ ] `docs/` support matrix / supported-models list updated if one exists.
- [ ] PR body has test output (not just a command).

## Modifying an existing model

- [ ] PR body attaches SSIM regression output for the affected pipeline.
- [ ] If renaming a module, `param_names_mapping` updated consistently across
      every variant (causal / non-causal / i2v / t2v).
- [ ] No new `.float()` / `.half()` / `.bfloat16()` without justification.
- [ ] SP/TP-safe: any reshape/einsum on the sequence dimension uses the
      correct parallel group.
- [ ] No silent API break in public model classes (check `__init__`
      signatures and exported symbols).

## SSIM test changes

- [ ] Any change to `min_acceptable_ssim` thresholds is called out explicitly
      in the PR body with a reason. **Silent threshold drops weaken the
      regression guard — treat as BLOCKER unless justified.**
- [ ] If the PR body cites a reference test (e.g. "follows
      `test_turbodiffusion_similarity.py`"), diff the new test against it and
      confirm the orchestrator-contract fields (`REQUIRED_GPUS`,
      `*_MODEL_TO_PARAMS`, helper usage) match.
- [ ] Test prompt list has >1 prompt (single-prompt tests are one hash change
      away from false-green).
- [ ] New test uses shared helpers (`resolve_inference_device_reference_folder`,
      `run_text_to_video_similarity_test`) — not inline reimplementation.

## Shared-helper changes (cross-file consistency)

- [ ] If the PR modifies a shared helper (e.g. adds a supported device to a
      device-folder tuple, changes a public-helper signature), every inline
      duplicate of that pattern in peer files is also updated. Example:
      adding `B200` to `inference_similarity_utils.py` must also update
      inline device tuples in any `test_*_similarity.py` that still holds
      its own copy.

## Arch config

- [ ] Default-arg changes in `@dataclass` configs are explicitly documented in
      the PR body.
- [ ] `param_names_mapping` edits correspond to a matching DiT code change.

## Pipeline (stages or basic)

- [ ] If `fastvideo/pipelines/stages/` was edited, PR body documents which
      per-model pipelines are affected and shows SSIM for at least one of them.
- [ ] Per-model pipeline doesn't bypass `build_pipeline` / registry.

## Tests

- [ ] SSIM test naming matches the `test_<model>_similarity.py` convention.
- [ ] Encoder/transformer unit tests added if a new encoder or transformer
      type was introduced.
- [ ] GPU assumptions documented in docstrings or conftest (per `AGENTS.md`).

## PR template compliance

- [ ] Title starts with a valid tag (`[feat]` / `[new-model]` / `[bugfix]` /
      `[perf]` / etc).
- [ ] "For model/pipeline changes" checkbox in PR template is checked.
- [ ] Purpose + Changes sections are filled out, not the template placeholder.
