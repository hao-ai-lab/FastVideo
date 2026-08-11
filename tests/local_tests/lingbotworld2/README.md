# LingBot World 2 Local Tests

Local-only smoke, conversion, and generated-output parity notes for the
`lingbotworld2` causal-fast FastVideo port.

## Reference Assets

| Field               | Value                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Model family        | `lingbotworld2`                                                                                                   |
| First scope         | `robbyant/lingbot-world-v2-14b-causal-fast`, I2V causal-fast inference                                            |
| Workload types      | I2V: text + first image + camera/action path to video                                                             |
| Official reference  | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2`                                                                 |
| HF weights          | `robbyant/lingbot-world-v2-14b-causal-fast`                                                                       |
| Raw weights         | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/ckpts/lingbot-world-v2-14b-causal-fast`                          |
| FastVideo bundle    | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/ckpts/lingbot-world-v2-14b-causal-fast-fastvideo`                |
| Source layout       | `raw_official`                                                                                                    |
| Needs conversion    | `yes`                                                                                                             |
| Token env var       | none recorded                                                                                                     |

Do not write token values in this file.

## Shared Environment Setup

Use the existing venv and do not rebuild FlashAttention:

```bash
/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/pip install -r /mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/requirements.txt
/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/pip install --no-deps -e /mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot
```

The current environment already has the FastVideo editable install pointing at:

```text
/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot
```

Runtime cache and temporary paths used by the matrix scripts:

```text
/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/.cache
```

## Conversion

The converter writes a FastVideo-loadable bundle with native component configs,
the exact LingBot World 2 Wan VAE checkpoint, and symlinked raw LingBot World 2 transformer/T5
assets:

```bash
cd /mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot
/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python scripts/checkpoint_conversion/convert_lingbotworld2_causal_fast.py \
  --source-dir /mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/ckpts/lingbot-world-v2-14b-causal-fast \
  --output-dir /mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/ckpts/lingbot-world-v2-14b-causal-fast-fastvideo
```

Strict-load checks already run locally:

| Component    | Status | Notes                                                                  |
| ------------ | ------ | ---------------------------------------------------------------------  |
| VAE          | pass   | `LingBotWorld2WanVAE` loads the original `Wan2.1_VAE.pth` checkpoint.  |
| Text encoder | pass   | Native `LingBotWorld2T5EncoderModel` strict-load from symlinked `.pt`. |
| Transformer  | pass   | Full 14B load and 8-GPU causal-fast generation pass.                   |

## Official Frame-Count Constraint

The requested verification frame counts are `5, 9, 17, 33, 65`. The official
causal-fast source uses default `chunk_size=4`, then truncates latent frames to a
multiple of the chunk size. This makes `5` and `9` source-impossible and changes
the generated frame counts for the remaining values.

| Requested frames | Official effective frames | Status                         |
| ---------------- | ------------------------- | ------------------------------ |
| `5`              | `-3`                      | official source fails          |
| `9`              | `-3`                      | official source fails          |
| `17`             | `13`                      | runs with truncated output     |
| `33`             | `29`                      | runs with truncated output     |
| `65`             | `61`                      | runs with truncated output     |

Evidence: `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py`
rounds `frame_num` to `4n+1`, computes latent frames, truncates `lat_f` by
`chunk_size`, and resets `F = (lat_f - 1) * 4 + 1`.

## Local Tests

```bash
cd /mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot
/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python -m pytest \
  tests/local_tests/lingbotworld2/test_lingbotworld2_causal_fast_preflight.py \
  tests/local_tests/pipelines/test_lingbotworld2_causal_fast_pipeline_smoke.py \
  tests/local_tests/pipelines/test_lingbotworld2_causal_fast_pipeline_parity.py \
  -q -rs
```

Current result:

```text
2026-07-10: 5 passed, 1 skipped.
Skipped test: opt-in 14B heavy smoke gated by LINGBOTWORLD2_RUN_HEAVY_SMOKE=1.
```

Heavy smoke is opt-in because it loads the 14B model:

```bash
cd /mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot
LINGBOTWORLD2_RUN_HEAVY_SMOKE=1 /mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python -m pytest \
  tests/local_tests/pipelines/test_lingbotworld2_causal_fast_pipeline_smoke.py \
  -q -rs
```

Current heavy result:

```text
2026-07-10: 2 passed in 3:15 on 8 H200 GPUs.
```

## Matrix Verification

Reference generation:

```bash
/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/scripts/run_reference_matrix.sh
```

FastVideo generation:

```bash
/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python \
  /mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/scripts/run_fastvideo_matrix.py
```

Decoded-frame comparison:

```bash
/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python \
  /mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/scripts/compare_matrix_outputs.py
```

Current comparison result:

```text
2026-07-09: all 9 official-supported FastVideo videos are exactly equal to the
reference videos after decoded-frame comparison: max abs 0, mean abs 0.0.
The requested 5-frame and 9-frame cases are recorded as official_unsupported for
both reference and FastVideo.
```

Expected output roots:

| Output type  | Path                                                                       |
| ------------ | -------------------------------------------------------------------------- |
| Reference    | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/outputs/reference`        |
| FastVideo    | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/outputs/fastvideo`        |
| Verification | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/outputs/verification`     |

Generated-output parity test:

```bash
cd /mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot
/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python -m pytest \
  tests/local_tests/pipelines/test_lingbotworld2_causal_fast_pipeline_parity.py \
  -q -rs
```
