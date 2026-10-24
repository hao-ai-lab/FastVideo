# Bernini-R parity tests

These tests compare FastVideo's Bernini-R 1.3B integration with the official
implementation. Run them from the repository root in a CUDA development
environment.

## Reference setup

The oracle uses these immutable releases:

- [ByteDance/Bernini](https://github.com/bytedance/Bernini), commit
  `e6c2cf11843470b08491745947abcba9c3c6b079`.
- `ByteDance/Bernini-R-1.3B-Diffusers`, revision
  `ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce`.
- Diffusers `0.35.2` and Transformers `4.57.3` for the official implementation.

```bash
git clone https://github.com/bytedance/Bernini.git reference_repos/bernini
git -C reference_repos/bernini checkout e6c2cf11843470b08491745947abcba9c3c6b079

hf download ByteDance/Bernini-R-1.3B-Diffusers \
  --revision ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce \
  --local-dir official_weights/bernini-r-1.3b

python scripts/checkpoint_conversion/bernini_r_to_diffusers.py \
  --source official_weights/bernini-r-1.3b \
  --output converted_weights/bernini-r-1.3b

uv pip install --target reference_repos/bernini/.oracle-deps \
  diffusers==0.35.2 transformers==4.57.3 decord==0.6.0
```

The local helper imports the official numerical modules directly. Their
attention, packing, APG and sampler remain the comparison implementation.
The isolated dependency directory supplies the reference scheduler version.

## Component checks

Select an available GPU and run the weight-free contracts before the
real-checkpoint comparisons:

```bash
export CUDA_VISIBLE_DEVICES=0
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_FA4=0

pytest fastvideo/tests/loader/test_bernini_checkpoint.py \
  fastvideo/tests/api/test_bernini.py \
  fastvideo/tests/api/test_bernini_ssim.py \
  fastvideo/tests/stages/test_bernini_stages.py \
  fastvideo/tests/transformers/test_bernini.py -v

pytest tests/local_tests/bernini/test_official_parity.py -v -s
```

The oracle tests cover source-ID RoPE, APG, every UniPC update in 3/10/50-step
schedules, strict checkpoint loading, and FP32/BF16 transformer outputs with
intermediate probes at layers 1/5/10/20/30. The reference owns its patching
and unpatching. Missing local assets are reported as skips.

The numerical contracts include separate source/target grids with source
IDs 1/0, `(pT,pH,pW,C)` patch order, per-frame APG over C/H/W, float64 RoPE
tables and projection, FP32 time/norm/scale-shift exceptions, and 512-token
text padding.

## Pipeline comparison

Each implementation runs in a separate process. Both commands use the same
input video, prompt, seed and sampling arguments:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  tests/local_tests/bernini/run_pipeline_parity.py --implementation native

PYTHONPATH=reference_repos/bernini/.oracle-deps \
  python tests/local_tests/bernini/run_pipeline_parity.py --implementation official

python tests/local_tests/bernini/run_pipeline_parity.py \
  --compare outputs_bernini_parity
```

Defaults are the official `source_case1.mp4`, a watercolor edit, seed 42,
192x352, 17 frames, 16 fps, 40 steps and an empty negative prompt.
The harness compares source latents, positive/negative text embeddings,
final latents and decoded frames. It requires source relative L2 below
`1e-5`, text relative L2 below `1e-3`, and decoded MS-SSIM at least `0.98`.
Saved artifacts include tensors, MP4, a first/middle/last-frame preview and
a JSON comparison.

For the full snowman-edit recipe, append the following arguments to both
implementation commands, then compare `outputs_bernini_full`:

```bash
--prompt-file reference_repos/bernini/assets/testcases/v2v/v2v_case1.json \
--height 480 --width 848 --frames 81 --steps 40 --output outputs_bernini_full
```

The 480x848 dimensions match the official aspect-preserving resize with
maximum edge 848. Keep checkpoints, the reference checkout and generated
artifacts outside the Git change set.

## SSIM regression

The CI-discovered regression uses the same pinned snowman-edit recipe:

```bash
FASTVIDEO_TEST_BERNINI_MODEL_DIR=official_weights/bernini-r-1.3b \
  pytest fastvideo/tests/ssim/test_bernini_similarity.py -vs
```

References are keyed by GPU, backend and quality tier. See the
[inference guide](../../../docs/inference/bernini_r.md) and
[SSIM workflow](../../../docs/contributing/testing.md#ssim-tests) for
reference preparation and CI execution.
