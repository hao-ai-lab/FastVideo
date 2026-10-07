# MMAudio dataset inference and evaluation

This example evaluates a trained MMAudio checkpoint without importing the
upstream MMAudio package. Generation uses FastVideo's native
`MMAudioPipeline`; the default metrics use `fastvideo.eval`.

## Relation to official MMAudio

Official `batch_eval.py` launches one complete model replica per GPU with
`torchrun`, uses a `DistributedSampler`, runs batched bf16 inference, and
writes audio without composing videos. Metric calculation is a separate step
through `av_bench.extract` and `av_bench.evaluate`.

This FastVideo path keeps that high-level separation:

1. one complete MMAudio pipeline replica per GPU;
2. data-parallel dataset inference to WAV;
3. a canonical JSONL manifest that binds id, source video, generated audio,
   caption, seed, and reference-audio source;
4. independent multi-GPU metric workers over the complete corpus.

There are two intentional differences. Rank-strided indices avoid the padded
duplicates that PyTorch `DistributedSampler(drop_last=False)` can introduce,
and cached CLIP/Synchformer/text features skip video decoding and conditioning
encoder work already completed during FastVideo preprocessing. The current
cached-feature stage accepts one sample per pipeline call; four GPUs still
process four samples concurrently.

## Environment

Use the same FastVideo environment as training:

```bash
cd /mnt/lustre/vlm-kai/FastVideo
source .venv/bin/activate
uv pip install --python .venv/bin/python hear21passt pyloudnorm
```

The four default native metrics are:

| FastVideo metric | Protocol |
| --- | --- |
| `audio.frechet_distance` | corpus PaSST FAD / FD_PaSST |
| `audio.kl_divergence` | paired PaSST KL, `KL(gt || pred)` |
| `audio.clap_score` | text-audio CLAP similarity |
| `audio.desync` | source-video/generated-audio Synchformer DeSync |

FAD and KL port the av-benchmark mathematics. DeSync vendors the same
Synchformer family. FastVideo CLAP uses the Hugging Face LAION CLAP checkpoint
and is not byte-identical to the external package's CLAP implementation; use
the optional official backend below when exact paper-table reproduction is
required.

## 1. Export the final EMA

The training result is a transformer state dict. Inference also needs the
frozen audio VAE, vocoder, scheduler, and (for raw-video inference) the frozen
conditioning encoders. Export a complete FastVideo model once:

```bash
bash examples/inference/eval/mmaudio/export_final_ema.sh
```

Default input:

```text
outputs/mmaudio_small_44k_ddp_from_scratch/posthoc_ema/official_ddp/
  mmaudio_ema_final_sigma_0p05_step_000300000.pth
```

Default output:

```text
converted_weights/mmaudio/small_44k_ema_300000/
```

Override `EMA_CHECKPOINT`, `OUTPUT_MODEL`, or `ASSET_ROOT` before the command
for another run or variant.

## 2. Multi-GPU inference

The launcher defaults to the VGGSound test feature cache and four GPUs:

```bash
bash examples/inference/eval/mmaudio/run_inference_vggsound.sh
```

For a four-sample smoke test:

```bash
MAX_SAMPLES=4 \
OUTPUT_DIR=outputs/mmaudio_small_44k_ema_300000_vggsound_smoke \
bash examples/inference/eval/mmaudio/run_inference_vggsound.sh
```

For a background full run with a persistent log:

```bash
mkdir -p logs
nohup bash examples/inference/eval/mmaudio/run_inference_vggsound.sh \
  > logs/mmaudio_vggsound_inference.log 2>&1 &
tail -f logs/mmaudio_vggsound_inference.log
```

The output directory contains:

```text
audio/<sample-id>.wav
eval_manifest.jsonl
failures.jsonl
summary.json
manifest_rank_*.jsonl
failures_rank_*.jsonl
```

Existing WAV files are resumed rather than regenerated. Add `--overwrite` to
the Python runner when intentional regeneration is required. Seeds are
`base_seed + global_index`, so results do not change with GPU count.

## 3. Native FastVideo evaluation

Run FAD, KL, CLAP, and DeSync with four metric replicas:

```bash
bash examples/inference/eval/mmaudio/run_evaluate_vggsound.sh
```

The first run extracts ground-truth audio from each source MP4 into
`<OUTPUT_DIR>/reference_audio` and caches metric weights. Later runs reuse
both. Results are written to `fastvideo_eval_results.json`; per-sample metrics
are under `samples`, while FAD is under `corpus`.

The underlying generic command is:

```bash
fastvideo eval run \
  --manifest outputs/mmaudio_small_44k_ema_300000_vggsound_test/eval_manifest.jsonl \
  --metrics audio.frechet_distance,audio.kl_divergence,audio.clap_score,audio.desync \
  --num-gpus 4 \
  --extract-audio outputs/mmaudio_small_44k_ema_300000_vggsound_test/reference_audio \
  --extract-workers 16 \
  --output-format full \
  --output outputs/mmaudio_small_44k_ema_300000_vggsound_test/fastvideo_eval_results.json
```

Do not evaluate FAD by invoking the evaluator once per file. FAD is a
set-vs-set statistic; this CLI submits the complete manifest in one call and
serializes `EvalResults.corpus`.

## Exact external av-benchmark backend

For strict comparison to official MMAudio numbers, FastVideo provides the
dedicated `fastvideo eval v2a` command. It launches the exact
`av_bench.extract` and `av_bench.evaluate` functions with a Python interpreter
from an isolated environment. This keeps av-benchmark's CLAP, ImageBind, and
PyTorch dependencies out of the main FastVideo environment.

Create the isolated environment once:

```bash
cd /mnt/lustre/vlm-kai
git clone https://github.com/hkchengrex/av-benchmark.git
uv venv av-benchmark/.venv --python 3.12
uv pip install --python av-benchmark/.venv/bin/python -e av-benchmark
# PyTorch 2.9+ routes torchaudio decoding through TorchCodec. Transformers 5
# removed an API used by the vendored Synchformer AST, while SciPy 1.17 removed
# an argument used by the official FAD implementation. Keep these constraints
# in this isolated environment; select the wheel matching its PyTorch CUDA build.
UV_TORCH_BACKEND=cu130 uv pip install \
  --python av-benchmark/.venv/bin/python \
  torchcodec==0.15.0 'transformers<5' 'scipy<1.17'

mkdir -p av-benchmark/weights
curl -L https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt \
  -o av-benchmark/weights/music_speech_audioset_epoch_15_esc_89.98.pt
curl -L https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth \
  -o av-benchmark/weights/synchformer_state_dict.pth
```

Download only the official VGGSound ground-truth cache (about 16 GB):

```bash
cd /mnt/lustre/vlm-kai/FastVideo
.venv/bin/hf download hkchengrex/MMAudio-precomputed-results \
  --repo-type dataset \
  --include "vggsound-test-eval-cache/*" \
  --local-dir /mnt/lustre/vlm-kai/datasets/VGGSound/av_benchmark
```

Then run the dedicated launcher:

```bash
bash examples/inference/eval/mmaudio/run_evaluate_vggsound_av_benchmark.sh
```

For a background run:

```bash
nohup bash examples/inference/eval/mmaudio/run_evaluate_vggsound_av_benchmark.sh \
  > logs/mmaudio_vggsound_av_benchmark.log 2>&1 &
tail -f logs/mmaudio_vggsound_av_benchmark.log
```

The official VGGSound cache produces the complete paper-style metric set:

```text
FD-VGG, FD-PANN, FD-PASST
KL-PANNS-softmax, KL-PASST-softmax
ISC-PANNS-mean/std, ISC-PASST-mean/std
IB-Score, DeSync
```

The official VGGSound cache does not contain CLAP text features, so the launcher
defaults to `SKIP_CLAP=1`; this saves prediction feature passes without changing
the metric set. Set `SKIP_CLAP=0` for a custom GT cache that includes CLAP text
features. Prediction features are cached and reused; set `RECOMPUTE=1` to force
a clean extraction.

The launcher also defaults to `ALIGN_PREDICTION_KEYS=1`. The official
VGGSound cache prepends underscores to some sanitized ids; FastVideo keeps the
original ids. The backend changes only unique matches in prediction feature
caches, leaves WAV filenames untouched, and records exact/remapped/ambiguous/
unmatched counts under `prediction_key_alignment` in the result JSON.

The launcher defaults to `NUM_WORKERS=0`: this avoids both the machine's 64 MB
`/dev/shm` limit and forking workers after the official extractor initializes
CUDA. Raise it only after validating the container and backend combination.

## Evaluation protocol warning

The current FastVideo test cache contains the filtered natural-language
descriptions used in this training run. Official MMAudio's VGGSound
`batch_eval.py` reads class-label captions from the original VGGSound CSV.
FAD, KL, and DeSync formulas do not read captions directly, but changing the
generation caption changes the generated waveform and can therefore change all
scores. CLAP additionally consumes the caption during evaluation. Record the
generation manifest, test split, and ground-truth cache with reported results.
