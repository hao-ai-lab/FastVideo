# DiffusionNFT Wan Video RL

This guide reproduces the DiffusionNFT Wan video RL run without any private
launcher. Modal scripts used by individual developers should only call these
tracked commands.

## 1. Prepare Prompts, Rewards, Parquet, and Run Config

Install the reward-stack pins after installing FastVideo:

```bash
uv pip install -e .
uv pip install -r examples/train/requirements-diffusion-nft.txt
```

```bash
python examples/train/prepare_diffusion_nft_assets.py \
    --config examples/train/configs/rl/wan/diffusion_nft_videoalign.yaml \
    --data-root data/diffusion_nft \
    --cache-root .cache/diffusion_nft \
    --output-dir outputs/wan2.1_diffusion_nft_videoalign \
    --dataset world-r1-enhanced \
    --reward videoalign \
    --max-prompts quarter \
    --num-frames 77 \
    --num-gpus 4 \
    --hsdp-replicate-dim 1 \
    --hsdp-shard-dim 4 \
    --max-train-steps 100 \
    --gradient-accumulation-steps 24 \
    --sample-num-steps 50 \
    --sample-flow-shift 8 \
    --sample-guidance-scale 6 \
    --preprocess-batch-size 128 \
    --check-rewards \
    --json
```

The script:

- resolves `num_latent_t` from Wan's frame rule,
- downloads World-R1 prompts or reads a DiffusionNFT `dataset/<name>/train.txt`,
- clones `DiffusionNFT` only when a selected dataset/reward needs Flow-GRPO,
- downloads the `KwaiVGI/VideoReward` snapshot when VideoAlign rewards are selected,
- preprocesses prompts into FastVideo text-only parquet,
- writes `outputs/diffusion_nft_run_configs/diffusion_nft_wan_run.yaml`.
- optionally loads the selected reward suite on a dummy video with
  `--check-rewards`.

Set `VIDEOALIGN_CHECKPOINT_PATH` or `DIFFUSION_NFT_ROOT` through the matching
CLI flags if those assets already live somewhere else.

## 2. Launch Training

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    outputs/diffusion_nft_run_configs/diffusion_nft_wan_run.yaml
```

W&B logging records the resolved config and a code snapshot by default. Set
`FASTVIDEO_WANDB_LOG_CODE=0` only when intentionally disabling code capture.

The video config intentionally uses the Wan/MAY-30 rollout sampler profile:
`flow_unipc`, 50 denoising steps, flow shift 8, and CFG 6. Validation uses the
same 50-step sampler so W&B qualitative videos are comparable to rollout
quality.

Validation reward metrics are logged under `validation/reward/*` every
`method.validation.every_steps`. Qualitative W&B videos are only a sanity check:
`--log-sample-max-videos 0` disables them, while a positive value caps how many
validation samples are logged per eval.

For short 17-frame H100 runs, keep gradient accumulation matched to the number
of rollout train batches. For example, if you use `--num-batches-per-epoch 4`,
`--collection-batch-size 4`, `--train-batch-size 4`, and 50 sample steps, use
`--gradient-accumulation-steps 4` for one full optimizer update per outer
DiffusionNFT step. A much larger accumulation value, such as 30, still runs, but
only performs a partial final update for that outer step and W&B will report it
under `nft/partial_optimizer_step_ratio`.

DiffusionNFT relies on repeated samples for the same prompt to estimate
per-prompt advantages. `--num-samples-per-prompt 4` is a cheap smoke setting,
but it is high variance. If the reward curves look noisy and memory allows it,
try `--num-samples-per-prompt 8` or increase `--num-batches-per-epoch` before
judging the training run.

For Wan video reward rollouts, keep `method.sampling.guidance_scale` enabled
unless you are intentionally testing unguided generation. The May 30 video run
used CFG sampling at `6.0`; unguided rollouts can look foggy and static before
the reward model ever sees them.

For offline logging:

```bash
WANDB_MODE=offline NUM_GPUS=4 bash examples/train/run.sh \
    outputs/diffusion_nft_run_configs/diffusion_nft_wan_run.yaml
```

## Reward Presets

- `--reward videoalign`: `videoalign_vq`, `videoalign_mq`, `videoalign_ta`
- `--reward videoalign_hpsv3`: VideoAlign plus `hpsv3_general`
- `--reward multi_reward`: DiffusionNFT image reward preset
- `--reward <name>`: one explicit reward name

Use `examples/train/configs/rl/wan/diffusion_nft_videoalign.yaml` as the
checked-in base config and treat the generated run config as the exact
experiment instance.
