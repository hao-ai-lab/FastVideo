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
    --gradient-accumulation-steps 60 \
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
