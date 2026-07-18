# QAD Wan2.1 MixKit Attn-QAT

This scenario runs entirely on the modular `fastvideo/train` stack. The
student's attention backend is configured per role, so DMD2 can keep the
teacher and critic on Flash Attention while the student uses the fake-quantized
Attn-QAT kernel.

From the repository root:

```bash
# 1. Download the preprocessed MixKit data.
bash examples/training/finetune/wan_t2v_1.3B/mixkit/download_mixkit_data.sh

# 2. Stage 1: Attn-QAT supervised finetune.
NUM_GPUS=4 bash examples/train/scenario/qad_wan2_1_mixkit/run_stage1.sh

# 3. Export the stage-1 DCP checkpoint to a Diffusers weight file.
bash examples/train/scenario/qad_wan2_1_mixkit/export_stage1.sh

# 4. Stage 2: three-step Attn-QAT DMD2 distillation.
NUM_GPUS=4 bash examples/train/scenario/qad_wan2_1_mixkit/run_stage2.sh
```

The two YAML configs are also directly runnable through `examples/train/run.sh`.
The wrapper scripts only provide dataset/checkpoint paths and derive distributed
dimensions from `NUM_GPUS`.
