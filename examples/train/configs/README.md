# Training Configs

Single-step training configurations organized by method and model.

```
configs/
├── fine_tuning/          # Standard finetuning and DFSFT
├── distribution_matching/  # DMD2 and Self-Forcing
├── knowledge_distillation/ # KD from teacher to student
└── example.yaml          # Annotated reference config with all fields
```

Each method directory contains per-model subdirectories (e.g. `wan/`, `hunyuan/`).

Launch any config with:

```bash
bash examples/train/run.sh examples/train/configs/<method>/<model>/<config>.yaml
```

For multi-step training pipelines, see `examples/train/scenario/`.
