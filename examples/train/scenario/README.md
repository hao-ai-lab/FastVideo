# Training Scenarios

End-to-end multi-step training pipelines. Each scenario directory contains
all configs, scripts, and data needed to run a complete workflow.

```
scenario/
├── ode_init_self_forcing_wan_causal/   # KD → export → Self-Forcing
└── qad_wan2_1_mixkit/                  # Attn-QAT SFT → export → DMD2
```

See the `usage.md` inside each scenario for step-by-step instructions.

For single-step configs, see `examples/train/configs/`.
