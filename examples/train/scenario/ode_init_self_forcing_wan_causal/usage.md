# ODE-Init Self-Forcing: Wan 2.1 Causal 1.3B

End-to-end scenario that trains a causal Wan 1.3B model via
ODE-init knowledge distillation followed by self-forcing.

## Steps

```bash
# Step 1 — KD training (produces ODE-init checkpoint)
bash examples/train/run.sh \
    examples/train/scenario/ode_init_self_forcing_wan_causal/step1_kd.yaml

# Step 2 — Export ODE-init weights to diffusers format
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role student \
    --checkpoint outputs/kdtest/kd_causal/checkpoint-300 \
    --output-dir outputs/kdtest/kd_ode_init

# Step 3 — Self-forcing with the exported ODE-init
bash examples/train/run.sh \
    examples/train/scenario/ode_init_self_forcing_wan_causal/step3_self_forcing.yaml
```
