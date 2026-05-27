Diagnosed the actionable crash from the complete adjacent log: PEFT was trying to wrap FastVideo’s `ReplicatedLinear`, which PEFT does not support.

Changed:
- `GenRLWanModel` uses FastVideo’s native LoRA wrapper for FastVideo linear layers instead of PEFT.
- Updated `examples/train/configs/genrl_wan2.1_t2v_1.3B_longcat.yaml` LoRA targets from Diffusers names to FastVideo names:
  - `to_out.0` -> `to_out`
  - `net.0.proj` -> `ffn.fc_in`
  - `net.2` -> `ffn.fc_out`

Validation:
- `python -m py_compile modal_train_genrl.py` passed.
- `python -m compileall fastvideo modal_train_genrl.py` passed.
- Static LoRA config/source checks passed.
- Direct import check could not run locally because this environment lacks `torch`; I did not run the full Modal training command.