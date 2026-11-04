---
date: 2026-11-02
experiment: FastVideo CI repair on main (Transformer and Unit lanes)
category: infrastructure
severity: important
---

# pytest 9 monkeypatch cannot restore torch flag proxies

## What Happened

After the CI image picked up pytest 9, the Transformer lane reported 22
teardown errors (`'cuBLASModule' object has no attribute 'allow_tf32'`) and
the Unit lane reported 28 setup errors in `test_lora_side_branch.py`
(`torch._dynamo.config ... has no attribute 'disable'`). The test bodies
still passed.

## Root Cause

pytest 9's `monkeypatch.setattr` reads the previous value from the target's
instance `__dict__`. `torch.backends.cuda.matmul` and `torch._dynamo.config`
serve their flags through `__getattr__`, so pytest records the value as
missing. Teardown then calls `delattr`: on cuBLAS that raises and leaves TF32
changed, and on the dynamo config it deletes the `disable` entry, so every
later `setattr` on it fails. pytest 8.4 restores both correctly.

## Fix / Workaround

Use the owner's own scoping instead of monkeypatch:
`torch._dynamo.config.patch(name=value)` for dynamo/inductor config, and the
`tf32_disabled` fixture in `fastvideo/tests/conftest.py` for the TF32 flags.

## Prevention

Do not `monkeypatch.setattr` attributes of `torch.backends.cuda.matmul`,
`torch._dynamo.config`, `torch._inductor.config` or other torch config
proxies. Module functions such as `torch.compiler.is_compiling` are safe.
