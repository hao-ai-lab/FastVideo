# Examples

- `generate_t2v.py` — text-to-video via the SDK (`fv2.load(...)` →
  `model.generate(...)` → `result.save(...)`). Card defaults, everything
  overridable by flag. The standalone reference implementation (no SDK, no
  runtime) lives at `fastvideo2/wan21/reference.py`.

The `--model` flag takes any catalog id — e.g. the 3-step FastWan students
(`fastwan-qad-fp8-1.3b`, `fastwan-t2v-1.3b`); their step count, sampler, and
sparsity/quant recipe come from the card, so no other flags change.
