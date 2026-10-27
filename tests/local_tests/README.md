# Local Tests

Local-only parity, smoke, component, and developer-facing behavior tests for
FastVideo. They cover model ports and supporting tooling and are **skipped in
CI**; run them locally on a single GPU (or CPU where noted).

For the CI-backed test suite, see [`fastvideo/tests/`](../../fastvideo/tests/).

## Layout

Most tests are organized by **model family**, one directory per port. Each family
directory follows this layout:

```
tests/local_tests/<family>/
├── README.md         # reference assets, setup, test inventory
├── __init__.py
└── test_<family>_<component>_*.py
```

## Families

| Family | Workload | Dir |
|---|---|---|
| Flux2 | T2I | [`flux2/`](./flux2/) |
| GEN3C | T2V | [`gen3c/`](./gen3c/) |
| Kandinsky-5 | T2V | [`kandinsky5/`](./kandinsky5/) |
| Kandinsky-6 | TI2VA (T2V/I2V + audio) | [`kandinsky6/`](./kandinsky6/) |
| Kandinsky-6 SR | V2V (upscale) | [`kandinsky6_sr/README.md`](./kandinsky6_sr/README.md) |
| LTX-2 | T2V (+ audio) | [`ltx2/`](./ltx2/) |
| Stable Diffusion 3.5 | T2I | [`sd35/`](./sd35/) |
| Stable Audio Open 1.0 | T2A | [`stable_audio/`](./stable_audio/) |

Wan2.2 I2V record-schema checks live in the
[package dataset tests](../../fastvideo/tests/dataset/test_schema_record_creator.py).

## Other Suites

| Suite | Purpose | Dir |
|---|---|---|
| Performance CI behavior | Config, comparison policy, identity, result schema, dashboard, and worker-log capture checks | [`performance/`](./performance/) |

## Running a family

```bash
pytest tests/local_tests/<family>/ -v -s
```

See each family's `README.md` for setup (clones, weights, env vars, gated
tokens) and the per-component parity coverage.

## Adding a new family

Follow the [model porting guide](../../docs/contributing/coding_agents.md) to
stage the reference clone and weights, then add the family directory and its
`README.md` in the layout above. (In-progress ports may also keep a
`PORT_STATUS.md`; remove it once the port lands.)
