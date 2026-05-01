# Tests

- `tests/local_tests/` holds local-only parity, smoke, and component tests
  organized **by model family** (e.g. `tests/local_tests/ltx2/`,
  `tests/local_tests/stable_audio/`). They typically require a checked-out
  reference repo and/or model weights and are skipped in CI; see each family's
  `README.md` and `PORT_STATUS.md` for setup. The top-level
  [`tests/local_tests/README.md`](./local_tests/README.md) is the family index.
- The CI-backed test suite still lives in `fastvideo/tests/`.
- Eventually, all tests will move under `tests/`.
