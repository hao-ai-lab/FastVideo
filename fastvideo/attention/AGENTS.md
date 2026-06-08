# `fastvideo/attention/` — Attention Backends

**Generated:** 2026-05-02

Backend registry + selector wrapping FlashAttn / SageAttn / SageAttn3 / SDPA / VSA / VMoBA / SLA / BSA.

## Layout

```
attention/
├── __init__.py            # Exports DistributedAttention, LocalAttention, get_attn_backend
├── layer.py               # DistributedAttention, DistributedAttention_VSA, LocalAttention
├── selector.py            # get_attn_backend (cached) + env-var override
├── backends/
│   ├── abstract.py        #   AttentionBackend / AttentionMetadata / AttentionMetadataBuilder
│   ├── flash_attn.py      #   FA2/FA3
│   ├── sage_attn.py       #   SageAttention v1
│   ├── sage_attn3.py      #   SageAttention v3
│   ├── sdpa.py            #   torch SDPA fallback
│   ├── video_sparse_attn.py  # VSA (paper: Video Sparse Attention)
│   ├── vmoba.py           #   Video-MoBA
│   ├── sla.py             #   Sliding-window (STA)
│   └── bsa_attn.py        #   Block-sparse
└── utils/
    ├── flash_attn_cute.py
    └── flash_attn_no_pad.py
```

## Selection Order

`get_attn_backend()` resolves via:

1. Env-var override `FASTVIDEO_ATTENTION_BACKEND` (see `STR_BACKEND_ENV_VAR` in `fastvideo/utils.py`).
2. Per-platform default from `fastvideo/platforms/`.
3. Heuristic fallback to SDPA.

The result is `@lru_cache`d. Tests that need a specific backend must use the
`global_force_attn_backend(...)` context manager from `selector.py`, never set
the env var mid-process.

## Adding a Backend

1. Subclass `AttentionBackend` in `backends/<name>.py`.
2. Implement `AttentionMetadata` + `AttentionMetadataBuilder` for the new path.
3. Register the enum value in `fastvideo/platforms/interface.py` (`AttentionBackendEnum`).
4. Wire string → class resolution in `selector.py`.
5. Verify the new backend works with `DistributedAttention` (sequence parallel)
   and `LocalAttention` (single-rank). If it cannot support SP, document the
   gap in the backend file's module docstring.

## Anti-Patterns

- Calling `torch.nn.functional.scaled_dot_product_attention` directly inside a
  model's forward — go through `DistributedAttention` / `LocalAttention`.
- Reading `os.environ[STR_BACKEND_ENV_VAR]` from arbitrary call sites. Use
  `get_env_variable_attn_backend()`.
- Caching backend instances per-module. The selector cache is process-wide; do
  not duplicate it.
