# `fastvideo/attention/` — Attention Backends

**Generated:** 2026-05-02

Backend registry + selector wrapping FlashAttn / SageAttn / SageAttn3 / SDPA / VSA / VMoBA / SLA / BSA.

## Layout

```
attention/
├── __init__.py            # Exports DistributedAttention, LocalAttention, get_attn_backend
├── layer.py               # DistributedAttention, DistributedAttention_VSA, LocalAttention
├── selector.py            # get_attn_backend (cached) + _component_attention_backend_scope
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

## Where the Decision Is Made

**Resolved once, at load time, and carried on the component.** The environment
variable is folded into `FastVideoArgs.attention_backend` in `__post_init__`
(the parse-once adapter); `PipelineComponentLoader.load_module` applies that
request per component; each loader records what its component resolved onto
that component's own config as `ModelConfig._resolved_attention_backend`. After
load, the decision is readable from the component itself:

```python
transformer.config._resolved_attention_backend
```

A loader may narrow the request for one component — the DMD teacher/critic
transformers build dense — and the recorded value is what that component
actually resolved, not what the run asked for globally.

A call site that already knows its component passes the decision explicitly:

```python
get_attn_backend(..., requested=component_attention_backend(self.transformer))
```

`requested` distinguishes three cases, and the distinction matters — `None` is
an answer, not an absence:

| value | meaning |
|---|---|
| a backend | use it; ignore the scope and the environment |
| `None` | this component resolved to *automatic selection*; do **not** consult the environment behind it |
| omitted (`NO_REQUEST`) | the caller has no opinion; fall back to the scope, then the environment |

Nothing switches backends at runtime and nothing should: every request is an
input to *construction*.

## Selection Order

`get_attn_backend()` reads every selection input, then resolves via, in
precedence order:

1. An explicit `requested=` argument — the component's own recorded decision.
2. The per-component request carried by the construction scope.
3. Env-var `FASTVIDEO_ATTENTION_BACKEND` (see `STR_BACKEND_ENV_VAR` in
   `fastvideo/utils.py`), consulted only when the caller supplied no request
   and no scope is active — layers built outside a loader, and direct model
   construction in tests.
4. The layer-declared `default_backend`.
5. Per-platform automatic selection from `fastvideo/platforms/`, which probes
   the *current device's* capability.

The result is cached on **all** of those inputs (plus component identity and
device index), so a changed request simply lands on a different cache key —
no `cache_clear()` is needed and none should be added.

A layer that does not declare support for the requested backend falls back, so
the component-level decision is the request every layer of that component
resolves against, not a promise about the kernel any one layer runs.

Never mutate `FASTVIDEO_ATTENTION_BACKEND` mid-process.

### Transitional: how the request reaches layers

The request travels from loader to layers via
`_component_attention_backend_scope`, a private ContextVar. It is plumbing with
an end date, not API: layer constructors nested inside a model receive only a
bare `supported_attention_backends` tuple, threaded through block constructors
up to four levels deep, so they cannot yet read the decision off their own
config. Thread the request alongside that tuple, one model family at a time —
Wan, LTX-2 and Kandinsky5 first, since those carry per-role requests — and the
scope goes away when the last family lands. Do not build on it.

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
