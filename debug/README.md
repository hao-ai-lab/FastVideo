## Debug / Repro scripts

### Force VSA to use Triton on H100

Set:

```bash
export FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1
```

This forces `fastvideo_kernel.block_sparse_attn.block_sparse_attn(...)` to take the
**Triton** path even when SM90 (H100) compiled ops are available.

### Route B: minimal padding/backward repro

Run:

```bash
python debug/repro_vsa_padding_nan.py --attend-last-only --num-blocks 8 --partial-block-size 48 --scales 1 10 30 100
```

What it does:
- Builds `q/k/v` with `seq_len = 64 * num_blocks`
- Sets `variable_block_sizes[-1] < 64` to emulate a padded (partial) last block
- Zeros out invalid tokens in the last block
- Runs forward + backward and checks for NaNs in `dq/dk/dv`

