# MiniMax H3 porting pitfalls

- **Scheduler:** video/audio shifts must stay `12/3`; never apply the global `flow_shift`.
- **Steps:** the grid includes terminal zero; `N` grid points mean `N-1` denoiser calls.
- **Packing:** row order, channel-major layout, float64 positions, and RNG order are model semantics.
- **Parity:** reference and FastVideo packers must remain independent.
- **RoPE:** rotate only the first `96/128` head channels; the generic full-head path is wrong.
- **Sequence parallel:** padding is transport-only and must never become a semantic row.
- **Dtype:** the DiT has FP32 islands in a BF16 body; both VAEs stay FP32.
- **Meta loading:** rebuild non-persistent RoPE buffers after meta-device initialization.
- **Audio VAE:** H3 needs encode and decode; a decoder-only loader breaks Ref2VA.
- **Partitions:** FL2VA and Ref2VA Transformers are alternatives, not two simultaneous denoisers.
- **Evidence:** synthetic parity does not establish real-weight or CUDA compatibility.
