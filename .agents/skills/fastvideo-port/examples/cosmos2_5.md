# Worked Example: Cosmos 2.5

**Official repo:** NVIDIA Cosmos 2.5 (internal / HF gated)  
**Architecture:** MiniTrainDIT — a standard cross-attention DiT (28 layers,
hidden 2048, 16 heads, head_dim 128). Separate self-attention and cross-
attention blocks with AdaLN-LoRA modulation.

## Key architecture details

- 28 transformer blocks; each block has self-attn + cross-attn + MLP.
- AdaLN-LoRA: low-rank (256-dim) modulation per block instead of full AdaLN.
- Text encoder: `Reason1` (Qwen 7B-based VLM); embeddings are layer-norm-
  concatenated across all hidden layers, producing 100,352-dim vectors.
- VAE: `Cosmos25VAE` (16-channel latent, patch_size `[1,2,2]`).
- Scheduler: flow matching with `shift=5.0`.
- RoPE positional encoding with per-axis scaling `(T=1.0, H=3.0, W=3.0)`.
- `concat_padding_mask=True` doubles in_channels for the patch embedding.

## Non-standard aspects requiring attention

- **Official checkpoint has `net.*` prefix on all keys** — every rule in
  `param_names_mapping` must strip the leading `net.` prefix.
- **`pos_embedder.*` keys in checkpoint** — these are dynamically recomputed
  in FastVideo's `Cosmos25RotaryPosEmbed`; safe to ignore (not mapped).
- **`accum_*` training metadata keys** — skip during loading (not mapped).
- **`_extra_state` on RMSNorm layers** — include pass-through rules or
  `load_state_dict` will report unexpected keys. These are PyTorch internal
  state and will be recomputed automatically.
- **`use_crossattn_projection=True` in official checkpoint** — requires
  100,352-dim embeddings. FastVideo defaults to `False`; set it to `True`
  when loading the official weights or key shapes won't match.
- **Reason1 postprocessing** — hidden states from all layers must be
  layer-norm'd and concatenated before passing to the DiT. This is a custom
  `postprocess_text_funcs` callable, not a standard encoder output.

## param_names_mapping highlights

Key structural differences between official and FastVideo:

| Official | FastVideo |
|----------|-----------|
| `net.x_embedder.proj.1.*` | `patch_embed.proj.*` |
| `net.t_embedder.1.linear_1.*` | `time_embed.t_embedder.linear_1.*` |
| `net.t_embedding_norm.*` | `time_embed.norm.*` |
| `net.blocks.N.self_attn.q_proj.*` | `transformer_blocks.N.attn1.to_q.*` |
| `net.blocks.N.cross_attn.q_proj.*` | `transformer_blocks.N.attn2.to_q.*` |
| `net.blocks.N.mlp.layer1.*` | `transformer_blocks.N.mlp.fc_in.*` |
| `net.blocks.N.mlp.layer2.*` | `transformer_blocks.N.mlp.fc_out.*` |
| `net.blocks.N.adaln_modulation_*.1.*` | `transformer_blocks.N.adaln_modulation_*.1.*` |
| `net.blocks.N.layer_norm_*._extra_state` | `transformer_blocks.N.norm*.norm._extra_state` |
| `net.final_layer.linear.*` | `final_layer.proj_out.*` |
| `net.final_layer.adaln_modulation.1.*` | `final_layer.linear_1.*` |

## Lessons learned (filed under `.agents/lessons/`)

| Issue | Lesson file |
|-------|-------------|
| AutoTokenizer fails on Reason1 | `2026-04-09_autotokenizer-fails-on-multimodal-processor.md` |
| bf16 embeddings crash numpy | `2026-04-09_bf16-embeddings-need-float-before-numpy.md` |
| fp32/bf16 dtype mismatch at projection | `2026-04-09_fp32-bf16-dtype-mismatch-at-projection.md` |
| Hardcoded (512, 4096) in utils.py | `2026-04-09_hardcoded-text-encoder-shape-in-utils.md` |
| ReplicatedLinear required for LoRA | `2026-04-09_lora-requires-replicated-linear.md` |
| VLM processor needs inner tokenizer | `2026-04-09_multimodal-processor-needs-inner-tokenizer.md` |
| Preprocessing entrypoint not ported | `2026-04-09_preprocessing-entrypoint-not-always-ported.md` |
| Scheduler shift not wired to training | `2026-04-09_scheduler-shift-not-wired-to-training.md` |

## Credentials required

- `HF_TOKEN` with access to the Cosmos 2.5 gated HF repo.
- `HF_TOKEN` with access to `Qwen/Qwen2.5-7B-Instruct` (Reason1 base model, gated).