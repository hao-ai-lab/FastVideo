# v2_examples/omni — MoT & cascade models

Runnable examples of the omni models: one request that spans **multiple loop types**. Two topologies
(designv4 §2.3):

  * **MoT / shared-weight** (Cosmos3, BAGEL) — ONE resident instance whose `transformer` runs *both* an
    `ar_decode` loop and a `diffusion_denoise` loop on the same weights. Every AR token and denoise step
    is a runtime-visible WorkUnit (the differentiation vs vllm-omni's opaque DIFFUSION stage).
  * **Cascade / separate experts** (Qwen-Omni) — three *distinct* experts on three loop types
    (`thinker → talker → vocoder`), cross-stage conditioning, streaming codec→waveform.

Run from anywhere:

```bash
python3 v2_examples/omni/01_cosmos3.py
```

| Script | Model | Loops in one request |
|---|---|---|
| `01_cosmos3.py` | Cosmos3 (shared MoT) | `ar_decode` (reason) → `diffusion_denoise` (joint video) |
| `02_bagel.py` | BAGEL (shared MoT) | `ar_decode` (generate_text) → `diffusion_denoise` (generate_image) |
| `03_qwen_omni.py` | Qwen-Omni (3 separate experts) | `ar_decode` → `ar_decode` → `audio_decode` (text + speech) |
| `04_interleave_across_loop_types.py` | — | the interleave parity gate holds across AR + diffusion steps |

`build_omni_engine()` registers all three; see designv4 §9.5 (Qwen-Omni cascade).
