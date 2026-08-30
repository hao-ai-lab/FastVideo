# Compatibility Matrix

This page summarizes common model + optimization combinations.

For the canonical, code-level list of model IDs recognized by
`VideoGenerator.from_pretrained(...)`, see the registrations in
`fastvideo/registry.py` (`register_configs(...)` entries).

!!! note
    The full STA integration in `fastvideo/` is archived from `main` and kept
    in `sta_do_not_delete`:
    https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete
    We do this because we believe VSA is strictly better than STA for the
    actively maintained `main` inference path.

## Registered Model IDs

Every Hugging Face model ID registered in `fastvideo/registry.py` on `main`
(commit `8d89f30d`), grouped by family. Any ID below can
be passed to `VideoGenerator.from_pretrained(...)`; FastVideo resolves the
matching pipeline and sampling defaults. The **Family** column is a
documentation grouping: it follows each registration's declared `model_family`,
except `black-forest-labs/FLUX.1-dev`, which declares none and is listed under
`flux` for readability. The **Workloads** column shows each
registration's declared `workload_types`; `—` means the entry is registered
without a UI workload option but is still loadable by ID. The **Example**
column links a runnable script in `examples/inference/basic/` where one exists.

| Family | HuggingFace Model ID | Workloads | Example |
|--------|----------------------|-----------|---------|
| cosmos | `nvidia/Cosmos-Predict2-2B-Video2World` | T2V | — |
| cosmos25 | `KyleShao/Cosmos-Predict2.5-2B-Diffusers` | T2V | [basic_cosmos2_5_t2w.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_cosmos2_5_t2w.py) |
| cosmos25 | `nvidia/Cosmos-Predict2.5-14B` | T2V | [basic_cosmos2_5_t2w.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_cosmos2_5_t2w.py) |
| dreamx_world | `FastVideo/DreamX-World-5B-Cam-Diffusers` | I2V | [basic_dreamx_world.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_dreamx_world.py) |
| dreamx_world | `FastVideo/DreamX-World-5B-Diffusers` | I2V | [basic_dreamx_world.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_dreamx_world.py) |
| flux | `black-forest-labs/FLUX.1-dev` | T2I | [basic_flux_dev.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_flux_dev.py) |
| flux2 | `black-forest-labs/FLUX.2-klein-4B`<br>`black-forest-labs/FLUX.2-klein-9B` | T2I | [basic_flux2_klein.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_flux2_klein.py) |
| flux2 | `black-forest-labs/FLUX.2-dev` | T2I | [basic_flux2.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_flux2.py) |
| gamecraft | `FastVideo/HunyuanGameCraft-Diffusers` | I2V | [basic_gamecraft.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_gamecraft.py) |
| gen3c | `FastVideo/GEN3C-Cosmos-7B-Diffusers` | T2V | [basic_gen3c.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_gen3c.py) |
| glm_image | `zai-org/GLM-Image` | T2I | [basic_glm_image.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_glm_image.py) |
| hunyuan | `hunyuanvideo-community/HunyuanVideo` | T2V | — |
| hunyuan | `FastVideo/FastHunyuan-diffusers` | T2V | — |
| hunyuan15 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | T2V | [basic_hy15.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_hy15.py) |
| hunyuan15 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled` | I2V | [basic_hy15.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_hy15.py) |
| hunyuan15 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | T2V | [basic_hy15.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_hy15.py) |
| hunyuan15 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled` | I2V | [basic_hy15.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_hy15.py) |
| hunyuan15 | `weizhou03/HunyuanVideo-1.5-Diffusers-1080p`<br>`weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR` | — | [basic_hy15_1080p.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_hy15_1080p.py) |
| hyworld | `FastVideo/HY-WorldPlay-Bidirectional-Diffusers` | — | [basic_hyworld.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_hyworld.py) |
| kandinsky5 | `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` | T2V | [basic_kandinsky5_t2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky5_t2v.py) |
| kandinsky5 | `kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers` | T2V | [basic_kandinsky5_t2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky5_t2v.py) |
| kandinsky5 | `kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers` | T2V | [basic_kandinsky5_t2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky5_t2v.py) |
| kandinsky5 | `kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers` | T2V | [basic_kandinsky5_t2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky5_t2v.py) |
| kandinsky5 | `kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers` | I2V | [basic_kandinsky5_i2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky5_i2v.py) |
| kandinsky5 | `kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers` | I2V | [basic_kandinsky5_i2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky5_i2v.py) |
| kandinsky5 | `kandinskylab/Kandinsky-5.0-I2V-Pro-distilled-5s-Diffusers` | I2V | [basic_kandinsky5_i2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky5_i2v.py) |
| lingbot_video | `FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers` | T2V | [basic_lingbot_video.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_lingbot_video.py) |
| lingbot_video | `FastVideo/LingBot-Video-Dense-1.3B-Diffusers` | T2V | [basic_lingbot_video.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_lingbot_video.py) |
| lingbotworld | `FastVideo/LingBot-World-Base-Cam-Diffusers` | I2V | [basic_lingbotworld_base_cam.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_lingbotworld_base_cam.py) |
| lingbotworld2 | `robbyant/lingbot-world-v2-14b-causal-fast` | I2V | [basic_lingbotworld2_causal_fast.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_lingbotworld2_causal_fast.py) |
| longcat | `FastVideo/LongCat-Video-T2V-Diffusers` | T2V | [basic_longcat_t2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_longcat_t2v.py) |
| longcat | `FastVideo/LongCat-Video-I2V-Diffusers` | I2V | [basic_longcat_i2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_longcat_i2v.py) |
| longcat | `FastVideo/LongCat-Video-VC-Diffusers` | — | [basic_longcat_vc.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_longcat_vc.py) |
| ltx2 | `FastVideo/LTX2-Distilled-Diffusers`<br>`FastVideo/LTX2.3-Distilled-Diffusers`<br>`FastVideo/LTX-2.3-Distilled-Diffusers` | T2V | [basic_ltx2_distilled.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_ltx2_distilled.py) |
| ltx2 | `Lightricks/LTX-2.3`<br>`FastVideo/LTX2.3-base`<br>`FastVideo/LTX2.3-Diffusers` | T2V | [basic_ltx2.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_ltx2.py) |
| ltx2 | `Lightricks/LTX-2`<br>`FastVideo/LTX2-base`<br>`FastVideo/LTX2-Diffusers` | T2V | [basic_ltx2.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_ltx2.py) |
| mmaudio | `FastVideo/MMAudio-large-44k-v2-Diffusers` | V2A, T2A | [basic_mmaudio.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_mmaudio.py) |
| matrixgame | `FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers`<br>`FastVideo/Matrix-Game-2.0-GTA-Distilled-Diffusers`<br>`FastVideo/Matrix-Game-2.0-TempleRun-Distilled-Diffusers`<br>`FastVideo/Matrix-Game-2.0-Base-Diffusers`<br>`FastVideo/Matrix-Game-2.0-GTA-Diffusers`<br>`FastVideo/Matrix-Game-2.0-TempleRun-Diffusers`<br>`mignonjia/mg_longtuning_distilled_zelda`<br>`mignonjia/mg_sf_distilled_zelda_1k_steps`<br>`mignonjia/mg_sf_distilled_zelda`<br>`mignonjia/mg_causal_zelda`<br>`mignonjia/mg_bidirectional_zelda` | I2V | [basic_matrixgame2.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_matrixgame2.py) |
| matrixgame | `FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers` | I2V | [basic_matrixgame3.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_matrixgame3.py) |
| minimax_h3 | `MiniMaxAI/MiniMax-H3` | T2V, I2V | [T2VA](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_t2v.py)<br>[FL2VA](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_fl2va.py)<br>[Ref2VA](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_ref2va.py) |
| sd35 | `stabilityai/stable-diffusion-3.5-medium` | T2I | [basic_sd35_t2i.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_sd35_t2i.py) |
| stable_audio | `FastVideo/stable-audio-open-1.0-Diffusers` | T2V | [basic_stable_audio.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_stable_audio.py) |
| stable_audio | `FastVideo/stable-audio-open-small-Diffusers` | T2V | [basic_stable_audio_small.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_stable_audio_small.py) |
| turbodiffusion | `loayrashid/TurboWan2.1-T2V-1.3B-Diffusers` | T2V | [basic_turbodiffusion.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_turbodiffusion.py) |
| turbodiffusion | `loayrashid/TurboWan2.1-T2V-14B-Diffusers` | T2V | [basic_turbodiffusion_14b.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_turbodiffusion_14b.py) |
| turbodiffusion | `loayrashid/TurboWan2.2-I2V-A14B-Diffusers` | I2V | [basic_turbodiffusion_i2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_turbodiffusion_i2v.py) |
| wan | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | T2V | [basic.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic.py) |
| wan | `Wan-AI/Wan2.1-T2V-14B-Diffusers`<br>`FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` | T2V | — |
| wan | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | I2V | — |
| wan | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | I2V | — |
| wan | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` | I2V | — |
| wan | `IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers` | — | [basic_wan2_2_Fun.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_wan2_2_Fun.py) |
| wan | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`<br>`FastVideo/FastWan2.1-T2V-14B-480P-Diffusers` | T2V | [basic_dmd.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_dmd.py) |
| wan | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | T2V, I2V | [basic_wan2_2_ti2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_wan2_2_ti2v.py) |
| wan | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`<br>`FastVideo/FastWan2.2-TI2V-5B-Diffusers` | T2V, I2V | [basic_dmd.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_dmd.py) |
| wan | `decart-ai/Lucy-Edit-Dev`<br>`decart-ai/Lucy-Edit-1.1-Dev` | — | [basic_lucy_edit.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_lucy_edit.py) |
| wan | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | T2V | [basic_wan2_2.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_wan2_2.py) |
| wan | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | I2V | [basic_wan2_2_i2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_wan2_2_i2v.py) |
| wan | `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers` | T2V | [basic_self_forcing_causal.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal.py) |
| wan | `rand0nmr/SFWan2.2-T2V-A14B-Diffusers` | T2V | [basic_self_forcing_causal_wan2_2_t2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_t2v.py) |
| wan | `FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers` | I2V | [basic_self_forcing_causal_wan2_2_i2v.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py) |
| zimage | `Tongyi-MAI/Z-Image-Turbo` | T2I | [basic_zimage.py](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_zimage.py) |

**Note (stable_audio)**: the Stable Audio Open pipelines generate audio
(`StableAudioT2AConfig` / `StableAudioOpenSmallConfig`); they are registered
under the generic T2V workload option in the registry.

**Note (MMAudio)**: the registered Hugging Face model ID is reserved but not
yet public. Follow the [MMAudio inference guide](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/pipelines/basic/mmaudio/README.md)
to convert the official weights locally and set `MMAUDIO_MODEL_PATH`.

**Note (MiniMax H3)**: T2VA, FL2VA, and Ref2VA all generate video with stereo
audio. Use the Ref2VA example when passing ordered image, video, or audio
references.

**Note (Wan-VACE)**: not currently supported — no VACE pipeline or registered
model ID exists on `main`
([#1435](https://github.com/hao-ai-lab/FastVideo/issues/1435)). The closest
supported path is the Wan2.1-Fun control pipeline
(`IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers`).

The symbols used have the following meanings:

- ✅ = Full compatibility
- ❌ = No compatibility
- ⭕ = Does not apply to this model

## Models x Optimization

The `HuggingFace Model ID` can be passed directly to
`from_pretrained()`. FastVideo then uses model-specific default settings for
pipeline initialization and sampling.

Registered models absent from this table have not been validated against these
optimizations: absence means **untested**, not incompatible.

<style>
  /* Target tables in this section */
  #models-x-optimization + p + table {
    display: block;
    overflow-x: auto;
    width: 100%;
    font-size: 0.85rem;
  }
  
  #models-x-optimization + p + table td,
  #models-x-optimization + p + table th {
    text-align: center;
    white-space: nowrap;
    padding: 0.5em;
  }
  
  /* First two columns can wrap */
  #models-x-optimization + p + table td:nth-child(1),
  #models-x-optimization + p + table td:nth-child(2) {
    white-space: normal;
    min-width: 120px;
  }
  
  #models-x-optimization + p + table td:nth-child(2) code {
    font-size: 0.75rem;
  }
</style>

| Model Name | HuggingFace Model ID | Resolutions | TeaCache | Sliding Tile Attn (Legacy Branch) | Sage Attn | VSA | BSA |
|------------|---------------------|-------------|----------|-------------------|-----------|-----|-----|
| FastWan2.1 T2V 1.3B | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | 480P | ⭕ | ⭕ | ⭕ | ✅ | ⭕ |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720P | ⭕ | ⭕ | ⭕ | ✅ | ⭕ |
| Wan2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 720P | ⭕ | ⭕ | ✅ | ⭕ | ⭕ |
| DreamX-World 5B Cam | `FastVideo/DreamX-World-5B-Cam-Diffusers` | 480P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| DreamX-World 5B AR | `FastVideo/DreamX-World-5B-Diffusers` | 704px1280p | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Lucy Edit Dev 5B*** | `decart-ai/Lucy-Edit-Dev` | 480P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Wan2.2 T2V A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 480P<br>720P | ❌ | ❌ | ✅ | ⭕ | ⭕ |
| Wan2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 480P<br>720P | ❌ | ❌ | ✅ | ⭕ | ⭕ |
| HunyuanVideo | `hunyuanvideo-community/HunyuanVideo` | 720px1280p<br>544px960p | ❌ | ✅ | ✅ | ⭕ | ⭕ |
| FastHunyuan | `FastVideo/FastHunyuan-diffusers` | 720px1280p<br>544px960p | ❌ | ✅ | ✅ | ⭕ | ⭕ |
| Wan2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 480P | ✅ | ✅ | ✅ | ⭕ | ⭕ |
| Wan2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 480P, 720P | ✅ | ✅ | ✅ | ⭕ | ⭕ |
| Wan2.1 I2V 480P | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 480P | ✅ | ✅ | ✅ | ⭕ | ⭕ |
| Wan2.1 I2V 720P | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 720P | ✅ | ✅ | ✅ | ⭕ | ⭕ |
| TurboWan2.1 T2V 1.3B | `loayrashid/TurboWan2.1-T2V-1.3B-Diffusers` | 480P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| TurboWan2.1 T2V 14B | `loayrashid/TurboWan2.1-T2V-14B-Diffusers` | 480P, 720P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| TurboWan2.2 I2V A14B | `loayrashid/TurboWan2.2-I2V-A14B-Diffusers` | 480P<br>720P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| LongCat T2V 13.6B | `FastVideo/LongCat-Video-T2V-Diffusers` | 480P<br>720P | ❌ | ❌ | ❌ | ⭕ | ✅ |
| Matrix Game 2.0 Base Distilled | `FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers` | 352x640 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Matrix Game 2.0 GTA Distilled | `FastVideo/Matrix-Game-2.0-GTA-Distilled-Diffusers` | 352x640 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Matrix Game 2.0 TempleRun Distilled | `FastVideo/Matrix-Game-2.0-TempleRun-Distilled-Diffusers` | 352x640 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Matrix Game 3.0 Base Distilled | `FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers` | 720x1280 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| GEN3C Cosmos 7B | `FastVideo/GEN3C-Cosmos-7B-Diffusers` | 704px1280p | ❌ | ❌ | ❌ | ⭕ | ⭕ |

## Apple Silicon native runtime

| Release path | Model | Mode | Validated hardware | Status |
| --- | --- | --- | --- | --- |
| MLX FastMetal T2V 1.3B | [`FastVideo/FastMetal-1.3B-QAD`](https://huggingface.co/FastVideo/FastMetal-1.3B-QAD) | 480x832, 81 frames, 3-step DMD, INT8 DiT + TAEHV decode | Apple M4 Max, 16 GB+ unified memory | Released |
| MLX FastMetal TI2V 5B | [`FastVideo/FastMetal-5B-QAD`](https://huggingface.co/FastVideo/FastMetal-5B-QAD) | 480p / 720p, 81 frames, 3-step DMD, INT8 DiT + TAEHV decode | Apple M4 Max, 16 GB+ unified memory | Released |
| MLX FastMetal T2V 14B | [`FastVideo/FastMetal-14B-QAD`](https://huggingface.co/FastVideo/FastMetal-14B-QAD) | 480p / 720p, 81 frames, 3-step DMD, INT8 DiT + TAEHV decode | Apple M4 Max, 36 GB+ unified memory | Released |
| MLX FastH3 Preview T2VA | [`FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2`](https://huggingface.co/FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2) + locally converted DiT | 480p / 720p, 124 frames, 4-step DMD2, INT8/INT6/INT4 **weight-only** DiT, native video + audio VAE; optional temporal RIFE fast mode; optional VSA (tile 64/256, exempt/compete) on `--include-vsa` checkpoints | Apple M4 Max, 36 GB unified memory | Source runtime; T2VA only |

Apple Silicon uses the native MLX runtime. FastMetal-QAD is the packaged Wan
release, while FastH3 Preview currently uses a source checkout and local DiT
conversion. CUDA FastWan-QAD (`FastVideo/FastWan-QAD-1.3B`,
`FastVideo/FastWan-QAD-FP8-1.3B`) is the NVIDIA release. See the
[Apple Silicon guide](../getting_started/installation/mps.md) and the
[FastMetal-QAD blog](https://haoailab.com/blogs/fastmetal/).

**Note**: Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.

***Lucy Edit Dev uses a non-commercial model license. FastVideo support is
focused on inference integration for video editing workflows.

`Sliding Tile Attn (Legacy Branch)` entries refer to the archived
`sta_do_not_delete` branch workflow, not active `main` inference wiring.

## Canonical Supported IDs

The authoritative source for model-ID recognition is
`fastvideo/registry.py`. If a model ID is registered there, FastVideo can
resolve default pipeline and sampling configuration for it.

**Note (GEN3C)**: The official `nvidia/GEN3C-Cosmos-7B` repo provides a raw
`model.pt` checkpoint. Use a Diffusers-format repo (for example,
`FastVideo/GEN3C-Cosmos-7B-Diffusers`) or convert locally with
`scripts/checkpoint_conversion/convert_gen3c_to_fastvideo.py`.

## Hardware and OS

Per the installation guides:

- **NVIDIA GPU (x86_64)** — CUDA 12.6 or 13.0; see the
  [GPU install guide](../getting_started/installation/gpu.md).
- **NVIDIA DGX Spark (GB10, aarch64)** — CUDA 13, from-source kernel build; see
  the [DGX Spark install guide](../getting_started/installation/spark.md).
- **Apple silicon** — macOS 14 or newer; FastMetal-QAD via the MLX runtime. See the
  [Apple Silicon guide](../getting_started/installation/mps.md). The older
  [`basic_mps.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_mps.py)
  demo is PyTorch MPS only.

Optimization-specific hardware constraints (e.g. STA requiring Hopper) are
listed under [Special requirements](#special-requirements).

## Special requirements

### Sliding Tile Attention
- Full STA pipeline usage is on the archived branch:
  https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete
- STA currently requires Hopper GPUs (H100s).

### TurboWan2.1 (TurboDiffusion)
- Uses TurboDiffusionPipeline with RCM scheduler for 1-4 step generation
- Requires SLA attention backend: `export FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN`
- Uses `guidance_scale=1.0` (no classifier-free guidance)

### Matrix Game 2.0
- Image-to-video game world models with keyboard/mouse control input
- Three variants available: Base (universal), GTA, and TempleRun
- Each variant has different keyboard dimensions for control inputs
