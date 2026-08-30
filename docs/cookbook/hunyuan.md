---
hide:
- toc
---

# Hunyuan recipes

<div class="cookbook-shell cookbook-family-page" data-cookbook data-family="hunyuan" data-recipes="../../assets/cookbook-recipes.json?v=4">
  <header class="cookbook-family-header">
    <a class="cookbook-back-link" href="../"><span aria-hidden="true">←</span> All model families</a>
    <div class="cookbook-family-header__body">
      <span class="cookbook-family-header__logo">
        <img class="off-glb" src="../../assets/logos/tencent-hunyuan.webp" alt="Tencent Hunyuan" width="112" height="112">
      </span>
      <div>
        <p class="cookbook-eyebrow">Maintained family · Inference</p>
        <h2>Hunyuan inference recipes</h2>
        <p>HunyuanVideo 1.5 is Tencent's video generation family. The maintained examples cover 480p text-to-video with CPU offload and a full 480p-to-1080p upscale chain.</p>
      </div>
    </div>
    <div class="cookbook-lifecycle" aria-label="Lifecycle stages">
      <span class="cookbook-lifecycle__stage cookbook-lifecycle__stage--active">Inference <small>live</small></span>
      <span class="cookbook-lifecycle__stage">Distillation <small>planned</small></span>
      <span class="cookbook-lifecycle__stage">Fine-tuning <small>planned</small></span>
      <span class="cookbook-lifecycle__stage">Training <small>planned</small></span>
      <span class="cookbook-lifecycle__stage">Evaluation <small>planned</small></span>
      <span class="cookbook-lifecycle__stage">Optimization <small>planned</small></span>
      <span class="cookbook-lifecycle__stage">Deployment <small>planned</small></span>
    </div>
  </header>

  <section class="cookbook-builder" id="recipe-builder" aria-labelledby="builder-heading">
    <div class="cookbook-builder__intro">
      <h2 id="builder-heading">Choose a model and GPU count</h2>
      <p>Choose a model, then pick a GPU count. Each selection maps to one
      checked-in recipe. If that recipe does not use your chosen count, the
      page says so instead of guessing.</p>
    </div>

    <div class="cookbook-builder__layout">
      <div class="cookbook-controls">
        <div class="cookbook-selection-row">
          <div class="cookbook-selection-row__label">
            <strong>Recipe</strong>
            <span>Task and checkpoint</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--models" data-cookbook-model-options role="group" aria-label="Recipe">
            <button type="button" disabled>Loading Hunyuan recipes...</button>
          </div>
        </div>

        <div class="cookbook-selection-row">
          <div class="cookbook-selection-row__label">
            <strong>Runtime</strong>
            <span>Maintained paths only</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--hardware" data-cookbook-hardware-options role="group" aria-label="Runtime">
            <button type="button" disabled>Loading runtimes...</button>
          </div>
        </div>

        <p class="cookbook-selection-description" data-cookbook-description>Loading recipe details...</p>
        <p class="cookbook-hardware-note">Exact device and memory details appear only when a recorded run supports them.</p>

        <div class="cookbook-hardware-state" data-cookbook-hardware-state role="status" aria-live="polite">
          Reading recipe evidence...
        </div>
      </div>

      <article class="cookbook-result">
        <div class="cookbook-result__header">
          <h3 data-cookbook-label>Loading...</h3>
          <div class="cookbook-result__badges">
            <span class="cookbook-badge">Maintained</span>
            <span class="cookbook-badge" data-cookbook-evidence>Source-backed</span>
            <span class="cookbook-badge cookbook-badge--neutral" data-cookbook-hardware-badge>Source config</span>
          </div>
        </div>

        <dl class="cookbook-result__facts">
          <div><dt>Model</dt><dd data-cookbook-model>Loading...</dd></div>
          <div><dt>Workload</dt><dd data-cookbook-task>Loading...</dd></div>
          <div><dt>Source configuration</dt><dd data-cookbook-gpus>Loading...</dd></div>
          <div><dt>Expected output</dt><dd data-cookbook-artifact>Loading...</dd></div>
        </dl>

        <div class="cookbook-command">
          <div class="cookbook-command__bar">
            <span>Terminal</span>
          </div>
          <pre><code class="language-bash" data-cookbook-command>Loading...</code></pre>
        </div>

        <div class="cookbook-result__footer">
          <a data-cookbook-source href="../../inference/examples/basic/">Open example source</a>
          <a data-cookbook-model-link href="https://huggingface.co/Tencent-Hunyuan">View model card</a>
        </div>
        <p class="cookbook-picker__status" role="status" aria-live="polite" data-cookbook-status></p>
      </article>
    </div>

    <noscript>
      <div class="cookbook-noscript">
        JavaScript is needed for the guided selector. You can still browse the
        <a href="../../inference/examples/examples_inference_index/">maintained inference examples</a>.
      </div>
    </noscript>
  </section>
</div>

## Before you run

The generated commands expect a local clone:

    git clone https://github.com/hao-ai-lab/FastVideo.git
    cd FastVideo

Use [Configuration](../inference/configuration.md) for supported Python and
CLI settings, [Optimizations](../inference/optimizations.md) for attention and
memory tradeoffs, and the [support matrix](../inference/support_matrix.md) for
the supported model and optimization surface.

## Troubleshooting

- Out of memory: `basic_hy15.py` already enables dit/VAE/text-encoder CPU offload; further tradeoffs are described in [Optimizations](../inference/optimizations.md).
- `pin_cpu_memory` errors on low-RAM machines are documented inline in the example; set it to false as the source comment suggests.
- Gated or missing checkpoints: run `huggingface-cli login` and confirm you accepted the model's license on Hugging Face.

## Evidence status

All recipes on this page are **Source-backed**: their commands, model IDs, and flags were validated against the checked-in FastVideo sources listed above (static validation). No runtime GPU validation is recorded for these recipes, so GPU model fit, memory use, throughput, and runtime duration are **Unknown** and deliberately not claimed. Runtime buttons show only the GPU counts configured in checked-in sources.
