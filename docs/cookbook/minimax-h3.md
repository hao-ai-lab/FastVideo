---
hide:
- toc
---

# MiniMax H3 recipes

<div class="cookbook-shell cookbook-family-page" data-cookbook data-family="minimax_h3" data-recipes="../../assets/cookbook-recipes.json?v=4">
  <header class="cookbook-family-header">
    <a class="cookbook-back-link" href="../"><span aria-hidden="true">←</span> All model families</a>
    <div class="cookbook-family-header__body">
      <span class="cookbook-family-header__logo">
        <img class="off-glb" src="../../assets/logos/minimax.webp" alt="MiniMax" width="112" height="112">
      </span>
      <div>
        <p class="cookbook-eyebrow">Primary focus · Inference</p>
        <h2>MiniMax H3 recipes</h2>
        <p>Generate synchronized video and audio with the full H3 checkpoint, the four-step FastH3 Preview, reference and frame-conditioned paths, or the native Apple Silicon MLX runtime.</p>
        <span class="cookbook-count" data-cookbook-count>6 maintained recipes</span>
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
      <h2 id="builder-heading">Pick an H3 recipe and runtime</h2>
      <p>Choose the result you want, then use a maintained CUDA or MLX path.
      Device claims stay tied to checked-in sources and recorded runs.</p>
    </div>

    <div class="cookbook-builder__layout">
      <div class="cookbook-controls">
        <div class="cookbook-selection-row">
          <div class="cookbook-selection-row__label">
            <strong>Recipe</strong>
            <span>Task and checkpoint</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--models" data-cookbook-model-options role="group" aria-label="Recipe">
            <button type="button" disabled>Loading MiniMax H3 recipes...</button>
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
          <div><dt>Hardware</dt><dd data-cookbook-gpus>Loading...</dd></div>
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
          <a data-cookbook-model-link href="https://huggingface.co/MiniMaxAI">View model card</a>
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

CUDA FastH3 uses the pinned performance dependencies:

    UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"

Apple Silicon uses the native MLX extra and a locally converted H3 DiT:

    uv pip install -e ".[mlx]"

Follow the [Apple Silicon guide](../getting_started/installation/mps.md#run-fasth3-preview)
for the download, conversion, and storage requirements.

## Troubleshooting

- The full CUDA H3 examples request four GPUs by default. Their sources do not claim a GPU model or memory minimum.
- The FastH3 CUDA performance profile was measured on four GB200 GPUs. Use its strict profile when exact operation order matters more than the measured performance configuration.
- The MLX source runtime is limited to T2VA. FL2VA, Ref2VA, VSA, spatial fast mode, and two-pass refinement are not wired on MLX.
- Gated or missing checkpoints: run `huggingface-cli login` and confirm you accepted the model's license on Hugging Face.

## Evidence status

Every command, model ID, and flag on this page maps to a checked-in FastVideo source. Recipes marked **Verified** also have a recorded hardware path in linked FastVideo evidence. The full H3 CUDA examples remain **Source-backed** where the source records a GPU count but no GPU model or memory requirement. Unlisted hardware is unknown, not unsupported.
