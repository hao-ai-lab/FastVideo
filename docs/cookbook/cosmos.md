---
hide:
- toc
---

# Cosmos recipes

<div class="cookbook-shell cookbook-family-page" data-cookbook data-family="cosmos" data-recipes="../../assets/cookbook-recipes.json?v=7">
  <header class="cookbook-family-header">
    <a class="cookbook-back-link" href="../"><span aria-hidden="true">←</span> All model families</a>
    <div class="cookbook-family-header__body">
      <span class="cookbook-family-header__logo">
        <img class="off-glb" src="../../assets/logos/nvidia.webp" alt="NVIDIA" width="112" height="112">
      </span>
      <div>
        <p class="cookbook-eyebrow">Maintained family · Inference</p>
        <h2>Cosmos inference recipes</h2>
        <p>NVIDIA Cosmos Predict 2.5 generates navigable world videos. The maintained example runs the 2B text-to-world checkpoint on a single GPU.</p>
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
  <nav class="cookbook-jumpnav" aria-label="Recipe page sections">
    <a href="#recipe-builder">Builder</a>
    <a href="#cookbook-setup">Setup</a>
    <a href="#cookbook-troubleshooting">Troubleshooting</a>
    <a href="#cookbook-evidence">Evidence</a>
  </nav>

  <section class="cookbook-builder" id="recipe-builder" aria-labelledby="builder-heading">
    <div class="cookbook-builder__intro">
      <h2 id="builder-heading">Pick a recipe and runtime</h2>
      <p>Start with the result you want, then choose one of the runtimes FastVideo actually maintains for it.</p>
    </div>

    <div class="cookbook-builder__layout">
      <div class="cookbook-controls">
        <div class="cookbook-selection-row">
          <div class="cookbook-selection-row__label">
            <strong>Recipe</strong>
            <span>Task and checkpoint</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--models" data-cookbook-model-options role="group" aria-label="Recipe">
            <button type="button" disabled>Loading Cosmos recipes...</button>
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
          <a data-cookbook-model-link href="https://huggingface.co/nvidia">View model card</a>
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

<details class="cookbook-collapsible" id="cookbook-setup">
  <summary>Setup</summary>
  <div class="cookbook-collapsible__body">
      <p>The generated commands expect a local clone:</p>
      <pre><code>git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo</code></pre>
      <p>Use <a href="../../inference/configuration/">Configuration</a> for supported Python and CLI settings, <a href="../../inference/optimizations/">Optimizations</a> for attention and memory tradeoffs, and the <a href="../../inference/support_matrix/">support matrix</a> for the supported model and optimization surface.</p>
  </div>
</details>

<details class="cookbook-collapsible" id="cookbook-troubleshooting">
  <summary>Troubleshooting</summary>
  <div class="cookbook-collapsible__body">
      <ul>
        <li>World-generation prompts work best describing a scene and camera motion; the built-in prompt in the example is a known-good starting point.</li>
        <li>Gated or missing checkpoints: run <code>huggingface-cli login</code> and confirm you accepted the model's license on Hugging Face.</li>
      </ul>
  </div>
</details>

<details class="cookbook-collapsible" id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div class="cookbook-collapsible__body">
      <p>All recipes on this page are <strong>Source-backed</strong>: their commands, model IDs, and flags were validated against the checked-in FastVideo sources listed above (static validation). No runtime GPU validation is recorded for these recipes, so GPU model fit, memory use, throughput, and runtime duration are <strong>Unknown</strong> and deliberately not claimed. Runtime buttons show only the GPU counts configured in checked-in sources.</p>
  </div>
</details>
