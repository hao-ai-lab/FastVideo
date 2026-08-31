---
hide:
- toc
---

# Wan recipes

<div class="cookbook-shell cookbook-family-page" data-cookbook data-family="wan" data-recipes="../../assets/cookbook-recipes.json?v=5">
  <header class="cookbook-family-header">
    <a class="cookbook-back-link" href="../"><span aria-hidden="true">←</span> All model families</a>
    <div class="cookbook-family-header__body">
      <span class="cookbook-family-header__logo">
        <img class="off-glb" src="../../assets/logos/wan-ai.webp" alt="Wan-AI" width="112" height="112">
      </span>
      <div>
        <p class="cookbook-eyebrow">Maintained family · Inference</p>
        <h2>Wan inference recipes</h2>
        <p>CUDA covers FastWan and Wan2.1/2.2 text and image recipes. Apple Silicon uses the released FastMetal 1.3B, 5B, and 14B MLX T2V paths. The speed flags below are switches on those same scripts, not extra recipes.</p>
        <span class="cookbook-count" data-cookbook-count>7 maintained recipes</span>
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

  <section class="cookbook-modes" aria-labelledby="wan-modes-heading">
    <h2 id="wan-modes-heading">Supported modes</h2>
    <p>
      FastMetal MLX is T2V in the checked-in examples. Image-to-video and
      TI2V stay on the CUDA recipes. Temporal <code>--fast</code> composes
      with either spatial path. <code>--refine</code> and
      <code>--fast-spatial</code> cannot run together.
      <code>--refine</code> wins if both are set.
      <code>basic_mps.py</code> is the older PyTorch MPS demo and is not a
      FastMetal recipe.
    </p>
    <div class="cookbook-modes__table-wrap">
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            <th>CUDA</th>
            <th>MLX FastMetal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>T2V</td>
            <td>FastWan2.1 1.3B, Wan2.2 A14B</td>
            <td>1.3B, 5B, and 14B</td>
          </tr>
          <tr>
            <td>I2V</td>
            <td>Wan2.1 14B 480P</td>
            <td>Not in the released examples</td>
          </tr>
          <tr>
            <td>TI2V</td>
            <td>Wan2.2 TI2V 5B</td>
            <td>FastMetal 5B is T2V in <code>mlx_wan22_generate.py</code></td>
          </tr>
          <tr>
            <td>Temporal <code>--fast</code></td>
            <td>No cookbook recipe</td>
            <td>RIFE. Fewer frames, then interpolate to <code>--num-frames</code></td>
          </tr>
          <tr>
            <td>Spatial <code>--fast-spatial</code></td>
            <td>No cookbook recipe</td>
            <td>Denoise and decode at half resolution, then upsample. No second denoise</td>
          </tr>
          <tr>
            <td>Two-pass <code>--refine</code></td>
            <td>No cookbook recipe</td>
            <td>Denoise at base resolution, upsample, re-noise, denoise again. Wins over <code>--fast-spatial</code></td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  <section class="cookbook-builder" id="recipe-builder" aria-labelledby="builder-heading">
    <div class="cookbook-builder__intro">
      <h2 id="builder-heading">Pick a recipe and runtime</h2>
      <p>Choose the result you want, then use a maintained CUDA or native MLX path.</p>
    </div>

    <div class="cookbook-builder__layout">
      <div class="cookbook-controls">
        <div class="cookbook-selection-row">
          <div class="cookbook-selection-row__label">
            <strong>Recipe</strong>
            <span>Task and checkpoint</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--models" data-cookbook-model-options role="group" aria-label="Recipe">
            <button type="button" disabled>Loading Wan recipes...</button>
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
          <a data-cookbook-model-link href="https://huggingface.co/Wan-AI">View model card</a>
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

- Out of memory on the A14B recipes: the checked-in sources already enable CPU offload; see [Configuration](../inference/configuration.md) for the offload surface before reducing resolution or frames.
- The FastWan2.1 recipe requires `VIDEO_SPARSE_ATTN`; confirm the environment variable in the command was set in the same shell.
- FastMetal MLX: install with `uv pip install -e ".[mlx]"`, then follow the [Apple Silicon guide](../getting_started/installation/mps.md). CUDA FastWan-QAD checkpoints are refused on the MLX runtime.
- FastMetal 5B uses `mlx_wan22_generate.py`. 1.3B and 14B use `mlx_wan_prompt_to_video.py`.
- Gated or missing checkpoints: run `huggingface-cli login` and confirm you accepted the model's license on Hugging Face.

## Evidence status

Every recipe on this page maps to a checked-in FastVideo source. The FastMetal MLX releases include the recorded M4 Max system memory, documented unified-memory floor, and measured peak MLX memory. CUDA entries remain **Source-backed** where the examples record a GPU count but no exact GPU model or VRAM. Unlisted hardware is unknown, not unsupported.
