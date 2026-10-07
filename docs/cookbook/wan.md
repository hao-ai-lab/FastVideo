---
hide:
- toc
---

# Wan recipes

<div class="cookbook-shell cookbook-family-page" data-cookbook data-family="wan" data-recipes="../../assets/cookbook-recipes.json?v=15">
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
  <nav class="cookbook-jumpnav" aria-label="Recipe page sections">
    <a href="#recipe-builder">Builder</a>
    <a href="#cookbook-setup">Setup</a>
    <a href="#cookbook-troubleshooting">Troubleshooting</a>
    <a href="#cookbook-evidence">Evidence</a>
  </nav>

  <details class="cookbook-modes">
    <summary>Compare Wan modes and options</summary>
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
  </details>

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
        <div class="cookbook-selection-row" data-cookbook-usage>
          <div class="cookbook-selection-row__label">
            <strong>Workflow</strong>
            <span>Both can run locally</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--hardware" role="group" aria-label="How to run this recipe">
            <button type="button" data-cookbook-mode="server" aria-pressed="false"><strong>Run a server</strong><span data-cookbook-server-hint>Playground, cURL, or an API client</span></button>
            <button type="button" data-cookbook-mode="python" aria-pressed="false"><strong>Use Python directly</strong><span>Call the model in your own process</span></button>
          </div>
        </div>
        <p class="cookbook-hardware-note" data-cookbook-serving-availability></p>
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
          <pre id="cookbook-local-command"><code class="language-bash" data-cookbook-command>Loading...</code></pre>
        </div>
        <p class="cookbook-hardware-note" data-cookbook-python-note>Running this script again starts a new process and reloads the model. To iterate in Python, create the generator once and reuse it for multiple prompts.</p>

        <div class="cookbook-serving" data-cookbook-serving hidden>
          <p class="cookbook-serving__intro" data-cookbook-server-lifetime>Start once, then change prompts in the playground or your app. You can run the server and clients on the same machine.</p>
          <section class="cookbook-serving__step" aria-labelledby="serving-install-heading">
            <h4 id="serving-install-heading"><span aria-hidden="true">1</span> Prepare the machine</h4>
            <p>Run from your FastVideo clone in an activated Python environment. See <a data-cookbook-install-guide href="../../getting_started/installation/gpu/">installation requirements</a>.</p>
            <div class="cookbook-command"><div class="cookbook-command__bar"><span>GPU machine · Terminal</span></div><pre id="cookbook-server-install"><code class="language-bash" data-cookbook-server-install></code></pre></div>
            <details class="cookbook-serving__prepare" data-cookbook-prepare hidden><summary>Download and convert MLX weights once</summary><p>Skip this if the weights are already prepared. Edit the paths in the serving config to use your existing files.</p><div class="cookbook-command"><pre id="cookbook-server-prepare"><code class="language-bash" data-cookbook-server-prepare></code></pre></div></details>
          </section>
          <section class="cookbook-serving__step" aria-labelledby="serving-start-heading">
            <h4 id="serving-start-heading"><span aria-hidden="true">2</span> Start the server</h4>
            <p>Keep this terminal running while you use <span data-cookbook-playground-only>the playground or </span>API clients.</p>
            <div class="cookbook-command"><div class="cookbook-command__bar"><span>GPU machine · Terminal</span></div><pre id="cookbook-server-command"><code class="language-bash" data-cookbook-server-command></code></pre></div>
            <details class="cookbook-serving__check"><summary>Check that the server is ready</summary><p>In another terminal, this returns <code>{"status":"ok"}</code> after startup.</p><div class="cookbook-command"><pre id="cookbook-health-command"><code class="language-bash" data-cookbook-health-command></code></pre></div></details>
          </section>
          <section class="cookbook-serving__step" aria-labelledby="serving-client-heading">
            <h4 id="serving-client-heading"><span aria-hidden="true">3</span> Generate and download a video</h4>
            <div class="cookbook-serving__playground" data-cookbook-playground-only>
              <div><strong>Try prompts in your browser</strong><p>Edit a prompt, generate, and watch the result. The playground uses the same server as cURL and your app.</p></div>
              <a class="cookbook-serving__launch" data-cookbook-playground href="http://127.0.0.1:8000/playground/" target="_blank" rel="noopener">Open playground <span aria-hidden="true">↗</span></a>
            </div>
            <p class="cookbook-serving__local-hint"><span data-cookbook-playground-only>Open after the server is ready. </span>On a remote GPU machine, <a href="../openai-api/#connect-your-app">forward port 8000</a> to your computer first.<span data-cookbook-playground-only> This opens a local page, not a hosted demo.</span></p>
            <details class="cookbook-serving__code"><summary>Use cURL or an SDK</summary>
            <p>Each example submits a job, checks its status, and saves the MP4. The Python and JavaScript examples use OpenAI-compatible clients; no OpenAI account is needed.</p>
            <div class="cookbook-serving__clients" role="group" aria-label="API client language">
              <button type="button" data-cookbook-client="curl" aria-pressed="false">cURL</button>
              <button type="button" data-cookbook-client="python" aria-pressed="true">Python</button>
              <button type="button" data-cookbook-client="javascript" aria-pressed="false">JavaScript</button>
            </div>
            <div class="cookbook-command"><div class="cookbook-command__bar"><span>Client dependencies</span></div><pre id="cookbook-client-install"><code class="language-bash" data-cookbook-client-install></code></pre></div>
            <div class="cookbook-command cookbook-command--client"><div class="cookbook-command__bar"><span data-cookbook-client-filename>video.py</span><a data-cookbook-client-source href="https://github.com/hao-ai-lab/FastVideo/tree/main/examples/serving/clients">View source</a></div><pre id="cookbook-client-code"><code data-cookbook-client-code></code></pre></div>
            <p data-cookbook-client-run></p>
            </details>
          </section>
          <p class="cookbook-serving__boundary">This is a local development server without built-in API-key authentication. The client key <code>local</code> is a placeholder. Keep the server on loopback; use an authenticated TLS proxy before exposing it publicly. Run the JavaScript client in your webapp's backend, not in a browser with a private key.</p>
          <a href="../openai-api/">Server guide and API compatibility →</a>
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
        <li>Out of memory on the A14B recipes: the checked-in sources already enable CPU offload; see <a href="../../inference/configuration/">Configuration</a> for the offload surface before reducing resolution or frames.</li>
        <li>The FastWan2.1 recipe requires <code>VIDEO_SPARSE_ATTN</code>; confirm the environment variable in the command was set in the same shell.</li>
        <li>FastMetal MLX: install with the <a href="../../getting_started/installation/mlx/">MLX install guide</a>, then pick a FastMetal recipe in the builder. CUDA FastWan-QAD checkpoints are refused on the MLX runtime.</li>
        <li>FastMetal 5B uses <code>mlx_wan22_generate.py</code>. 1.3B and 14B use <code>mlx_wan_prompt_to_video.py</code>.</li>
        <li>Gated or missing checkpoints: run <code>huggingface-cli login</code> and confirm you accepted the model's license on Hugging Face.</li>
      </ul>
  </div>
</details>

<details class="cookbook-collapsible" id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div class="cookbook-collapsible__body">
      <p>Every recipe on this page maps to a checked-in FastVideo source. The FastMetal MLX releases include the recorded M4 Max system memory, documented unified-memory floor, and measured peak MLX memory. CUDA entries remain <strong>Source-backed</strong> where the examples record a GPU count but no exact GPU model or VRAM. Unlisted hardware is unknown, not unsupported.</p>
  </div>
</details>
