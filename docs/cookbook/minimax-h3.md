---
hide:
- toc
---

# MiniMax H3 recipes

<div class="cookbook-shell cookbook-family-page" data-cookbook data-family="minimax_h3" data-default-recipe="fasth3-preview-cuda" data-recipes="../../assets/cookbook-recipes.json?v=8">
  <header class="cookbook-family-header">
    <a class="cookbook-back-link" href="../"><span aria-hidden="true">←</span> All model families</a>
    <div class="cookbook-family-header__body">
      <span class="cookbook-family-header__logo">
        <img class="off-glb" src="../../assets/logos/minimax.webp" alt="MiniMax" width="112" height="112">
      </span>
      <div>
        <p class="cookbook-eyebrow">Primary focus · Inference</p>
        <h2>MiniMax H3 recipes</h2>
        <p>Generate video and audio with H3. Run a server on CUDA, one DGX Spark, or Apple Silicon MLX to iterate on prompts, or call the pipeline directly from Python.</p>
        <span class="cookbook-count" data-cookbook-count>8 maintained recipes</span>
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
    <summary>Compare H3 modes and options</summary>
    <h2 id="h3-modes-heading">Supported modes</h2>
    <p>
      CUDA covers T2VA, FL2VA, and Ref2VA on the full checkpoint, plus FastH3
      Preview and FastH3 LoRA. FastH3 Preview also has a DGX Spark runtime with
      a 1-Spark or 2-Spark device row. MLX is T2VA only. Temporal <code>--fast</code>,
      spatial <code>--fast-spatial</code>, and opt-in VSA are flags on the same
      MLX script, not extra recipes.
    </p>
    <div class="cookbook-modes__table-wrap">
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            <th>CUDA</th>
            <th>MLX FastH3</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>T2VA</td>
            <td>Full H3, FastH3 Preview, FastH3 LoRA</td>
            <td>FastH3 Preview after a local DiT conversion</td>
          </tr>
          <tr>
            <td>FL2VA</td>
            <td>Full H3</td>
            <td>Not wired</td>
          </tr>
          <tr>
            <td>Ref2VA</td>
            <td>Full H3</td>
            <td>Not wired</td>
          </tr>
          <tr>
            <td>Temporal <code>--fast</code></td>
            <td>No cookbook recipe</td>
            <td>Shorter video denoise, MLX RIFE back to <code>--num-frames</code>, full-duration audio</td>
          </tr>
          <tr>
            <td>VSA</td>
            <td>Trained sparse attention on FastH3 CUDA</td>
            <td>Opt-in. Convert with <code>--include-vsa</code> into a new directory such as <code>./FastH3-MLX-vsa</code>, then pass <code>--vsa</code>. Do not overwrite an existing dense export.</td>
          </tr>
          <tr>
            <td>Spatial <code>--fast-spatial</code></td>
            <td>No cookbook recipe</td>
            <td>Denoise and decode at height/width divided by <code>--fast-spatial-scale</code>, then resample. Composes with <code>--fast</code>.</td>
          </tr>
          <tr>
            <td>Two-pass refine</td>
            <td>No cookbook recipe</td>
            <td>Not wired</td>
          </tr>
          <tr>
            <td>DGX Spark</td>
            <td>FastH3 Preview on one GB10, or two Sparks with Ray sequence parallel (<code>sp_size=2</code>) over QSFP RoCE. Select NVIDIA DGX Spark, then 1 Spark or 2 Sparks.</td>
            <td>Not wired</td>
          </tr>
        </tbody>
      </table>
    </div>
  </details>

  <section class="cookbook-builder" id="recipe-builder" aria-labelledby="builder-heading">
    <div class="cookbook-builder__intro">
      <h2 id="builder-heading">Pick an H3 recipe and runtime</h2>
      <p>Choose the result you want, then use a maintained CUDA, DGX Spark, or MLX path.
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

        <div class="cookbook-selection-row" data-cookbook-device-row hidden>
          <div class="cookbook-selection-row__label">
            <strong>Devices</strong>
            <span data-cookbook-device-caption>1 Spark or a QSFP pair</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--hardware" data-cookbook-device-options role="group" aria-label="Devices">
          </div>
        </div>

        <div data-cookbook-knobs></div>

        <p class="cookbook-selection-description" data-cookbook-description>Loading recipe details...</p>
        <div class="cookbook-selection-row" data-cookbook-usage>
          <div class="cookbook-selection-row__label">
            <strong>Workflow</strong>
            <span>Both can run locally</span>
          </div>
          <div class="cookbook-option-grid cookbook-option-grid--hardware" role="group" aria-label="How to run this recipe">
            <button type="button" data-cookbook-mode="server" aria-pressed="false"><strong>Run a server</strong><span>Playground, cURL, or an API client</span></button>
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
            <details class="cookbook-serving__prepare" data-cookbook-prepare hidden><summary>Download and convert MLX weights once</summary><p>Skip this if the weights are already prepared. Edit the paths in <code>examples/serving/mlx_fasth3.yaml</code> to use your existing files. Install <code>ffmpeg</code> for video and audio output.</p><div class="cookbook-command"><pre id="cookbook-server-prepare"><code class="language-bash" data-cookbook-server-prepare></code></pre></div></details>
          </section>
          <section class="cookbook-serving__step" aria-labelledby="serving-start-heading">
            <h4 id="serving-start-heading"><span aria-hidden="true">2</span> Start the server</h4>
            <p>Keep this terminal running while you use the playground or API clients.</p>
            <div class="cookbook-command"><div class="cookbook-command__bar"><span>GPU machine · Terminal</span></div><pre id="cookbook-server-command"><code class="language-bash" data-cookbook-server-command></code></pre></div>
            <details class="cookbook-serving__check"><summary>Check that the server is ready</summary><p>In another terminal, this returns <code>{"status":"ok"}</code> after startup.</p><div class="cookbook-command"><pre id="cookbook-health-command"><code class="language-bash" data-cookbook-health-command></code></pre></div></details>
          </section>
          <section class="cookbook-serving__step" aria-labelledby="serving-client-heading">
            <h4 id="serving-client-heading"><span aria-hidden="true">3</span> Generate and download a video</h4>
            <div class="cookbook-serving__playground">
              <div><strong>Try prompts in your browser</strong><p>Edit a prompt, generate, and watch the result. The playground uses the same server as cURL and your app.</p></div>
              <a class="cookbook-serving__launch" data-cookbook-playground href="http://127.0.0.1:8000/playground/" target="_blank" rel="noopener">Open playground <span aria-hidden="true">↗</span></a>
            </div>
            <p class="cookbook-serving__local-hint">Open after the server is ready. On a remote GPU machine, <a href="../openai-api/#connect-your-app">forward port 8000</a> to your computer first. This opens a local page, not a hosted demo.</p>
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
          <a data-cookbook-model-link href="https://huggingface.co/MiniMaxAI">View model card</a>
        </div>
        <p class="cookbook-picker__status" role="status" aria-live="polite" data-cookbook-status></p>
      </article>
    </div>

    <noscript>
      <div class="cookbook-noscript">
        JavaScript is needed for the guided selector. You can still browse the
        <a href="../openai-api/">H3 server guide</a> or the
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
      <p class="cookbook-eyebrow">CUDA</p>
      <pre><code>UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"</code></pre>
      <p class="cookbook-eyebrow">Apple Silicon</p>
      <pre><code>uv pip install -e ".[mlx]"</code></pre>
      <p>Follow the <a href="../../getting_started/installation/mps/#run-fasth3-preview">Apple Silicon guide</a> for the download, conversion, and storage requirements.</p>
      <p class="cookbook-eyebrow">NVIDIA DGX Spark</p>
      <pre><code>UV_TORCH_BACKEND=cu130 uv pip install -e .</code></pre>
      <p>Follow the <a href="../../getting_started/installation/spark/">DGX Spark install guide</a> for ARM64 CUDA 13. One Spark is a local process. Two Sparks need Ray on the QSFP link:</p>
      <pre><code>uv pip install ray</code></pre>
      <p>Bring up the cluster from <a href="../../getting_started/installation/spark_pair/">pairing two Sparks</a> before selecting 2 Sparks in the builder.</p>
  </div>
</details>

<details class="cookbook-collapsible" id="cookbook-troubleshooting">
  <summary>Troubleshooting</summary>
  <div class="cookbook-collapsible__body">
      <ul>
        <li>The full CUDA H3 examples request four GPUs by default. Their sources do not claim a GPU model or memory minimum.</li>
        <li>The FastH3 CUDA performance profile was measured on four GB200 GPUs. Use its strict profile when exact operation order matters more than the measured performance configuration.</li>
        <li>The MLX source runtime supports T2VA, optional temporal <code>--fast</code>, optional spatial <code>--fast-spatial</code>, and opt-in VSA on <code>--include-vsa</code> checkpoints. FL2VA, Ref2VA, and two-pass refinement are not wired.</li>
        <li>GPU count and VAE decode backend are configurable in the builder above for FastH3 CUDA recipes. Only the value shown by default has a recorded run; other supported values are unmeasured here.</li>
        <li>DGX Spark is a runtime on FastH3 Preview, not a separate family card. Select NVIDIA DGX Spark, then 1 Spark or 2 Sparks. The CUDA GPU-count knob does not apply to Spark.</li>
        <li>GB10 has no FA4 / sm_100a VSA kernel. Keep <code>FASTVIDEO_FA4=0</code> and <code>FASTVIDEO_VSA_SM100A=0</code>. Legal <code>num_frames</code> values are <code>17n+5</code>, capped at 345 (15 s). A 345-frame request on one Spark can OOM.</li>
        <li>Gated or missing checkpoints: run <code>huggingface-cli login</code> and confirm you accepted the model's license on Hugging Face.</li>
      </ul>
  </div>
</details>

<details class="cookbook-collapsible" id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div class="cookbook-collapsible__body">
      <p>Every command, model ID, and flag on this page maps to a checked-in FastVideo source. Recipes marked <strong>Verified</strong> also have a recorded hardware path in linked FastVideo evidence. The full H3 CUDA examples remain <strong>Source-backed</strong> where the source records a GPU count but no GPU model or memory requirement. Unlisted hardware is unknown, not unsupported.</p>
  </div>
</details>
