# Serving cookbook examples

<!--
TODO(cookbook-ui): Remove this demo page, including its inline HTML/CSS and
walkthrough, when the new cookbook UI components are integrated. Move any useful
recipe-authoring guidance into the permanent contributor docs first, then remove
the demo navigation entry and replace links to this page. Remove the temporary
PR screenshot at docs/assets/images/cookbook-serving-example.png with the demo.
-->

This prototype follows three CUDA recipes through **authored YAML metadata →
generated browser JSON → option resolution → explicit CLI commands**. It is separate
from the existing cookbook. The page only generates text; it never launches a server
or submits an inference request.

These GB200 recipes are **source-configured, not GPU-benchmarked by this prototype**.
Only the controls below are exposed. All other recipe settings remain visible in the
server command. Installation assumes a FastVideo checkout and an activated environment;
the flag-only serving command uses the accompanying local CLI change.

<style>
[data-serving-controls], [data-serving-request-controls] { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
.serving-example-option { display: flex; flex-direction: column; gap: .25rem; }
.serving-example-option input:not([type=checkbox]), .serving-example-option select { border: 1px solid var(--md-default-fg-color--lighter); padding: .4rem; }
.serving-example-option input[type=checkbox] { align-self: flex-start; }
.serving-example-option small { color: var(--md-default-fg-color--light); }
[data-serving-error] { color: var(--md-typeset-a-color); }
[data-serving-example] pre { max-height: 36rem; overflow: auto; }
</style>

<div data-serving-example data-recipe="fasth3-v2" data-metadata="../../assets/cookbook-serving-example.json">
  <label class="serving-example-option"><strong>Model</strong><select data-serving-model aria-label="Model"></select></label>
  <p><strong data-serving-context>Loading model metadata…</strong></p>
  <h2>Server options</h2>
  <div data-serving-controls></div>
  <h2>Generation request</h2>
  <div data-serving-request-controls></div>
  <p data-serving-error role="status" aria-live="polite">Loading the example…</p>
  <div data-serving-output hidden>
    <h2>1. Install</h2>
    <button type="button" data-serving-copy="installCommand">Copy install command</button>
    <pre><code data-serving-code="installCommand"></code></pre>
    <h2>2. Start the server</h2>
    <p>Every resolved recipe setting is emitted explicitly. No example configuration file is loaded.</p>
    <button type="button" data-serving-copy="command">Copy server command</button>
    <pre><code data-serving-code="command"></code></pre>
    <h2>3. Check readiness</h2>
    <button type="button" data-serving-copy="healthCommand">Copy health command</button>
    <pre><code data-serving-code="healthCommand"></code></pre>
    <h2 data-serving-client-heading>4. Generate one video</h2>
    <p data-serving-client-note></p>
    <button type="button" data-serving-copy="clientCommand">Copy client command</button>
    <pre><code data-serving-code="clientCommand"></code></pre>
    <details>
      <summary>Inspect the resolved object before CLI serialization</summary>
      <pre><code data-serving-config></code></pre>
    </details>
  </div>
</div>

## Follow the implementation

Switching models rebuilds the controls and resets defaults and request inputs. These
three recipes demonstrate different task and model options using the same resolver:

| Model recipe | Task | GPU count | Model/task differences |
| --- | --- | --- | --- |
| [FastH3 V2](serving/models/fasth3-v2.yaml) | Text to video | 4 | 9 sigma points / 8 forwards, 124 frames, model-specific VSA sparsity |
| [Z-Image Turbo](serving/models/zimage-turbo.yaml) | Text to image | 1 | 8 steps, guidance 0, no video-frame controls; returns PNG in base64 JSON |
| [Wan2.1 I2V](serving/models/wan21-i2v.yaml) | Image to video | 2 | 40 steps, 77 frames, required reference-image path or URL |

1. [defaults.yaml](serving/defaults.yaml) defines common controls, types, default values,
   task applicability, runtime launch tokens, and dotted configuration paths.
2. [hardware.yaml](serving/hardware.yaml) defines the exposed GPU type, NVIDIA GB200,
   and its runtime. It does not set the number of GPUs.
3. Each model YAML above owns its full baseline and model differences. For example,
   FastH3 narrows GPUs to `[4]`, sampling points to `[9]`, and links GPU count to
   sequence parallelism. It adds its VSA sparsity control, fixed at `0.8`, without
   changing the shared option catalog or renderer. Z-Image adds guidance scale instead.
4. `docs/cookbook_serving.py` uses strict Pydantic models to validate these files and serializes them during the docs build into
   [cookbook-serving-example.json](../assets/cookbook-serving-example.json).
   This generated file is delivery data, not a second maintained configuration.
   Model validators check option bounds, recipe references, and configuration-path
   ownership. Shared defaults and partial model overrides stay separate in the JSON.
5. [cookbook-serving.js](../assets/cookbook-serving.js) exposes `resolveRecipe()`.
   It combines definitions, validates selections, creates one resolved object, and
   serializes that object into CLI flags. The generic controls call the same function
   whenever a selection changes.

GPU type and GPU count are separate selections. This example exposes only the
GB200 type; each model recipe restricts GPU count and sets sequence parallelism to
that count. The hardware catalog does not choose or constrain the count.

Default precedence is **shared → model → explicit user selection**.
Model properties override common properties; choice lists replace them. To observe
the flow, change the port: the server, health check, and client commands update
together. Change VAE offload or seed to see the corresponding dotted flag update.
Invalid values hide the generated output until corrected.

Prompt and reference image are request inputs, not server flags. Video requests use
the resolved server sampling defaults. The image endpoint does not merge those
defaults, so its generated request includes the resolved image sampling settings
explicitly. Its command saves `response.json`, then decodes the PNG to `output.png`.

For another already-supported model, a contributor would add one model recipe and a
default-output verification case, reusing these controls, resolver, and CLI formatter.
New runtimes, automatic hardware fitting, other tasks, and advanced dependencies are
outside these examples.

To inspect just the metadata build and resolver tests from the repository root:

```bash
python docs/cookbook_serving.py
node --test tests/local_tests/test_cookbook_serving_example.mjs
```
