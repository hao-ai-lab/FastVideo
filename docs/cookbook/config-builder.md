# Serving configuration builder

<!-- TODO(cookbook-ui): Replace this temporary demo page and inline styling with the final cookbook UI. -->

Select a model to inspect its serving options and declared defaults.
This demo offers a small selection of registered models and loads the selected model's configuration on demand.
The schemas and defaults come from FastVideo's existing configuration code.
Changed options are written to YAML; untouched values remain inherited at runtime.
Ajv checks the schema. Model, checkpoint and hardware restrictions remain runtime checks.

<style>
.config-builder-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; padding: 1rem; }
.config-builder-section { margin: .5rem 0; }
.config-builder-field { display: flex; flex-direction: column; gap: .3rem; }
.config-builder-field input:not([type=checkbox]), .config-builder-field select, .config-builder-field textarea { padding: .45rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: .2rem; width: 100%; }
.config-builder-field input[type=checkbox] { align-self: flex-start; }
.config-builder-field small { color: var(--md-default-fg-color--light); overflow-wrap: anywhere; }
.config-builder-actions { display: flex; flex-wrap: wrap; gap: .5rem; margin: .6rem 0; }
[data-config-builder] button { padding: .4rem .7rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: .2rem; }
[data-config-builder] button:disabled { opacity: .5; }
[data-config-builder] pre { max-height: 32rem; overflow: auto; }
[data-config-error] { color: var(--md-typeset-a-color); }
</style>

<div data-config-builder data-metadata="../../assets/cookbook-config/index.json">
  <label class="config-builder-field"><strong>Model ID</strong><select data-config-model-picker aria-label="Model ID" disabled></select></label>
  <p><strong data-config-model>Loading available models…</strong></p>
  <div data-config-controls></div>
  <button type="button" data-config-reset disabled>Reset defaults</button>
  <p data-config-error role="alert"></p>
  <p data-config-status role="status" aria-live="polite">Loading the exported schema and model defaults…</p>
  <div data-config-output hidden>
    <h2>1. Save config.yaml</h2>
    <div class="config-builder-actions">
      <button type="button" data-config-copy="yaml" data-copy-label="YAML" disabled>Copy YAML</button>
      <button type="button" data-config-download disabled>Download config.yaml</button>
    </div>
    <pre><code data-config-code="yaml"></code></pre>
    <h2>2. Launch the server</h2>
    <p>Run from the directory containing config.yaml in your FastVideo environment.</p>
    <button type="button" data-config-copy="command" data-copy-label="Launch command" disabled>Copy command</button>
    <pre><code data-config-code="command"></code></pre>
  </div>
</div>

The launch command stays short when selections change: the values go into YAML.
