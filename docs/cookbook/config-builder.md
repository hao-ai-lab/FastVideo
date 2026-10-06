# Serving configuration builder

<!-- TODO(cookbook-ui): Replace this temporary demo page and inline styling with the final cookbook UI. -->

Choose execution settings and request defaults for Wan2.2 TI2V 5B text-to-video.
Defaults and declared limits come from a standard JSON Schema exported from
FastVideo's Python schema and model preset. Ajv validates the completed configuration.
The page generates a configuration file; it does not launch inference or test GPU capacity.

<style>
.config-builder-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
.config-builder-card { padding: 1rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: .4rem; }
.config-builder-card h2 { margin-top: 0; }
[data-config-group] { display: grid; gap: .8rem; }
.config-builder-field { display: flex; flex-direction: column; gap: .3rem; }
.config-builder-field input:not([type=checkbox]), .config-builder-field select { padding: .45rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: .2rem; }
.config-builder-field input[type=checkbox] { align-self: flex-start; }
.config-builder-field small { color: var(--md-default-fg-color--light); }
.config-builder-actions { display: flex; flex-wrap: wrap; gap: .5rem; margin: .6rem 0; }
[data-config-builder] button { padding: .4rem .7rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: .2rem; }
[data-config-builder] button:disabled { opacity: .5; }
[data-config-builder] pre { max-height: 32rem; overflow: auto; }
[data-config-error] { color: var(--md-typeset-a-color); }
</style>

<div data-config-builder data-metadata="../../assets/cookbook-config.json">
  <p><strong data-config-model>Loading model defaults…</strong></p>
  <div class="config-builder-cards">
    <section class="config-builder-card" aria-label="Execution settings">
      <h2>Execution</h2>
      <div data-config-group="execution"></div>
    </section>
    <section class="config-builder-card" aria-label="Request defaults">
      <h2>Request defaults</h2>
      <p>Individual API requests can override these values.</p>
      <div data-config-group="request"></div>
    </section>
  </div>
  <details>
    <summary>Advanced request and server settings</summary>
    <div data-config-group="advanced"></div>
  </details>
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
    <details>
      <summary>Optional readiness check and test request</summary>
      <button type="button" data-config-copy="healthCommand" data-copy-label="Readiness command" disabled>Copy readiness check</button>
      <pre><code data-config-code="healthCommand"></code></pre>
      <p>The prompt-only request uses the defaults saved in config.yaml. It writes the returned video to output.mp4.</p>
      <button type="button" data-config-copy="clientCommand" data-copy-label="Test request" disabled>Copy test request</button>
      <pre><code data-config-code="clientCommand"></code></pre>
    </details>
  </div>
</div>

The launch command stays short when selections change: the values go into YAML.
