# Serving configuration builder

<!-- TODO(cookbook-ui): Replace this temporary demo page and inline styling with the final cookbook UI. -->

Choose a model and one of its maintained deployments, then adjust its reviewed controls.
The downloaded YAML preserves **every explicit baseline setting**, including settings hidden from this form.
Omitted fields are shown as inherited; the editor does not pretend to resolve runtime defaults.
Reset restores the baseline. Browser validation checks declared field constraints.
This demo does not start a server or load model weights.

<style>
.config-builder-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; padding: 1rem; }
.config-builder-selectors { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
.config-builder-section { margin: .5rem 0; }
.config-builder-field { display: flex; flex-direction: column; gap: .3rem; }
.config-builder-field input:not([type=checkbox]), .config-builder-field select, .config-builder-field textarea { padding: .45rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: .2rem; width: 100%; }
.config-builder-field input[type=checkbox] { align-self: flex-start; }
.config-builder-field small { color: var(--md-default-fg-color--light); overflow-wrap: anywhere; }
.config-builder-actions { display: flex; flex-wrap: wrap; gap: .5rem; margin: .6rem 0; }
.config-builder-links { display: flex; flex-wrap: wrap; gap: 1rem; }
[data-config-builder] button { padding: .4rem .7rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: .2rem; }
[data-config-builder] button:disabled { opacity: .5; }
[data-config-builder] pre { max-height: 32rem; overflow: auto; }
.config-builder-hardware-table { overflow-x: auto; }
[data-config-error] { color: var(--md-typeset-a-color); }
</style>

<div data-config-builder data-metadata="../../assets/cookbook-config/index.json">
  <div class="config-builder-selectors">
    <label class="config-builder-field"><strong>Model</strong><select data-config-model-picker aria-label="Model" disabled></select></label>
    <label class="config-builder-field"><strong>Deployment</strong><select data-config-deployment-picker aria-label="Deployment" disabled></select></label>
  </div>
  <p><strong data-config-model>Loading available models…</strong></p>
  <p data-config-summary></p>
  <p class="config-builder-links"><a data-config-source>View the native recipe configuration</a> <a data-config-guide hidden>Read serving guide</a></p>
  <section data-config-hardware hidden>
    <h2>Hardware</h2>
    <p data-config-topology></p>
    <p data-config-hardware-state aria-live="polite"></p>
    <div class="config-builder-hardware-table" data-config-hardware-table>
      <table>
        <thead><tr><th scope="col">GPU</th><th scope="col">Rated GPU memory</th><th scope="col">Baseline status</th><th scope="col">Baseline record</th></tr></thead>
        <tbody data-config-hardware-rows></tbody>
      </table>
    </div>
    <p data-config-hardware-note>Rated memory is manufacturer capacity, not required or available memory. Unverified means no serving test is recorded for this deployment and GPU.</p>
  </section>
  <div data-config-controls></div>
  <button type="button" data-config-reset disabled>Reset recipe</button>
  <p data-config-error role="alert"></p>
  <p data-config-status role="status" aria-live="polite">Loading maintained model deployments…</p>
  <div data-config-output hidden>
    <h2>1. Prepare the environment</h2>
    <p>Follow the <a data-config-install>installation guide</a>, then apply these deployment requirements.</p>
    <ul data-config-requirements></ul>
    <h2>2. Save config.yaml</h2>
    <div class="config-builder-actions">
      <button type="button" data-config-copy="yaml" data-copy-label="YAML" disabled>Copy YAML</button>
      <button type="button" data-config-download disabled>Download config.yaml</button>
    </div>
    <pre><code data-config-code="yaml"></code></pre>
    <h2>3. Launch the server</h2>
    <p>Run from the directory containing config.yaml in your FastVideo environment.</p>
    <button type="button" data-config-copy="command" data-copy-label="Launch command" disabled>Copy command</button>
    <pre><code data-config-code="command"></code></pre>
    <h2 data-config-client-title>4. Send a sample request</h2>
    <p data-config-client-note></p>
    <button type="button" data-config-copy="clientCommand" data-copy-label="Sample request" disabled>Copy sample request</button>
    <pre><code data-config-code="clientCommand"></code></pre>
    <div data-config-streaming-client hidden>
      <p>WebSocket endpoint: <code data-config-websocket-url></code></p>
      <p><a data-config-client-guide>Read the streaming client contract</a></p>
    </div>
  </div>
  <p data-config-request-note hidden>Request defaults apply when a client omits those values. Explicit client request values take precedence.</p>
</div>
