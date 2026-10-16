# MMAudio recipes

<RecipeFamily family="mmaudio" />

<details  id="cookbook-setup">
  <summary>Setup</summary>
  <div >
      <p>The generated commands expect a local clone:</p>
      <pre><code>git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo</code></pre>
      <p>Use <a href="../../inference/configuration/">Configuration</a> for supported Python and CLI settings, <a href="../../inference/optimizations/">Optimizations</a> for attention and memory tradeoffs, and the <a href="../../inference/support_matrix/">support matrix</a> for the supported model and optimization surface.</p>
  </div>
</details>

<details  id="cookbook-troubleshooting">
  <summary>Troubleshooting</summary>
  <div >
      <ul>
        <li>If the example reports a missing local <code>converted_weights/mmaudio/large_44k_v2</code> path, the <code>MMAUDIO_MODEL_PATH</code> env var from the cookbook command was not applied; export it in the same shell.</li>
        <li>Alternatively convert upstream weights yourself with <code>scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py</code> and point the env var at the result.</li>
      </ul>
  </div>
</details>

<details  id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div >
      <p>All recipes on this page are <strong>Source-backed</strong>: their commands, model IDs, and flags were validated against the checked-in FastVideo sources listed above (static validation). No runtime GPU validation is recorded for these recipes, so GPU model fit, memory use, throughput, and runtime duration are <strong>Unknown</strong> and deliberately not claimed. Runtime buttons show only the GPU counts configured in checked-in sources.</p>
  </div>
</details>
