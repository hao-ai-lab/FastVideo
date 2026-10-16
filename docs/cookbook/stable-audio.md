# Stable Audio recipes

<RecipeFamily family="stable_audio" />

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
        <li>Loader errors about monolithic checkpoints mean an upstream <code>stabilityai/stable-audio-open-*</code> ID was used; use the FastVideo converted repos from the recipes.</li>
        <li>Duration and step knobs (<code>audio_end_in_s</code>, <code>num_inference_steps</code>) are documented inline in the example source.</li>
      </ul>
  </div>
</details>

<details  id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div >
      <p>The Stable Audio Open 1.0 recipe maps to a checked-in example and a recorded single-GPU B200 run. The Stable Audio Open Small recipe remains <strong>Source-backed</strong> because the implementation PR did not record a full run for that gated checkpoint. Neither recipe claims a minimum VRAM requirement.</p>
  </div>
</details>
