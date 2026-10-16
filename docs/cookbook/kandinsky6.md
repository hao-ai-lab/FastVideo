# Kandinsky 6 recipes

<RecipeFamily family="kandinsky6" />

<details  id="cookbook-setup">
  <summary>Setup</summary>
  <div >
      <p>The generated commands expect a local clone:</p>
      <pre><code>git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo</code></pre>
      <p>Use <a href="../../inference/configuration/">Configuration</a> for supported Python and CLI settings, <a href="../../inference/optimizations/">Optimizations</a> for attention and memory tradeoffs, and the <a href="../../inference/support_matrix/">support matrix</a> for the supported model and optimization surface.</p>
      <p>For model-specific behavior, see <a href="../../inference/kandinsky6/">Kandinsky 6 video with audio</a> and <a href="../../inference/kandinsky6_sr/">Kandinsky 6 video super-resolution</a>.</p>
  </div>
</details>

<details  id="cookbook-troubleshooting">
  <summary>Troubleshooting</summary>
  <div >
      <ul>
        <li>Text and image conditioning use the same TI2VA pipeline and example. Set <code>IMAGE_PATH</code> in the script to condition on an image; both base and distilled outputs include synchronized audio.</li>
        <li>For pi-Flow, set <code>KANDINSKY6_MODEL_PATH</code> to the distilled model ID when running <code>basic_kandinsky6_ti2va.py</code>. Its registered preset uses 10 inference steps, guidance 1.0, <code>eps=1e-6</code>, <code>final_step_size_scale=0.5</code>, and <code>num_policy_substeps=128</code>.</li>
        <li>For VSR, set <code>INPUT_VIDEO</code> in the generated command. The pipeline accepts x2, x2.25 and x4 scales, processes up to 121 frames at 24 fps, and preserves source audio.</li>
        <li>Checkpoint key-layout errors indicate that the checkpoint and FastVideo checkout target different Kandinsky 6 Diffusers revisions.</li>
        <li>Gated or missing checkpoints: run <code>huggingface-cli login</code> and confirm you accepted the model's license on Hugging Face.</li>
      </ul>
  </div>
</details>

<details  id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div >
      <p>All recipes on this page are <strong>Source-backed</strong>: their commands, model IDs, and flags were validated against the checked-in FastVideo sources listed above (static validation). No runtime GPU validation is recorded for these recipes, so GPU model fit, memory use, throughput, and runtime duration are <strong>Unknown</strong> and deliberately not claimed. Runtime buttons show only the GPU counts configured in checked-in sources.</p>
  </div>
</details>
