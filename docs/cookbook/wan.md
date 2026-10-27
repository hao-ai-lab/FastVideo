# Wan recipes

<RecipeFamily family="wan" />

<details >
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
    <div >
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
        <li>Out of memory on the A14B recipes: the checked-in sources already enable CPU offload; see <a href="../../inference/configuration/">Configuration</a> for the offload surface before reducing resolution or frames.</li>
        <li>The FastWan2.1 recipe requires <code>VIDEO_SPARSE_ATTN</code>; confirm the environment variable in the command was set in the same shell.</li>
        <li>FastMetal MLX: install with the <a href="../../getting_started/installation/mlx/">MLX install guide</a>, then pick a FastMetal recipe in the builder. CUDA FastWan-QAD checkpoints are refused on the MLX runtime.</li>
        <li>FastMetal 5B uses <code>mlx_wan22_generate.py</code>. 1.3B and 14B use <code>mlx_wan_prompt_to_video.py</code>.</li>
        <li>Gated or missing checkpoints: run <code>huggingface-cli login</code> and confirm you accepted the model's license on Hugging Face.</li>
      </ul>
  </div>
</details>

<details  id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div >
      <p>Every recipe on this page maps to a checked-in FastVideo source. The FastMetal MLX releases include the recorded M4 Max system memory, documented unified-memory floor, and measured peak MLX memory. CUDA entries remain <strong>Source-backed</strong> where the examples record a GPU count but no exact GPU model or VRAM. Unlisted hardware is unknown, not unsupported.</p>
  </div>
</details>
