# MiniMax H3 recipes

<RecipeFamily family="minimax_h3" />

<details >
    <summary>Compare H3 modes and options</summary>
    <h2 id="h3-modes-heading">Supported modes</h2>
    <p>
      CUDA covers T2VA, FL2VA, and Ref2VA on the full checkpoint, plus FastH3
      V1 and FastH3 V2, plus CompactH3 NVFP4 on one Blackwell GPU. FastH3 V1 also has a DGX Spark runtime with
      a 1-Spark or 2-Spark device row. MLX is T2VA only: V1 and V2.
      Temporal <code>--fast</code>, spatial <code>--fast-spatial</code>, and opt-in VSA are flags on the same
      MLX script, not extra recipes.
    </p>
    <div >
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
            <td>Full H3, FastH3 V1, FastH3 LoRA, FastH3 V2, CompactH3 NVFP4</td>
            <td>FastH3 V1 or FastH3 V2 after a local DiT conversion</td>
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
            <td>FastH3 V1 on one GB10, or two Sparks with Ray sequence parallel (<code>sp_size=2</code>) over QSFP RoCE. Select NVIDIA DGX Spark, then 1 Spark or 2 Sparks.</td>
            <td>Not wired</td>
          </tr>
          <tr>
            <td>CompactH3 NVFP4</td>
            <td>42-block 20B checkpoint on one RTX 5090 (32 GB, sequential encoder offload) or one RTX PRO 6000 Blackwell (96 GB, encoder+DiT+VAE resident). SageAttention3 FP4, packed NVFP4 DiT, Comfy int8-convrot VAE.</td>
            <td>Not wired</td>
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
      <p >CUDA</p>
      <pre><code>UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"</code></pre>
      <p >Apple Silicon</p>
      <pre><code>uv pip install -e ".[mlx]"</code></pre>
      <p>Follow the <a href="../../getting_started/installation/mlx/">MLX install guide</a> for the extra, <code>ffmpeg</code>, and a clone. Then pick FastH3 V1 or V2 in the builder above.</p>
      <p >NVIDIA DGX Spark</p>
      <pre><code>UV_TORCH_BACKEND=cu130 uv pip install -e .</code></pre>
      <p>Follow the <a href="../../getting_started/installation/spark/">DGX Spark install guide</a> for ARM64 CUDA 13. One Spark is a local process. Two Sparks need Ray on the QSFP link:</p>
      <pre><code>uv pip install ray</code></pre>
      <p>Bring up the cluster from <a href="../../getting_started/installation/spark_pair/">pairing two Sparks</a> before selecting 2 Sparks in the builder.</p>
  </div>
</details>

<details  id="cookbook-troubleshooting">
  <summary>Troubleshooting</summary>
  <div >
      <ul>
        <li>The full CUDA H3 examples request four GPUs by default. Their sources do not claim a GPU model or memory minimum.</li>
        <li>The FastH3 CUDA performance profile was measured on four GB200 GPUs. Use its strict profile when exact operation order matters more than the measured performance configuration.</li>
        <li>The MLX source runtime supports T2VA, optional temporal <code>--fast</code>, optional spatial <code>--fast-spatial</code>, and opt-in VSA on <code>--include-vsa</code> checkpoints. FastH3 V2 MLX converts with <code>--include-vsa</code> and runs eight forwards. FL2VA, Ref2VA, and two-pass refinement are not wired.</li>
        <li>GPU count and VAE decode backend are configurable in the builder above for FastH3 CUDA recipes. Only the value shown by default has a recorded run; other supported values are unmeasured here.</li>
        <li>DGX Spark is a runtime on FastH3 V1, not a separate family card. Select NVIDIA DGX Spark, then 1 Spark or 2 Sparks. The CUDA GPU-count knob does not apply to Spark.</li>
        <li>CompactH3 NVFP4 is one Blackwell GPU. RTX 5090 (32 GB) parks the encoder in pinned host RAM. RTX PRO 6000 Blackwell (96 GB) keeps encoder, DiT, and VAE resident. Keep <code>FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER</code>, <code>FASTVIDEO_FA4=0</code>, <code>FASTVIDEO_VSA_SM100A=0</code>, and <code>FLASHINFER_CUDA_ARCH_LIST=12.0a</code>. On PRO 6000 enable VAE compile and leave DiT <code>inference_torch_compile</code> off. CompactH3 is a dense prune; do not enable <code>VIDEO_SPARSE_ATTN_H3</code> until a VSA-trained student exists.</li>
        <li>GB10 has no FA4 / sm_100a VSA kernel. Keep <code>FASTVIDEO_FA4=0</code> and <code>FASTVIDEO_VSA_SM100A=0</code>. Legal <code>num_frames</code> values are <code>17n+5</code>, capped at 362 (15.08 s). A 345-frame request on one Spark can OOM. Native 16:9 sizes include 832×480 and 1344×768.</li>
        <li>Gated or missing checkpoints: run <code>huggingface-cli login</code> and confirm you accepted the model's license on Hugging Face.</li>
      </ul>
  </div>
</details>

<details  id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div >
      <p>Every command, model ID, and flag on this page maps to a checked-in FastVideo source. Recipes marked <strong>Verified</strong> also have a recorded hardware path in linked FastVideo evidence. The full H3 CUDA examples remain <strong>Source-backed</strong> where the source records a GPU count but no GPU model or memory requirement. Unlisted hardware is unknown, not unsupported.</p>
  </div>
</details>
