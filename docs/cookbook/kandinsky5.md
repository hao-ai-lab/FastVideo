# Kandinsky 5 recipes

<RecipeFamily family="kandinsky5" />

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
        <li>Alternative Lite/Pro checkpoints are listed inline in the maintained examples; swap the model string only after checking its Hugging Face card.</li>
        <li>Gated or missing checkpoints: run <code>huggingface-cli login</code> and confirm you accepted the model's license on Hugging Face.</li>
      </ul>
  </div>
</details>

<details  id="cookbook-evidence">
  <summary>Evidence status</summary>
  <div >
      <p>Both recipes map to checked-in FastVideo examples and recorded single-GPU B200 runs. The image-to-video run also records 10,365.89 MB peak GPU memory. These measurements describe the recorded runs; they are not minimum hardware requirements.</p>
  </div>
</details>

## Sequence parallel inference

For dense inference on two GPUs, set:

    generator = VideoGenerator.from_pretrained(
        "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
        num_gpus=2, sp_size=2, tp_size=1, use_fsdp_inference=False,
    )
