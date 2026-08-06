# Inference Cookbook

Choose a complete recipe maintained in the FastVideo repository. Each command
runs its checked-in source directly, so coupled model, GPU, offload, and
attention settings do not drift into unsupported combinations.

The commands expect a local clone:

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
```

<div class="cookbook-picker" data-cookbook data-recipes="../assets/cookbook-recipes.json">
  <label for="cookbook-recipe"><strong>Recipe</strong></label>
  <select id="cookbook-recipe" data-cookbook-recipe disabled>
    <option>Loading recipes…</option>
  </select>
  <dl>
    <dt>Model</dt>
    <dd data-cookbook-model>Loading…</dd>
    <dt>Source</dt>
    <dd><a data-cookbook-source href="../inference/examples/basic/">Browse maintained examples</a></dd>
  </dl>
  <pre><code class="language-bash" data-cookbook-command>Loading…</code></pre>
  <p class="cookbook-picker__status" role="status" aria-live="polite" data-cookbook-status></p>
  <noscript>
    JavaScript is needed for the recipe picker. Browse the
    <a href="../inference/examples/examples_inference_index/">inference examples</a>
    instead.
  </noscript>
</div>

## Customize a recipe

Start from the checked-in source, then change only the settings your model
supports:

- [Configuration](../inference/configuration.md) covers the Python and CLI
  config surfaces.
- [Optimizations](../inference/optimizations.md) covers attention backends,
  compilation, and memory tradeoffs.
- [Support matrix](../inference/support_matrix.md) lists supported models and
  optimizations.
