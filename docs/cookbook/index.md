---
hide:
- toc
---

# Inference Cookbook

<div class="cookbook-shell cookbook-catalog" data-cookbook-gallery>
  <header class="cookbook-hero">
    <p class="cookbook-eyebrow">FastVideo inference cookbook</p>
    <h2>Choose a model family.</h2>
    <p class="cookbook-hero__lede">
      Open a family to pick a maintained recipe and a runtime FastVideo
      actually supports. Every command runs a checked-in source, so the model,
      platform, offload, and attention settings stay tied to that example. The catalog
      is derived from the model families registered in
      <code>fastvideo/registry.py</code>.
    </p>
    <a class="cookbook-inline-link" href="../inference/support_matrix/">
      View the full support matrix <span aria-hidden="true">→</span>
    </a>
  </header>

  <section class="cookbook-section" id="model-families" aria-label="Model families">
    <div class="cookbook-family-grid">
      <a class="cookbook-family-tile cookbook-family-tile--ready cookbook-family-tile--featured" href="./minimax-h3/" aria-label="Open MiniMax H3 recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/minimax.webp" alt="" width="132" height="132" loading="eager">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>MiniMax H3</strong><small>Video + audio · CUDA + MLX</small></span>
          <span class="cookbook-count">6 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./wan/" aria-label="Open Wan recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/wan-ai.webp" alt="" width="132" height="132" loading="eager">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Wan</strong><small>Video generation · CUDA + MLX</small></span>
          <span class="cookbook-count">7 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./ltx/" aria-label="Open LTX recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/ltx.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>LTX</strong><small>Video and audio generation</small></span>
          <span class="cookbook-count">2 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./hunyuan/" aria-label="Open Hunyuan recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/tencent-hunyuan.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Hunyuan</strong><small>Video generation</small></span>
          <span class="cookbook-count">2 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./cosmos/" aria-label="Open Cosmos recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/nvidia.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Cosmos</strong><small>World and video generation</small></span>
          <span class="cookbook-count">1 recipe</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./kandinsky5/" aria-label="Open Kandinsky 5 recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/kandinsky.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Kandinsky 5</strong><small>Text and image to video</small></span>
          <span class="cookbook-count">2 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./flux/" aria-label="Open FLUX recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/black-forest-labs.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>FLUX</strong><small>Image generation</small></span>
          <span class="cookbook-count">3 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./glm-image/" aria-label="Open GLM-Image recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/zai.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>GLM-Image</strong><small>Image generation and editing</small></span>
          <span class="cookbook-count">2 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./z-image/" aria-label="Open Z-Image recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/tongyi.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Z-Image</strong><small>Image generation</small></span>
          <span class="cookbook-count">1 recipe</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./stable-diffusion/" aria-label="Open Stable Diffusion recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/stabilityai.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Stable Diffusion</strong><small>Image generation</small></span>
          <span class="cookbook-count">1 recipe</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./longcat/" aria-label="Open LongCat recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/meituan-longcat.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>LongCat</strong><small>Video generation</small></span>
          <span class="cookbook-count">2 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./stable-audio/" aria-label="Open Stable Audio recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/stabilityai.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Stable Audio</strong><small>Audio generation</small></span>
          <span class="cookbook-count">2 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./mmaudio/" aria-label="Open MMAudio recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/fastvideo.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>MMAudio</strong><small>Audio generation</small></span>
          <span class="cookbook-count">1 recipe</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./matrix-game/" aria-label="Open Matrix Game recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/fastvideo.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>Matrix Game</strong><small>Interactive world generation</small></span>
          <span class="cookbook-count">2 recipes</span>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./turbodiffusion/" aria-label="Open TurboDiffusion recipes">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <span class="cookbook-family-tile__monogram" aria-hidden="true">Turbo</span>
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>TurboDiffusion</strong><small>Accelerated Wan profiles</small></span>
          <span class="cookbook-count">3 recipes</span>
        </span>
      </a>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="GameCraft cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/tencent-hunyuan.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>GameCraft</strong><small>Game world generation</small></span>
          <span class="cookbook-count">Page planned</span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="GEN3C cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/nvidia.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>GEN3C</strong><small>Novel-view video</small></span>
          <span class="cookbook-count">Page planned</span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="HY-World cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <span class="cookbook-family-tile__monogram" aria-hidden="true">HY</span>
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>HY-World</strong><small>Interactive world play</small></span>
          <span class="cookbook-count">Page planned</span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="DreamX cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/fastvideo.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>DreamX</strong><small>World generation</small></span>
          <span class="cookbook-count">Page planned</span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="LingBot cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-card-pattern" data-cookbook-pattern aria-hidden="true"></span>
          <span class="cookbook-family-tile__logo-wrap">
            <span class="cookbook-family-tile__monogram" aria-hidden="true">LB</span>
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span><strong>LingBot</strong><small>Video and world models</small></span>
          <span class="cookbook-count">Page planned</span>
        </span>
      </article>
    </div>
  </section>

  <section class="cookbook-roadmap" aria-labelledby="roadmap-heading">
    <h2 id="roadmap-heading">Inference first, then the full workflow</h2>
    <p>
      Inference is the first complete stage. Distillation, fine-tuning,
      training, evaluation, optimization, and deployment will reuse the same
      family-first structure as their recipes land. Each family page shows
      which stages are available and which are planned.
    </p>
  </section>
</div>

<small class="cookbook-logo-credit">
Catalog marks come from the official model publishers' Hugging Face
organizations; typographic tiles are placeholders, never invented logos. See
<a href="https://github.com/hao-ai-lab/FastVideo/blob/main/docs/assets/logos/SOURCES.md">docs/assets/logos/SOURCES.md</a>.
</small>
