---
hide:
- toc
---

# Inference Cookbook

<div class="cookbook-shell cookbook-catalog" data-cookbook-gallery>
  <header class="cookbook-hero">
    <p class="cookbook-eyebrow">FastVideo inference cookbook</p>
    <h2>Choose by output, then by family.</h2>
    <p class="cookbook-hero__lede">
      Open a family to pick a maintained recipe and a runtime FastVideo
      actually supports. Every command runs a checked-in source, so the model,
      platform, offload, and attention settings stay tied to that example.
      Cards below group video, image, audio, and interactive world models.
      Mode chips on each card are the workloads FastVideo maintains for that
      family, not a promise that every flag works on every runtime.
    </p>
    <a class="cookbook-inline-link" href="../inference/support_matrix/">
      View the full support matrix <span aria-hidden="true">→</span>
    </a>
    <a class="cookbook-inline-link" href="./openai-api/">Run FastH3 with a playground and API <span aria-hidden="true">→</span></a>
  </header>

  <section class="cookbook-section" id="video-models" aria-labelledby="video-models-heading">
    <div class="cookbook-section__heading">
      <h2 id="video-models-heading">Video</h2>
    </div>
    <div class="cookbook-family-grid">
      <a class="cookbook-family-tile cookbook-family-tile--ready cookbook-family-tile--featured" href="./minimax-h3/" aria-label="Open MiniMax H3 recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/minimax.webp" alt="" width="132" height="132" loading="eager">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>MiniMax H3</strong><small>Video + stereo audio</small></span>
            <span class="cookbook-count">8 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2VA</li>
            <li>FL2VA</li>
            <li>Ref2VA</li>
            <li>MLX T2VA</li>
            <li>DGX Spark</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./wan/" aria-label="Open Wan recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/wan-ai.webp" alt="" width="132" height="132" loading="eager">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Wan</strong><small>FastWan CUDA and FastMetal MLX</small></span>
            <span class="cookbook-count">7 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2V</li>
            <li>I2V</li>
            <li>TI2V</li>
            <li>MLX T2V</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./ltx/" aria-label="Open LTX recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/ltx.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>LTX</strong><small>Video with synchronized audio</small></span>
            <span class="cookbook-count">2 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2V</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./hunyuan/" aria-label="Open Hunyuan recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/tencent-hunyuan.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Hunyuan</strong><small>480p and 1080p upscale</small></span>
            <span class="cookbook-count">2 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2V</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./kandinsky5/" aria-label="Open Kandinsky 5 recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/kandinsky.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Kandinsky 5</strong><small>Text and image to video</small></span>
            <span class="cookbook-count">2 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2V</li>
            <li>I2V</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./longcat/" aria-label="Open LongCat recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/meituan-longcat.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>LongCat</strong><small>T2V, I2V, optional refine</small></span>
            <span class="cookbook-count">2 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2V</li>
            <li>I2V</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./turbodiffusion/" aria-label="Open TurboDiffusion recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <span class="cookbook-family-tile__monogram" aria-hidden="true">Turbo</span>
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>TurboDiffusion</strong><small>Accelerated Wan profiles</small></span>
            <span class="cookbook-count">3 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2V</li>
            <li>I2V</li>
          </ul>
        </span>
      </a>
    </div>
  </section>

  <section class="cookbook-section" id="image-models" aria-labelledby="image-models-heading">
    <div class="cookbook-section__heading">
      <h2 id="image-models-heading">Image</h2>
    </div>
    <div class="cookbook-family-grid">
      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./flux/" aria-label="Open FLUX recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/black-forest-labs.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>FLUX</strong><small>FLUX.1 and FLUX.2</small></span>
            <span class="cookbook-count">3 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2I</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./glm-image/" aria-label="Open GLM-Image recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/zai.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>GLM-Image</strong><small>Generate and edit</small></span>
            <span class="cookbook-count">2 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2I</li>
            <li>Edit</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./z-image/" aria-label="Open Z-Image recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/tongyi.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Z-Image</strong><small>Turbo text to image</small></span>
            <span class="cookbook-count">1 recipe</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2I</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./stable-diffusion/" aria-label="Open Stable Diffusion recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/stabilityai.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Stable Diffusion</strong><small>SD 3.5 Medium</small></span>
            <span class="cookbook-count">1 recipe</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2I</li>
          </ul>
        </span>
      </a>
    </div>
  </section>

  <section class="cookbook-section" id="audio-models" aria-labelledby="audio-models-heading">
    <div class="cookbook-section__heading">
      <h2 id="audio-models-heading">Audio</h2>
    </div>
    <div class="cookbook-family-grid">
      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./stable-audio/" aria-label="Open Stable Audio recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/stabilityai.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Stable Audio</strong><small>Open 1.0 and Small</small></span>
            <span class="cookbook-count">2 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2A</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./mmaudio/" aria-label="Open MMAudio recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/fastvideo.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>MMAudio</strong><small>Video or text to audio</small></span>
            <span class="cookbook-count">1 recipe</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>V2A</li>
            <li>T2A</li>
          </ul>
        </span>
      </a>
    </div>
  </section>

  <section class="cookbook-section" id="world-models" aria-labelledby="world-models-heading">
    <div class="cookbook-section__heading">
      <h2 id="world-models-heading">World and interactive</h2>
    </div>
    <div class="cookbook-family-grid">
      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./cosmos/" aria-label="Open Cosmos recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/nvidia.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Cosmos</strong><small>Text to world video</small></span>
            <span class="cookbook-count">1 recipe</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>T2W</li>
          </ul>
        </span>
      </a>

      <a class="cookbook-family-tile cookbook-family-tile--ready" href="./matrix-game/" aria-label="Open Matrix Game recipes">
        <span class="cookbook-family-tile__visual" data-evervault>
          <span class="cookbook-evervault" aria-hidden="true">
            <span class="cookbook-evervault__gradient"></span>
            <span class="cookbook-evervault__noise" data-cookbook-pattern></span>
          </span>
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/fastvideo.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>Matrix Game</strong><small>Image-conditioned worlds</small></span>
            <span class="cookbook-count">2 recipes</span>
          </span>
          <ul class="cookbook-mode-row">
            <li>I2W</li>
          </ul>
        </span>
      </a>
    </div>
  </section>

  <section class="cookbook-section" id="planned-families" aria-labelledby="planned-families-heading">
    <div class="cookbook-section__heading">
      <h2 id="planned-families-heading">Pages still to write</h2>
      <p>These families already have runnable examples. The cookbook page is not ready, so the cards are not links.</p>
    </div>
    <div class="cookbook-family-grid">
      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="GameCraft cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/tencent-hunyuan.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>GameCraft</strong><small>Game world generation</small></span>
            <span class="cookbook-count">Page planned</span>
          </span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="GEN3C cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/nvidia.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>GEN3C</strong><small>Novel-view video</small></span>
            <span class="cookbook-count">Page planned</span>
          </span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="HY-World cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-family-tile__logo-wrap">
            <span class="cookbook-family-tile__monogram" aria-hidden="true">HY</span>
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>HY-World</strong><small>Interactive world play</small></span>
            <span class="cookbook-count">Page planned</span>
          </span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="DreamX cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-family-tile__logo-wrap">
            <img class="off-glb" src="../assets/logos/fastvideo.webp" alt="" width="132" height="132" loading="lazy">
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>DreamX</strong><small>World generation</small></span>
            <span class="cookbook-count">Page planned</span>
          </span>
        </span>
      </article>

      <article class="cookbook-family-tile cookbook-family-tile--coming" aria-label="LingBot cookbook page planned; runnable examples exist">
        <span class="cookbook-family-tile__visual">
          <span class="cookbook-family-tile__logo-wrap">
            <span class="cookbook-family-tile__monogram" aria-hidden="true">LB</span>
          </span>
        </span>
        <span class="cookbook-family-tile__footer">
          <span class="cookbook-family-tile__footer-top">
            <span><strong>LingBot</strong><small>Video and world models</small></span>
            <span class="cookbook-count">Page planned</span>
          </span>
        </span>
      </article>
    </div>
  </section>
</div>

<small class="cookbook-logo-credit">
Catalog marks come from the official model publishers' Hugging Face
organizations; typographic tiles are placeholders, never invented logos. See
<a href="https://github.com/hao-ai-lab/FastVideo/blob/main/docs/assets/logos/SOURCES.md">docs/assets/logos/SOURCES.md</a>.
</small>
