---
hide:
  - toc
---

<div class="adv-tuning-guide-wrap">
  <iframe
    class="adv-tuning-guide-frame"
    src="/config-generator/tuning/"
    title="FastVideo Advanced Tuning Guide"
  ></iframe>
</div>

<script>
window.addEventListener("message", (e) => {
  if (e.data && e.data.type === "adv-tuning-guide-height") {
    const frame = document.querySelector(".adv-tuning-guide-frame");
    if (frame) frame.style.height = e.data.height + "px";
  }
});
</script>
