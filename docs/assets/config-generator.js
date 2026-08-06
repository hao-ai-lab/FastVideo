(() => {
  const resizeType = "fastvideo-config-generator:resize";

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    if (!event.data || typeof event.data !== "object" || event.data.type !== resizeType) return;

    const frame = Array.from(document.querySelectorAll("iframe[data-config-generator]"))
      .find((candidate) => candidate.contentWindow === event.source);
    const height = Number(event.data.height);
    if (!frame || !Number.isFinite(height) || height < 320 || height > 20000) return;

    frame.style.height = `${Math.ceil(height)}px`;
  });
})();
