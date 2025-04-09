// Update URL search params when tab is clicked
  document.addEventListener("DOMContentLoaded", function () {
    const tabs = document.querySelectorAll(".sd-tab-label");

    function updateURL(tab) {
      const syncGroup = tab.getAttribute("data-sync-group");
      const syncId = tab.getAttribute("data-sync-id");
      if (syncGroup && syncId) {
          const url = new URL(window.location);
          url.searchParams.set(syncGroup, syncId);
          window.history.replaceState(null, "", url);
      }
    }

    tabs.forEach(tab => {
        tab.addEventListener("click", () => updateURL(tab));
    });
});
