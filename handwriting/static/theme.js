"use strict";
// Runs before first paint so a dark theme never flashes light.
(function () {
  const root = document.documentElement;
  let stored = null;
  try { stored = localStorage.getItem("hs_theme"); } catch { /* storage can be blocked */ }
  const dark = stored ? stored === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.classList.toggle("dark", dark);
  document.addEventListener("click", event => {
    if (!event.target.closest("[data-theme-toggle]")) return;
    const next = !root.classList.contains("dark");
    root.classList.toggle("dark", next);
    try { localStorage.setItem("hs_theme", next ? "dark" : "light"); } catch { /* ignore */ }
  });
})();
