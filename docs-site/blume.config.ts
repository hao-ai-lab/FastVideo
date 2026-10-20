import { defineConfig } from "blume";
import { readFileSync } from "node:fs";

export default defineConfig({
  title: "FastVideo",
  description: "A unified inference and post-training framework for accelerated video generation.",
  content: { root: ".generated/content" },
  logo: { image: "/icon.svg", text: "FastVideo" },
  theme: { accent: "blue", background: { light: "#ffffff", dark: "#1c1d20" }, mode: "system" },
  search: {
    popular: [
      { label: "Quick start", href: "/getting_started/quick_start", icon: "rocket" },
      { label: "Fast Models", href: "/fast-models", icon: "boxes" },
      { label: "Installation", href: "/getting_started/installation", icon: "download" },
      { label: "Supported models", href: "/inference/support_matrix", icon: "video" },
      { label: "Python API", href: "/api", icon: "code" },
    ],
  },
  navigation: {
    actions: [{ label: "GitHub", href: "https://github.com/hao-ai-lab/FastVideo" }],
    tabs: [
      { label: "Documentation", path: "/" },
      { label: "Cookbook", path: "/cookbook" },
    ],
    sidebar: { items: JSON.parse(readFileSync(new URL("./.generated/navigation.json", import.meta.url), "utf8")) },
  },
  // Keep the production prefix in local builds so asset-path regressions are visible.
  deployment: { site: "https://haoailab.com", base: "/FastVideo" },
  seo: { robots: true, sitemap: true, og: { enabled: false } },
});
