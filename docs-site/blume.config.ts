import { defineConfig } from "blume";

export default defineConfig({
  title: "FastVideo",
  description: "Deploy accelerated video generation with the FastVideo cookbook.",
  content: { root: "content" },
  logo: { image: "/icon.svg", text: "FastVideo" },
  theme: { accent: "teal", mode: "system" },
  banner: "Cookbook preview. The full documentation remains at haoailab.com/FastVideo.",
  navigation: {
    featured: [{ label: "Full documentation", href: "https://haoailab.com/FastVideo/" }],
    tabs: [{ label: "Cookbook", path: "/cookbook" }],
  },
  // Keep the production prefix in local builds so asset-path regressions are visible.
  deployment: { site: "https://haoailab.com", base: "/FastVideo" },
  seo: { robots: false, sitemap: false, og: { enabled: false } },
});
