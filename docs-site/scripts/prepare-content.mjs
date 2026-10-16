import { sourceRecipes } from "../lib/cookbook-pages.ts";
/** Prepare the docs transfer and publish the existing cookbook catalogs. */
import { execFileSync } from "node:child_process";
import { cp, mkdir, rm, readFile, writeFile } from "node:fs/promises";

import { transferDocs } from "./transfer-docs.mjs";

await transferDocs();

const repo = new URL("../../", import.meta.url);
execFileSync("npm", ["run", "build:catalog", "--prefix", "docs"], {
  cwd: repo,
  stdio: "inherit",
});
const publicDir = new URL("../public/", import.meta.url);
await mkdir(publicDir, { recursive: true });
const destination = new URL("assets/cookbook-config/", publicDir);
await rm(destination, { recursive: true, force: true });
await cp(new URL("docs/assets/cookbook-config/", repo), destination, {
  recursive: true,
});
await cp(new URL("docs/assets/logos/icon_simple.svg", repo), new URL("icon.svg", publicDir));

await cp(new URL("../assets/hao-ai-lab.png", import.meta.url), new URL("assets/hao-ai-lab.png", publicDir));

// Preserve family recipe selectors during the cookbook redesign, inside a scoped stylesheet.
const legacyCss = await readFile(new URL("docs/assets/custom.css", repo), "utf8");
const tokens = `.legacy-cookbook { --md-default-fg-color: var(--blume-foreground); --md-default-fg-color--light: var(--blume-muted-foreground); --md-default-fg-color--lightest: var(--blume-border); --md-default-bg-color: var(--blume-background); --md-accent-fg-color: var(--blume-accent); --md-code-bg-color: var(--blume-muted); --md-code-fg-color: var(--blume-foreground); }`;
await writeFile(
  new URL("assets/cookbook-runbooks.css", publicDir),
  `${tokens}\n@scope (.legacy-cookbook) {\n${legacyCss.replaceAll(".md-typeset", ":scope")}\n}`,
);

// Publish exact checked-in YAML for inference-only model pages. No settings are synthesized.
const sourceDir = new URL("assets/cookbook-sources/", publicDir);
await rm(sourceDir, { recursive: true, force: true });
await mkdir(sourceDir, { recursive: true });
for (const recipe of sourceRecipes) {
  if (/\.ya?ml$/.test(recipe.source))
    await cp(new URL(recipe.source, repo), new URL(`${recipe.id}-inference.yaml`, sourceDir));
  if (recipe.serving?.source)
    await cp(new URL(recipe.serving.source, repo), new URL(`${recipe.id}-serving.yaml`, sourceDir));
}
