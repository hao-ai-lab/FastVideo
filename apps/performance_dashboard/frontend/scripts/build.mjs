import * as esbuild from "esbuild";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = resolve(root, "dist");
const assets = resolve(dist, "assets");

await mkdir(assets, { recursive: true });

await esbuild.build({
  entryPoints: [resolve(root, "src/main.tsx")],
  bundle: true,
  format: "esm",
  minify: true,
  sourcemap: true,
  target: ["es2020"],
  outfile: resolve(assets, "dashboard.js"),
  loader: {
    ".svg": "file"
  }
});

await writeFile(
  resolve(dist, "index.html"),
  `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FastVideo Performance Dashboard</title>
    <link rel="stylesheet" href="/assets/dashboard.css" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/assets/dashboard.js"></script>
  </body>
</html>
`
);
