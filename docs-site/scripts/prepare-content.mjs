/** Prepare the docs transfer and publish the existing cookbook catalogs. */
import { execFileSync } from "node:child_process";
import { cp, mkdir, rm } from "node:fs/promises";

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
