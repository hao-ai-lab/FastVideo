import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const site = new URL("../", import.meta.url);
const routes = JSON.parse(await readFile(new URL(".generated/routes.json", site), "utf8"));
const old = JSON.parse(await readFile(new URL(".generated/old-headings.json", site), "utf8"));

test("every original non-cookbook route and heading anchor survives the transfer", async () => {
  const failures = [];
  for (const [path, headings] of Object.entries(old)) {
    const route = path === "." ? "/" : `/${path}`;
    if (!routes.includes(route)) {
      failures.push(`Missing route: ${route}`);
      continue;
    }
    const html = await readFile(new URL(`dist/${path === "." ? "" : `${path}/`}index.html`, site), "utf8");
    for (const heading of headings) {
      const escaped = heading.id.replaceAll("&", "&amp;").replaceAll('"', "&quot;");
      if (!html.includes(`id="${escaped}"`) && !html.includes(`id='${escaped}'`))
        failures.push(`${route}#${heading.id}`);
    }
  }
  assert.deepEqual(failures, []);
});

test("converted guides render admonitions, preserve content, and publish local images", async () => {
  const quickStart = await readFile(new URL("dist/getting_started/quick_start/index.html", site), "utf8");
  assert.ok(quickStart.includes("Need more control?"));
  assert.ok(!quickStart.includes("!!! tip"));
  const home = await readFile(new URL("dist/index.html", site), "utf8");
  assert.ok(home.includes("Welcome to FastVideo"));
  assert.ok(home.includes("/FastVideo/assets/images/fastwan.png"));
  await readFile(new URL("dist/assets/images/fastwan.png", site));
});

test("Python API and generated examples contain the actual rendered source documentation", async () => {
  const api = await readFile(new URL("dist/api/fastvideo/configs/pipelines/base/index.html", site), "utf8");
  assert.ok(api.includes('id="fastvideo.configs.pipelines.base.PipelineConfig"'));
  assert.ok(api.includes("from_pretrained"));
  assert.ok(api.includes("python-reference"));
  const examples = routes.filter((route) => route.startsWith("/inference/examples/"));
  assert.ok(examples.length > 5);
  const example = await readFile(new URL("dist/inference/examples/basic/index.html", site), "utf8");
  assert.ok(example.includes("VideoGenerator"));
});
