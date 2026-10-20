import { expect, test, type Page } from "@playwright/test";

const route = "/FastVideo/cookbook/minimax-h3/fasth3-8step";
async function settings(page: Page, section: string) {
  await expect(page.locator(".recipe-matrix")).toBeVisible();
  if (section === "Serving" || section === "Hardware") {
    const advanced = page.locator(".matrix-advanced");
    if ((await advanced.getAttribute("open")) === null) await advanced.locator("summary").first().click();
  }
}
async function backend(page: Page, name: string) {
  await page
    .getByRole("group", { name: "Hardware backend", exact: true })
    .getByRole("button", { name, exact: true })
    .click();
}
async function chip(page: Page, group: string, name: string) {
  await page.getByRole("group", { name: group, exact: true }).getByRole("button", { name, exact: true }).click();
}
async function output(page: Page, section: string) {
  await page.getByRole("tablist", { name: "Output format" }).getByRole("tab", { name: section, exact: true }).click();
}

test("edits generate YAML and client instructions, invalid edits clear output, reset restores baseline", async ({
  page,
}) => {
  await page.goto(route);
  await output(page, "config.yaml");
  const yaml = page.getByLabel("Generated YAML", { exact: true });
  await settings(page, "Serving");
  await expect(page.getByLabel("Port", { exact: true })).toHaveValue("8000");
  const baseline = await yaml.innerText();
  await page.getByLabel("Port", { exact: true }).fill("9000");
  await expect(yaml).toContainText('"port": 9000');
  await output(page, "Commands");
  await expect(page.getByLabel("Client command", { exact: true })).toContainText(":9000/");
  await output(page, "config.yaml");
  await settings(page, "Model");
  await chip(page, "Offload VAE to CPU", "Off");
  await expect(yaml).toContainText('"vae": false');
  await expect(yaml).toContainText('"num_gpus": 4');
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download YAML", exact: false }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("config.yaml");
  const { readFile } = await import("node:fs/promises");
  const downloaded = await readFile((await download.path())!, "utf8");
  expect(downloaded).toContain('"port": 9000');
  await settings(page, "Serving");
  await page.getByLabel("Port", { exact: true }).fill("70000");
  await expect(page.getByRole("button", { name: "Download YAML", exact: false })).toBeDisabled();
  await expect(yaml).toHaveCount(0);
  await expect(page.getByLabel("Port", { exact: true })).toHaveAttribute("aria-invalid", "true");
  await page.getByRole("button", { name: "Reset changes", exact: true }).click();
  await expect(yaml).toHaveText(baseline);
});

test("copy buttons use their own generated output", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(route);
  await settings(page, "Serving");
  await page.getByLabel("Port", { exact: true }).fill("9001");
  await output(page, "config.yaml");
  const yaml = await page.getByLabel("Generated YAML", { exact: true }).innerText();
  await page.getByRole("button", { name: "Copy YAML", exact: true }).click();
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(yaml);
  await output(page, "Commands");
  const server = await page.getByLabel("Server command", { exact: true }).innerText();
  const launch = [
    "(",
    "set -e",
    "cat > config.yaml <<'FASTVIDEO_CONFIG'",
    yaml.trimEnd(),
    "FASTVIDEO_CONFIG",
    server,
    ")",
  ].join("\n");
  await page.getByRole("button", { name: "Copy", exact: true }).click();
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(launch);
  await page.locator(".recipe-request-code > summary").click();
  const client = await page.getByLabel("Client command", { exact: true }).innerText();
  expect(client).toContain(":9001/");
  await page.getByRole("button", { name: "Copy client command", exact: true }).click();
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(client);
  expect(new Set([yaml, launch, client]).size).toBe(3);
});

test("deployment changes reset edits and support MLX and streaming", async ({ page }) => {
  await page.goto(route);
  await settings(page, "Serving");
  await page.getByLabel("Port", { exact: true }).fill("9000");
  await backend(page, "MLX");
  await expect(page.getByLabel("Server command", { exact: true })).toContainText("mlx_server");
  await settings(page, "Serving");
  await expect(page.getByLabel("Port", { exact: true })).toHaveValue("8000");
  await page.goto("/FastVideo/cookbook/ltx/ltx2-distilled");
  await expect(page.getByLabel("Client command", { exact: true })).toContainText("/health");
  await page.locator(".recipe-request-code > summary").click();
  await expect(page.getByText(/WebSocket URL:/)).toBeVisible();
  await page.goto("/FastVideo/cookbook/wan/wan21-i2v");
  await expect(page.getByLabel("Client command", { exact: true })).toContainText("input_reference");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Wan2.1 I2V 14B");
});

test("failed loads offer retry without keeping stale commands", async ({ page }) => {
  await page.route("**/recipes/fasth3-8step/cuda-rest.json", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.goto(route);
  await expect(page.getByRole("alert")).toContainText("HTTP 503");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toHaveCount(0);
  await page.unroute("**/recipes/fasth3-8step/cuda-rest.json");
  await page.getByRole("button", { name: "Retry loading recipes" }).click();
  await output(page, "config.yaml");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toBeVisible();
});

test("an obsolete deployment response cannot replace the selected deployment", async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/recipes/fasth3-8step/mlx-rest.json", async (route) => {
    await gate;
    await route.continue().catch(() => {}); // The component may already have aborted the request.
  });
  await page.goto(route);
  await output(page, "config.yaml");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toBeVisible();
  await backend(page, "MLX");
  await backend(page, "CUDA");
  release();
  await output(page, "config.yaml");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toContainText("FastVideo-FastH3-8-Step-V2");
  await expect(
    page.getByRole("group", { name: "Hardware backend" }).getByRole("button", { name: "CUDA", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
});

test("mobile layout stays within the viewport and controls work by keyboard", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(route);
  await settings(page, "Serving");
  const port = page.getByLabel("Port", { exact: true });
  await port.focus();
  await port.press("ControlOrMeta+A");
  await port.press("9");
  await port.press("0");
  await port.press("0");
  await port.press("1");
  await expect(page.getByLabel("Client command", { exact: true })).toContainText(":9001/");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("inherited values, explicit false, zero and null retain their meanings", async ({ page }) => {
  await page.goto("/FastVideo/cookbook?deployment=fastwan21%2Fcuda-rest");
  await output(page, "config.yaml");
  const yaml = page.getByLabel("Generated YAML", { exact: true });
  await expect(yaml).toBeVisible();
  await expect(yaml).not.toContainText('"text_encoder":');
  await settings(page, "Model");
  await chip(page, "Offload text encoder to CPU", "Off");
  await expect(yaml).toContainText('"text_encoder": false');
  await page
    .getByRole("button", {
      name: "Use baseline for Offload text encoder to CPU",
      exact: true,
    })
    .click();
  await expect(yaml).not.toContainText('"text_encoder":');
  await settings(page, "Request");
  await page.getByLabel("Seed", { exact: true }).fill("0");
  await expect(yaml).toContainText('"seed": 0');
  await settings(page, "Serving");
  await page.getByRole("button", { name: "Set Served model name to null", exact: true }).click();
  await expect(yaml).toContainText('"served_model_name": null');
});

test("catalog discovery opens dedicated model recipes and filters by family and task", async ({ page }) => {
  await page.goto("/FastVideo/cookbook");
  const cards = page.locator(".recipe-card");
  await expect(cards).toHaveCount(34);
  await expect(cards.first()).toContainText("FastH3 8-Step V2");
  await page.getByLabel("Model family", { exact: true }).selectOption("Wan");
  await page.getByLabel("Search recipes").fill("Wan2.1 I2V");
  await expect(cards).toHaveCount(1);
  await cards.click();
  await expect(page).toHaveURL(/cookbook\/wan\/wan21-i2v$/);
  await expect(page.getByLabel("Base model")).toHaveCount(0);
  await page.getByRole("link", { name: "← All recipes", exact: true }).click();
  await page.getByLabel("Search recipes").fill("FastWan2.1");
  await page.locator(".recipe-card").click();
  await expect(page).toHaveURL(/cookbook\/wan\/fastwan21$/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("FastWan2.1 1.3B");
});

test("hardware preferences stay distinct from schema-backed GPU count and topology", async ({ page }) => {
  await page.goto(route);
  await page.getByRole("radio", { name: "RTX PRO 6000" }).check();
  await expect(page).toHaveURL(/hardware=rtx/);
  await settings(page, "Hardware");
  await page.getByLabel("Tensor parallelism", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Use automatic parallelism" }).click();
  await expect(page.getByLabel("Tensor parallelism", { exact: true })).toHaveValue("-1");
  await page.getByLabel("GPU count", { exact: true }).fill("1");
  await output(page, "config.yaml");
  const yaml = page.getByLabel("Generated YAML", { exact: true });
  await expect(yaml).toContainText('"num_gpus": 1');
  await expect(yaml).not.toContainText('"hardware"');
  await expect(page.getByText("Unverified recipe", { exact: true })).toBeVisible();
  await page.getByLabel("GPU count", { exact: true }).fill("0");
  await expect(yaml).toHaveCount(0);
  await expect(page.getByLabel("GPU count", { exact: true })).toHaveAttribute("aria-invalid", "true");
});

test("output tabs support keyboard navigation and precision chips use the backend schema", async ({ page }) => {
  await page.goto(route);
  const tabs = page.getByRole("tablist", { name: "Output format" });
  await tabs.getByRole("tab", { name: "Commands", exact: true }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(tabs.getByRole("tab", { name: "config.yaml", exact: true })).toHaveAttribute("aria-selected", "true");
  await backend(page, "MLX");
  await chip(page, "VAE compute precision", "bf16");
  await output(page, "config.yaml");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toContainText('"vae_dtype": "bf16"');
  await expect(
    page.getByRole("group", { name: "Hardware backend" }).getByRole("button", { name: "AMD", exact: true }),
  ).toHaveCount(0);
  await page.goto("/FastVideo/cookbook");
  await page.getByLabel("Search recipes").fill("does not exist");
  await expect(page.getByRole("heading", { name: "No matching recipes" })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(page.locator(".recipe-card")).toHaveCount(34);
});

test("cookbook cards and controls remain readable in dark mode and reflow at 320px", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/FastVideo/cookbook");
  await expect(page.locator(".recipe-card").first()).toBeVisible();
  await page.setViewportSize({ width: 320, height: 900 });
  await expect(page.locator(".recipe-card").first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.goto(route);
  await settings(page, "Serving");
  await expect(page.getByLabel("Port", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await expect(page.getByRole("button", { name: "Download YAML", exact: false })).toBeEnabled();
});

test("image and audio models have logos and dedicated source recipe pages", async ({ page }) => {
  await page.goto("/FastVideo/cookbook");
  await page.getByRole("button", { name: "Image", exact: true }).click();
  await expect(page.locator(".recipe-card")).toHaveCount(4);
  const card = page.locator(".recipe-card").filter({ hasText: "FLUX.2 Klein" });
  await expect(card.locator("img")).toBeVisible();
  await card.click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("FLUX.2 Klein 4B");
  await expect(page.getByLabel("Inference command")).toContainText("flux");
  await expect(
    page.getByRole("group", { name: "Run mode" }).getByRole("button", { name: "Python SDK", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "Serving", exact: true })).toHaveCount(0);
  await page.getByLabel("Seed", { exact: true }).fill("0");
  await expect(page.getByRole("button", { name: "Copy", exact: true })).toBeEnabled();
  await page.goto("/FastVideo/cookbook");
  await page.getByRole("button", { name: "Audio", exact: true }).click();
  await expect(page.locator(".recipe-card")).toHaveCount(3);
  await page.locator(".recipe-card").first().click();
  await page.getByLabel("Audio duration", { exact: true }).fill("1.5");
  await expect(page.getByRole("button", { name: "Copy", exact: true })).toBeEnabled();
});

test("provider labels match FastVideo logos and CompactH3 is replaced by FastH3 Trim", async ({ page }) => {
  await page.goto("/FastVideo/cookbook");
  const ownedCards = page.locator(".recipe-card").filter({ has: page.locator('img[src$="/fastvideo.webp"]') });
  for (const card of await ownedCards.all()) await expect(card.locator(".recipe-card-kicker")).toHaveText("FastVideo");
  await page.getByLabel("Search recipes").fill("FastH3 Trim");
  await expect(page.locator(".recipe-card")).toHaveCount(1);
  await page.locator(".recipe-card").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("FastH3 Trim");
  await expect(page.getByRole("link", { name: "View on Hugging Face ↗", exact: true })).toHaveAttribute(
    "href",
    /FastVideo-FastH3-Trim-8-Step-NVFP4/,
  );
  await expect(page.getByRole("button", { name: "Copy YAML", exact: true })).toBeEnabled();
});

test("FastH3 Trim edits drive inference, serving, and complete copyable launch scripts", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: async (text: string) => {
          (window as Window & { copiedCommand?: string }).copiedCommand = text;
        },
      },
    });
  });
  await page.goto("/FastVideo/cookbook/minimax-h3/fasth3-trim");
  await expect(page.getByRole("radio", { name: "DGX Spark GB10", exact: true })).toBeChecked();
  await expect(page.getByRole("radio", { name: "H100 Data center", exact: true })).toHaveCount(0);
  await page.getByLabel("Prompt", { exact: true }).fill("A potter finishes a blue bowl.");
  await page.getByLabel("Seed", { exact: true }).fill("0");
  await page.getByRole("button", { name: "Copy", exact: true }).click();
  const script = await page.evaluate(() => (window as Window & { copiedCommand?: string }).copiedCommand);
  expect(script).toContain("cat > config.yaml <<'FASTVIDEO_CONFIG'");
  expect(script).toContain("FastVideo-FastH3-Trim-8-Step-NVFP4");
  expect(script).toContain('"seed": 0');
  expect(script).toContain("FASTVIDEO_VSA_TRITON=1");
  expect(script).toContain("fastvideo serve --config config.yaml");
  await page.locator(".recipe-request-code summary").click();
  await expect(page.getByLabel("Client command", { exact: true })).toContainText("A potter finishes a blue bowl.");
  await chip(page, "Run mode", "CLI");
  await expect(page.getByLabel("Inference command", { exact: true })).toContainText("fastvideo generate");
  await output(page, "config.yaml");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toContainText("request:");
  await expect(page.getByLabel("Generated YAML", { exact: true })).not.toContainText("server:");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toContainText("num_inference_steps: 9");
  await chip(page, "Run mode", "Python SDK");
  await output(page, "Commands");
  await expect(page.getByLabel("SDK command", { exact: true })).toContainText("VideoGenerator.from_config");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("source-backed controls update the copied command without changing models", async ({ page }) => {
  await page.goto("/FastVideo/cookbook/minimax-h3/models/minimax-h3-t2v");
  await chip(page, "GPU count", "2");
  await expect(page.getByLabel("Inference command", { exact: true })).toContainText("--num-gpus '2'");
});

test("Wan I2V requires an image for CLI and SDK examples without changing server defaults", async ({ page }) => {
  await page.goto("/FastVideo/cookbook/wan/wan21-i2v");
  await chip(page, "Run mode", "CLI");
  await expect(page.getByRole("alert")).toContainText("Enter a source image path");
  await expect(page.getByRole("button", { name: "Copy", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Download YAML", exact: true })).toBeDisabled();
  await page.getByLabel("Source image path", { exact: true }).fill("/data/first-frame.png");
  await output(page, "config.yaml");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toContainText("image_path: /data/first-frame.png");
  await expect(page.getByRole("button", { name: "Copy", exact: true })).toBeEnabled();
  await chip(page, "Run mode", "Python SDK");
  await expect(page.getByLabel("Generated YAML", { exact: true })).toContainText("image_path: /data/first-frame.png");
  await page.getByLabel("Source image path", { exact: true }).fill("");
  await expect(page.getByRole("button", { name: "Copy YAML", exact: true })).toBeDisabled();
  await expect(page.getByLabel("Generated YAML", { exact: true })).toHaveCount(0);
  await chip(page, "Run mode", "Serving");
  await expect(page.getByRole("button", { name: "Copy", exact: true })).toBeEnabled();
  await expect(page.getByLabel("Generated YAML", { exact: true })).not.toContainText("image_path");
});

test("source editors retain required model flags and preparation commands", async ({ page }) => {
  await page.goto("/FastVideo/cookbook/wan/models/fastmetal-1-3b-mlx");
  await expect(page.getByLabel("Height", { exact: true })).toHaveValue("480");
  await expect(page.getByLabel("Inference command", { exact: true })).toContainText(
    "hf download FastVideo/FastMetal-1.3B-QAD",
  );
  await expect(page.getByLabel("Inference command", { exact: true })).toContainText(
    "--model-root ./FastMetal-1.3B-QAD",
  );
  await page.getByLabel("Seed", { exact: true }).fill("0");
  await expect(page.getByLabel("Inference command", { exact: true })).toContainText("--seed '0'");
  await page.getByRole("tab", { name: "Python example", exact: true }).click();
  await expect(page.getByLabel("Source example", { exact: true })).toContainText("default=480");
});

test("source-backed serving pages expose their actual server settings", async ({ page }) => {
  await page.goto("/FastVideo/cookbook/wan/models/wan22-t2v");
  await chip(page, "Run mode", "Serving");
  await page.locator(".matrix-advanced > summary").click();
  await page.getByLabel("Port", { exact: true }).fill("9005");
  await page.getByRole("button", { name: "cURL", exact: true }).click();
  await expect(page.getByLabel("Client command", { exact: true })).toContainText(":9005/v1/videos/sync");
  await page.getByRole("tab", { name: "config.yaml", exact: true }).click();
  await expect(page.getByLabel("Source example", { exact: true })).toContainText("port: 9005");
  await expect(page.getByLabel("Source example", { exact: true })).toContainText("Wan2.2-T2V-A14B");
  await expect(page.getByRole("button", { name: "Copy", exact: true })).toBeEnabled();
});

test("source model hardware preferences survive hydration and reload", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/FastVideo/cookbook/flux/models/flux2-klein-t2i?hardware=custom%3ARTX%204090");
  await expect(page.getByRole("radio", { name: "RTX 4090 Custom device", exact: true })).toBeChecked();
  await page.reload();
  await expect(page.getByRole("radio", { name: "RTX 4090 Custom device", exact: true })).toBeChecked();
  expect(errors).toEqual([]);
});

test("custom devices persist and component previews cannot silently generate incompatible YAML", async ({ page }) => {
  await page.goto(route);
  await page.getByRole("button", { name: "+ Find or add a GPU", exact: true }).click();
  await page.getByLabel("Device name", { exact: true }).fill("RTX 4090");
  await page.getByRole("button", { name: "Add device", exact: true }).click();
  await expect(page.getByRole("radio", { name: "RTX 4090 Custom device", exact: true })).toBeChecked();
  await page.reload();
  await expect(page.getByRole("radio", { name: "RTX 4090 Custom device", exact: true })).toBeChecked();
  await page.locator(".component-plan > summary").click();
  await chip(page, "Text encoder", "NVFP4 encoder");
  await expect(page.getByRole("button", { name: "Download YAML", exact: true })).toBeDisabled();
  await expect(page.getByRole("alert")).toContainText("compatible serving recipe");
  await page.getByRole("button", { name: "Restore recipe components", exact: true }).click();
  await expect(page.getByRole("button", { name: "Download YAML", exact: true })).toBeEnabled();
});
