import { expect, test } from "@playwright/test";

const route = "/FastVideo/cookbook?deployment=fasth3-8step%2Fcuda-rest";

test("edits generate YAML and client instructions, invalid edits clear output, reset restores baseline", async ({
  page,
}) => {
  await page.goto(route);
  const yaml = page.getByLabel("Generated YAML", { exact: true });
  await expect(page.getByLabel("Port", { exact: true })).toHaveValue("8000");
  const baseline = await yaml.innerText();
  await page.getByLabel("Port", { exact: true }).fill("9000");
  await expect(yaml).toContainText('"port": 9000');
  await expect(page.getByLabel("Client command", { exact: true })).toContainText(":9000/");
  await page.getByLabel("Offload VAE to CPU", { exact: true }).selectOption("false");
  await expect(yaml).toContainText('"vae": false');
  await expect(yaml).toContainText('"num_gpus": 4');
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download YAML", exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("config.yaml");
  const { readFile } = await import("node:fs/promises");
  const downloaded = await readFile((await download.path())!, "utf8");
  expect(downloaded).toContain('"port": 9000');
  await page.getByLabel("Port", { exact: true }).fill("70000");
  await expect(page.getByRole("button", { name: "Download YAML", exact: true })).toBeDisabled();
  await expect(yaml).toHaveCount(0);
  await expect(page.getByLabel("Port", { exact: true })).toHaveAttribute("aria-invalid", "true");
  await page.getByRole("button", { name: "Reset changes", exact: true }).click();
  await expect(yaml).toHaveText(baseline);
});

test("deployment changes reset edits and support MLX and streaming", async ({ page }) => {
  await page.goto(route);
  await page.getByLabel("Port", { exact: true }).fill("9000");
  const selector = page.getByLabel("Model and deployment");
  await selector.selectOption("fasth3-8step/mlx-rest");
  await expect(page.getByLabel("Server command", { exact: true })).toContainText("mlx_server");
  await expect(page.getByLabel("Port", { exact: true })).toHaveValue("8000");
  await selector.selectOption("ltx2-distilled/cuda-streaming");
  await expect(page.getByLabel("Client command", { exact: true })).toContainText("/health");
  await expect(page.getByText(/WebSocket URL:/)).toBeVisible();
  await selector.selectOption("wan21-i2v/cuda-rest");
  await expect(page.getByLabel("Client command", { exact: true })).toContainText("input_reference");
  await expect(page).toHaveURL(/deployment=wan21-i2v%2Fcuda-rest/);
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
  await expect(page.getByLabel("Generated YAML", { exact: true })).toBeVisible();
  const selector = page.getByLabel("Model and deployment");
  await selector.selectOption("fasth3-8step/mlx-rest");
  await selector.selectOption("fastwan21/cuda-rest");
  release();
  await expect(page.getByLabel("Generated YAML", { exact: true })).toContainText("FastWan2.1");
  await expect(selector).toHaveValue("fastwan21/cuda-rest");
});

test("mobile layout stays within the viewport and controls work by keyboard", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(route);
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
  const yaml = page.getByLabel("Generated YAML", { exact: true });
  await expect(yaml).toBeVisible();
  await expect(yaml).not.toContainText('"text_encoder":');
  await page.getByLabel("Offload text encoder to CPU", { exact: true }).selectOption("false");
  await expect(yaml).toContainText('"text_encoder": false');
  await page
    .getByRole("button", {
      name: "Use baseline for Offload text encoder to CPU",
      exact: true,
    })
    .click();
  await expect(yaml).not.toContainText('"text_encoder":');
  await page.getByLabel("Seed", { exact: true }).fill("0");
  await expect(yaml).toContainText('"seed": 0');
  await page.getByRole("button", { name: "Set Served model name to null", exact: true }).click();
  await expect(yaml).toContainText('"served_model_name": null');
});
