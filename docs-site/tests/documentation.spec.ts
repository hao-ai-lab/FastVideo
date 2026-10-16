import { expect, test } from "@playwright/test";

test("GitHub stars refresh automatically and survive a failed refresh", async ({ page }) => {
  await page.clock.install();
  let requests = 0;
  await page.route("https://api.github.com/repos/hao-ai-lab/FastVideo", async (route) => {
    requests++;
    if (requests === 2) return route.fulfill({ status: 503, body: "Unavailable" });
    await route.fulfill({ json: { stargazers_count: requests === 1 ? 4612 : 5234 } });
  });
  await page.goto("/FastVideo");
  const stars = page.locator(".home-github-stars");
  await expect(stars).toHaveText("4.6k");
  await expect(stars).toHaveAccessibleName("FastVideo on GitHub, 4,612 stars");
  await page.clock.fastForward(5 * 60 * 1000);
  await expect.poll(() => requests).toBe(2);
  await expect(stars).toHaveText("4.6k");
  await page.clock.fastForward(5 * 60 * 1000);
  await expect(stars).toHaveText("5.2k");
});

test("GitHub rate limits leave a usable Star link without a fabricated count", async ({ page }) => {
  await page.route("https://api.github.com/repos/hao-ai-lab/FastVideo", (route) =>
    route.fulfill({ status: 403, json: { message: "API rate limit exceeded" } }),
  );
  await page.goto("/FastVideo");
  const stars = page.getByRole("link", { name: "Star FastVideo on GitHub", exact: true });
  await expect(stars).toHaveText("Star");
  await expect(stars).toHaveAttribute("href", "https://github.com/hao-ai-lab/FastVideo");
});

test("documentation navigation opens transferred guides", async ({ page }) => {
  await page.goto("/FastVideo");
  await expect(page.getByRole("heading", { name: "Welcome to FastVideo", exact: true })).toBeVisible();
  await page.locator("#blume-nav-drawer").getByRole("link", { name: "Installation", exact: true }).click();
  await expect(page).toHaveURL(/getting_started\/installation/);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
});

test("the welcome page introduces the project and opens the cookbook", async ({ page }) => {
  await page.goto("/FastVideo");
  const logo = page.getByRole("heading", { level: 1, name: "Welcome to FastVideo" }).getByRole("img");
  await expect(logo).toBeVisible();
  expect(await logo.evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)).toBe(true);
  const column = await page.locator(".home-page").boundingBox();
  const image = await logo.boundingBox();
  expect(Math.abs(image!.x + image!.width / 2 - (column!.x + column!.width / 2))).toBeLessThan(2);
  await expect(page.getByRole("link", { name: "Hao AI Lab", exact: true })).toHaveAttribute(
    "href",
    "https://haoailab.com/",
  );
  const lab = page.locator(".home-lab-link img");
  expect(await lab.evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)).toBe(true);
  const trending = page.locator(".home-trendshift a");
  await expect(trending).toHaveAttribute(
    "href",
    "https://trendshift.io/repositories/14509?utm_source=repository-badge&utm_medium=badge&utm_campaign=badge-repository-14509",
  );
  await expect(trending.locator("img")).toHaveAttribute("src", "https://trendshift.io/api/badge/repositories/14509");
  await expect(page.locator(".home-news time")).toHaveCount(2);
  await expect(page.locator(".home-news")).toContainText("Kandinsky 6");
  await expect(page.getByRole("heading", { name: "Projects and research", exact: false })).toBeVisible();
  await expect(page.getByRole("link", { name: "DanceGRPO", exact: true })).toBeVisible();
  await expect(page.getByText("Use the navigation menu on the left to explore different sections:")).toHaveCount(0);
  await expect(page.getByText("About FastVideo and performance overview", { exact: true })).toHaveCount(0);
  await expect(page.locator(".home-page > p").first()).toContainText(
    "FastVideo is an inference and post-training framework",
  );
  await page.getByRole("link", { name: /^Cookbook Choose a model recipe/ }).click();
  await expect(page).toHaveURL(/FastVideo\/cookbook/);
  await expect(page.getByRole("heading", { name: "Configure a deployment", exact: true })).toBeVisible();
});

test("Fast Models connects the homepage, family collections, and checkpoints", async ({ page }) => {
  await page.goto("/FastVideo");
  await page.locator(".home-project-links").getByRole("link", { name: "Fast Models", exact: true }).click();
  await expect(page).toHaveURL(/FastVideo\/fast-models$/);
  await expect(page.getByRole("heading", { level: 1, name: "Fast Models", exact: true })).toBeVisible();
  for (const family of ["FastH3", "FastWan", "FastMetal"]) {
    await expect(page.getByRole("heading", { level: 2, name: family, exact: false })).toBeVisible();
  }
  await expect(page.getByRole("link", { name: "FastMetal collection", exact: true })).toHaveAttribute(
    "href",
    "https://huggingface.co/collections/FastVideo/fastmetal",
  );
  await expect(page.locator('a[href="https://huggingface.co/FastVideo/FastWan-QAD-FP8-1.3B"]').first()).toBeVisible();
  const checkpoints = await page
    .locator('a[href^="https://huggingface.co/FastVideo/"]')
    .evaluateAll((links) => [...new Set(links.map((link) => link.getAttribute("href")))]);
  expect(checkpoints.length).toBeGreaterThanOrEqual(18);
  await expect(
    page.locator("#blume-nav-drawer").getByRole("link", { name: "Fast Models", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("V1 API renders Python signatures within the Blume shell", async ({ page }) => {
  await page.goto("/FastVideo/getting_started/v1_api");
  await expect(page.getByRole("heading", { name: "V1 API", exact: true })).toBeVisible();
  await expect(page.locator('[id="fastvideo.entrypoints.video_generator.VideoGenerator"]')).toBeVisible();
  await expect(page.locator(".python-reference")).toContainText("from_pretrained");
  await expect(page.getByRole("button", { name: "Search", exact: true })).toBeVisible();
});

test("generated module links stay inside the transferred Python reference", async ({ page }) => {
  await page.goto("/FastVideo/api/fastvideo");
  await page.getByRole("link", { name: "configs", exact: true }).click();
  await expect(page).toHaveURL(/FastVideo\/api\/fastvideo\/configs/);
});

test("search finds a transferred guide and opens it by keyboard", async ({ page }) => {
  await page.goto("/FastVideo");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  const query = page.getByRole("combobox", { name: "Search docs", exact: true });
  await query.fill("offloading");
  await expect(page.getByRole("option", { name: /^Offloading/ }).first()).toBeVisible();
  await query.press("Enter");
  await expect(page).toHaveURL(/FastVideo\/inference\/offloading/);
});

test("the welcome cards reflow on mobile without horizontal scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/FastVideo");
  await expect(page.getByRole("heading", { name: "Welcome to FastVideo", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: /Quick start Generate/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("the model catalog filters by family and checkpoint name", async ({ page }) => {
  await page.goto("/FastVideo/fast-models");
  const catalog = page.locator(".model-catalog");
  await expect(catalog.getByRole("status")).toHaveText("18 checkpoints");
  await page.getByRole("button", { name: "FastMetal", exact: true }).click();
  await expect(catalog.getByRole("status")).toHaveText("3 checkpoints");
  await page.getByRole("searchbox", { name: "Search checkpoints", exact: true }).fill("14B");
  await expect(catalog.getByRole("status")).toHaveText("1 checkpoint");
  await page.getByRole("searchbox").fill("no-such-model");
  await expect(catalog.getByRole("status")).toHaveText("0 checkpoints");
  await page.getByRole("button", { name: "Clear filters", exact: true }).click();
  await expect(catalog.getByRole("status")).toHaveText("18 checkpoints");
});

test("hardware guides are grouped below general inference tasks", async ({ page }) => {
  await page.goto("/FastVideo");
  const nav = page.locator("#blume-nav-drawer");
  await expect(nav.getByRole("link", { name: "Two DGX Sparks", exact: true })).not.toBeVisible();
  await nav.getByText("Hardware", { exact: true }).click();
  await nav.getByRole("link", { name: "Two DGX Sparks", exact: true }).click();
  await expect(page).toHaveURL(/getting_started\/installation\/spark_pair/);
});

test("existing cookbook family recipes still load in the Blume shell", async ({ page }) => {
  await page.goto("/FastVideo/cookbook/wan/");
  await expect(page.locator("[data-cookbook-command]")).not.toHaveText("Loading...");
  await expect(page.locator("[data-cookbook-command]")).toContainText(/python|fastvideo/);
  await expect(page.getByRole("button", { name: "Search", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "All model families", exact: true })).toHaveAttribute(
    "href",
    /\/FastVideo\/cookbook\/?$/,
  );
});

test("generated example titles render once and retain their original fragment", async ({ page }) => {
  await page.goto("/FastVideo/inference/examples/basic");
  await expect(page.getByRole("heading", { name: "Basic Video Generation Tutorial", exact: true })).toHaveCount(1);
  await expect(page.locator('[id="basic-video-generation-tutorial"]')).toHaveCount(1);
  await expect(page.locator("article")).toContainText("examples/inference/basic");
});

test("hardware overview distinguishes runtime support and AMD documentation limits", async ({ page }) => {
  await page.goto("/FastVideo/getting_started/hardware");
  const table = page.getByRole("table");
  for (const platform of ["RTX", "data center", "DGX Spark", "Apple Silicon", "AMD"]) {
    await expect(table).toContainText(platform);
  }
  await page.getByRole("link", { name: "ROCm support and current limits", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("AMD and ROCm");
  await expect(page.locator("article")).toContainText("not currently documented");
});

test("homepage project buttons have visible borders in dark mode", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/FastVideo");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const colors = await page.locator(".home-project-links a").evaluateAll((links) =>
    links.map((link) => {
      const style = getComputedStyle(link);
      return {
        border: style.borderTopColor,
        width: style.borderTopWidth,
        background: getComputedStyle(document.body).backgroundColor,
      };
    }),
  );
  expect(colors).toHaveLength(3);
  for (const color of colors) {
    expect(color.width).toBe("1px");
    const luminance = (value: string) => {
      const channels = value
        .match(/[\d.]+/g)!
        .slice(0, 3)
        .map(Number)
        .map((channel) => (value.startsWith("color(srgb") ? channel : channel / 255))
        .map((channel) => (channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4));
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    };
    const border = luminance(color.border);
    const background = luminance(color.background);
    expect((Math.max(border, background) + 0.05) / (Math.min(border, background) + 0.05)).toBeGreaterThanOrEqual(3);
  }
});
