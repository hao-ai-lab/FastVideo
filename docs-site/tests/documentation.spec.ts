import { expect, test } from "@playwright/test";

test("documentation navigation opens transferred guides", async ({ page }) => {
  await page.goto("/FastVideo");
  await expect(page.getByRole("heading", { name: "Welcome to FastVideo", exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Installation", exact: true }).click();
  await expect(page).toHaveURL(/getting_started\/installation/);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
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
  await expect(page.getByRole("link", { name: /Open the cookbook Choose/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Run your first video Generate/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("the homepage opens the cookbook with the same guide navigation", async ({ page }) => {
  await page.goto("/FastVideo");
  const navigation = page.locator("#blume-nav-drawer");
  const guideLinks = await navigation
    .locator("a[href]")
    .evaluateAll((links) =>
      links.map((link) => ({ label: link.textContent?.trim(), href: link.getAttribute("href") })),
    );
  expect(guideLinks.length).toBeGreaterThan(20);
  await page.getByRole("link", { name: /Open the cookbook Choose/ }).click();
  await expect(page).toHaveURL(/FastVideo\/cookbook/);
  await expect(page.getByRole("heading", { name: "Configure a deployment", exact: true })).toBeVisible();
  expect(
    await navigation
      .locator("a[href]")
      .evaluateAll((links) =>
        links.map((link) => ({ label: link.textContent?.trim(), href: link.getAttribute("href") })),
      ),
  ).toEqual(guideLinks);
  await navigation.getByRole("link", { name: "Installation", exact: true }).click();
  await expect(page).toHaveURL(/getting_started\/installation/);
});

test("the welcome logo and self-hosted Paper Mono load under the deployment prefix", async ({ page }) => {
  const failedAssets: string[] = [];
  page.on("response", (response) => {
    if (["font", "image"].includes(response.request().resourceType()) && !response.ok())
      failedAssets.push(response.url());
  });
  await page.goto("/FastVideo");
  const logo = page.locator(".welcome-brand").getByRole("img", { name: "FastVideo", exact: true });
  await expect(logo).toBeVisible();
  expect(await logo.evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)).toBe(true);
  await page.evaluate(() => document.fonts.ready);
  const fonts = await page.evaluate(() => ({
    body: getComputedStyle(document.body).fontFamily,
    heading: getComputedStyle(document.querySelector("h1")!).fontFamily,
    loaded: [...document.fonts].some((font) => /paper.mono/i.test(font.family) && font.status === "loaded"),
  }));
  expect(fonts.body).toMatch(/paper.mono/i);
  expect(fonts.heading).toMatch(/paper.mono/i);
  expect(fonts.loaded).toBe(true);
  expect(failedAssets).toEqual([]);
});
