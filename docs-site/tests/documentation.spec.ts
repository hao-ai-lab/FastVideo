import { expect, test } from "@playwright/test";

test("documentation navigation opens transferred guides", async ({ page }) => {
  await page.goto("/FastVideo");
  await expect(page.getByRole("heading", { name: "Welcome to FastVideo", exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Installation", exact: true }).click();
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
  await expect(page.locator(".home-news time")).toHaveCount(2);
  await expect(page.locator(".home-news")).toContainText("Kandinsky 6");
  await expect(page.getByRole("heading", { name: "Projects and research", exact: false })).toBeVisible();
  await expect(page.getByRole("link", { name: "DanceGRPO", exact: true })).toBeVisible();
  await expect(page.getByText("Use the navigation menu on the left to explore different sections:")).toHaveCount(0);
  await page.getByRole("link", { name: /^Cookbook Choose a model recipe/ }).click();
  await expect(page).toHaveURL(/FastVideo\/cookbook/);
  await expect(page.getByRole("heading", { name: "Configure a deployment", exact: true })).toBeVisible();
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
