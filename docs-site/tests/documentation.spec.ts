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
  await expect(page.getByRole("link", { name: /Quick start Generate/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
