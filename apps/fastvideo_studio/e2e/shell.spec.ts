import { expect, test } from '@playwright/test';

import { skipWithoutMock } from './helpers';

/**
 * App-shell smoke: the Next.js frontend hydrates, renders the FastVideo logo
 * and the primary-sidebar navigation, and routing between the top-level
 * sections works. Each spec self-skips when the mock backend isn't reachable.
 */
test.describe('app shell', () => {
  skipWithoutMock();

  test('loads with the logo and primary-sidebar nav', async ({ page }) => {
    await page.goto('/');
    // The root route redirects to /inference.
    await expect(page).toHaveURL(/\/inference$/);

    await expect(page.getByRole('img', { name: /fastvideo/i })).toBeVisible();

    for (const label of ['Inference', 'Datasets', 'Gallery', 'Settings']) {
      await expect(page.getByRole('link', { name: label })).toBeVisible();
    }
  });

  test('navigates between the primary sections', async ({ page }) => {
    await page.goto('/inference');
    await expect(
      page.getByRole('heading', { level: 1, name: 'Jobs' }),
    ).toBeVisible();

    const sections: Array<{ link: string; url: RegExp; title: string }> = [
      { link: 'Datasets', url: /\/datasets$/, title: 'Datasets' },
      { link: 'Gallery', url: /\/gallery$/, title: 'Gallery' },
      { link: 'Settings', url: /\/settings$/, title: 'Settings' },
      { link: 'Inference', url: /\/inference$/, title: 'Jobs' },
    ];

    for (const section of sections) {
      await page.getByRole('link', { name: section.link }).click();
      await expect(page).toHaveURL(section.url);
      // The header <h1> is the only level-1 heading and reflects the route.
      await expect(
        page.getByRole('heading', { level: 1, name: section.title }),
      ).toBeVisible();
      await expect(page.getByRole('main')).toHaveCount(1);
    }
  });

  test('keeps navigation and content usable at responsive breakpoints', async ({
    page,
  }) => {
    for (const width of [320, 375, 414, 768]) {
      await page.setViewportSize({ width, height: 800 });
      await page.goto('/inference');

      const main = page.getByRole('main');
      await expect(main).toBeVisible();
      await expect(
        page.getByRole('button', { name: /Create Job/i }),
      ).toBeVisible();

      const initialBox = await main.boundingBox();
      expect(initialBox?.x).toBe(width < 768 ? 0 : 220);
      expect(initialBox?.width).toBe(width < 768 ? width : width - 220);

      const navigation = page.getByRole('navigation', {
        name: 'Primary navigation',
      });
      if (width < 768) {
        await expect(
          page.getByRole('button', { name: 'Open navigation' }),
        ).toBeVisible();
        await page.getByRole('button', { name: 'Open navigation' }).click();
      }
      await expect(navigation).toBeVisible();
      await navigation.getByRole('link', { name: 'Datasets' }).click();

      await expect(page).toHaveURL(/\/datasets$/);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      ).toBe(true);
    }
  });

  test('uses full-width detail drawers on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 800 });
    await page.goto('/inference');

    await page
      .locator('article button[aria-pressed="false"]')
      .first()
      .click();
    const jobDrawer = page.getByRole('dialog', { name: 'Job details' });
    await expect(jobDrawer).toBeVisible();
    expect(await jobDrawer.boundingBox()).toMatchObject({ x: 0, width: 320 });
    await jobDrawer.getByRole('button', { name: 'Close' }).click();

    await page.goto('/datasets');
    await page
      .locator('article button[aria-pressed="false"]')
      .first()
      .click();

    const datasetDrawer = page.getByRole('dialog', {
      name: /dataset details$/,
    });
    await expect(datasetDrawer).toBeVisible();
    expect(await datasetDrawer.boundingBox()).toMatchObject({
      x: 0,
      width: 320,
    });
  });
});
