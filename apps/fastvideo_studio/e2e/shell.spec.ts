import { expect, test } from '@playwright/test';

import { MOCK_SKIP_MESSAGE, mockIsUp, seedApiBaseUrl } from './helpers';

/**
 * App-shell smoke: the Next.js frontend hydrates, renders the FastVideo logo
 * and the primary-sidebar navigation, and routing between the top-level
 * sections works. Each spec self-skips when the mock backend isn't reachable.
 */
test.describe('app shell', () => {
  test.beforeEach(async ({ page, request }) => {
    test.skip(!(await mockIsUp(request)), MOCK_SKIP_MESSAGE);
    await seedApiBaseUrl(page);
  });

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
    }
  });
});
