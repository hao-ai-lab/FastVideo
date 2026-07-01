import { expect, test } from '@playwright/test';

import { API_BASE, MOCK_SKIP_MESSAGE, mockIsUp, seedApiBaseUrl } from './helpers';

/**
 * Datasets page: the seeded datasets render as cards, and the Create Dataset
 * modal opens from the header action.
 */
test.describe('datasets', () => {
  test.beforeEach(async ({ page, request }) => {
    test.skip(!(await mockIsUp(request)), MOCK_SKIP_MESSAGE);
    await seedApiBaseUrl(page);
  });

  test('lists the seeded datasets', async ({ page, request }) => {
    // Read the seeded names from the mock so the assertion tracks the fixture.
    const res = await request.get(`${API_BASE}/datasets`);
    const datasets = (await res.json()) as Array<{ name: string }>;
    expect(datasets.length).toBeGreaterThan(0);

    await page.goto('/datasets');

    for (const ds of datasets) {
      await expect(page.getByText(ds.name, { exact: true })).toBeVisible();
    }
  });

  test('opens the Create Dataset modal', async ({ page }) => {
    await page.goto('/datasets');

    await page.getByRole('button', { name: 'Add Dataset' }).click();

    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(
      dialog.getByRole('heading', { name: /Add Dataset/i }),
    ).toBeVisible();
    await expect(dialog.getByLabel('Name', { exact: true })).toBeVisible();
  });
});
