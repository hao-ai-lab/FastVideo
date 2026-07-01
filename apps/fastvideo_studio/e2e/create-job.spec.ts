import { expect, test } from '@playwright/test';

import { MOCK_SKIP_MESSAGE, mockIsUp, seedApiBaseUrl } from './helpers';

/**
 * Create-job flow: open the Create Job modal on /inference, fill the prompt
 * (the model auto-selects once the mock's /api/models loads), submit, and
 * confirm the new job lands in the queue.
 */
test.describe('create inference job', () => {
  test.beforeEach(async ({ page, request }) => {
    test.skip(!(await mockIsUp(request)), MOCK_SKIP_MESSAGE);
    await seedApiBaseUrl(page);
  });

  test('creates a T2V job and shows it in the queue', async ({ page }) => {
    await page.goto('/inference');

    // The "Create Job" button reveals a workload menu on hover; wait for the
    // T2V item to become visible before clicking so the CSS hover transition
    // can't race the click.
    await page.getByRole('button', { name: /create job/i }).hover();
    const t2vItem = page.getByRole('menuitem', { name: /T2V/i });
    await expect(t2vItem).toBeVisible();
    await t2vItem.click();

    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();

    // Wait for the mock's model catalogue to populate the dropdown (more than
    // just the disabled placeholder), then pick one explicitly — the app's
    // auto-selection is racy.
    const modelSelect = dialog.getByLabel('Model', { exact: true });
    await expect(modelSelect.locator('option')).not.toHaveCount(1);
    await modelSelect.selectOption({ index: 1 });

    const prompt = `e2e raccoon in sunflowers ${Date.now()}`;
    await dialog.getByLabel('Prompt', { exact: true }).fill(prompt);

    await dialog.getByRole('button', { name: 'Create Job' }).click();

    // Modal closes and the queue refreshes with the newly created job.
    await expect(dialog).toBeHidden();
    await expect(page.getByText(prompt)).toBeVisible();
  });
});
