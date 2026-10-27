import { expect, test } from '@playwright/test';

import { skipWithoutMock } from './helpers';

test.describe('create job interactions', () => {
  skipWithoutMock();

  for (const jobType of ['inference', 'finetuning', 'distillation']) {
    test(`${jobType} remains interactive after repeated dialog dismissals`, async ({ page }) => {
      await page.goto(`/${jobType}`);
      const trigger = page.getByRole('button', { name: 'Create Job', exact: true });
      const dialog = page.getByRole('dialog');

      // Exercise both dismissal paths and reopen without reloading the page.
      for (const closeWithEscape of [false, true]) {
        await trigger.click();
        await page.getByRole('menuitem').first().click();
        await expect(dialog).toBeVisible();
        if (closeWithEscape) {
          await page.keyboard.press('Escape');
        } else {
          await dialog.getByRole('button', { name: 'Close', exact: true }).click();
        }
        await expect(dialog).toBeHidden();
        await expect(page.locator('body')).toHaveCSS('pointer-events', 'auto');
        await expect(trigger).toBeFocused();
      }

      await page.getByRole('link', { name: 'Datasets', exact: true }).click();
      await expect(page).toHaveURL(/\/datasets$/);
    });
  }

  test('preserves keyboard menu dismissal and dialog focus trapping', async ({ page }) => {
    await page.goto('/inference');
    const trigger = page.getByRole('button', { name: 'Create Job', exact: true });
    await trigger.focus();
    await page.keyboard.press('Enter');
    const firstItem = page.getByRole('menuitem').first();
    await expect(firstItem).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('menu')).toBeHidden();
    await expect(trigger).toBeFocused();
    await expect(page.locator('body')).toHaveCSS('pointer-events', 'auto');

    await page.keyboard.press('Enter');
    await expect(firstItem).toBeFocused();
    await page.keyboard.press('Enter');
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog.getByLabel('Name (optional)')).toBeFocused();

    // Shift+Tab from the first field wraps to Close, then Tab wraps back.
    await page.keyboard.press('Shift+Tab');
    await expect(dialog.getByRole('button', { name: 'Close', exact: true })).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(dialog.getByLabel('Name (optional)')).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    await expect(trigger).toBeFocused();
    await expect(page.locator('body')).toHaveCSS('pointer-events', 'auto');
  });
});
