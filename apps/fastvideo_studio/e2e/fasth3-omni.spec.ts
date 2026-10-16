import { expect, test, type APIRequestContext, type Page } from '@playwright/test';

import { API_BASE, skipWithoutMock } from './helpers';

// The mock server configures this export as if started with --fasth3-omni-export.
const OMNI_ID = '/mock/models/fasth3_omni_export';
const SCREENSHOTS = process.env.FASTH3_OMNI_SCREENSHOT_DIR;

interface MockJob {
  id: string;
  model_id: string;
  prompt: string;
  status: string;
  references: { source: string; media_type: string }[];
  num_inference_steps: number;
  guidance_scale: number;
  error_category: string | null;
  recipe: { contract: { num_inference_steps: number } } | null;
}

async function jobs(request: APIRequestContext): Promise<MockJob[]> {
  return (await request.get(`${API_BASE}/jobs`)).json();
}

function file(name: string, mimeType: string) {
  return { name, mimeType, buffer: Buffer.from(`mock ${name}`) };
}

async function openOmniForm(page: Page) {
  await page.goto('/inference');
  await page.getByRole('button', { name: 'Create Job', exact: true }).click();
  await page.getByRole('menuitem', { name: /Image to Video/ }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('Model').selectOption(OMNI_ID);
  await expect(dialog.getByText('Recipe (locked to the checkpoint)')).toBeVisible();
  return dialog;
}

test.describe('FastH3-Omni', () => {
  skipWithoutMock();

  test('upload, reorder, submit, fail with a kernel error, edit and duplicate', async ({ page, request }) => {
    const before = new Set((await jobs(request)).map((j) => j.id));
    const dialog = await openOmniForm(page);

    // The recipe replaces the generic sampling controls; keyframes are gone.
    await expect(dialog.getByText('8 fused blocks of a 32-interval grid = 8 transformer forwards')).toBeVisible();
    await expect(dialog.getByRole('alert').filter({ hasText: 'sm_89' })).toBeVisible();
    await expect(dialog.getByLabel('Image', { exact: true })).toHaveCount(0);
    await expect(dialog.getByLabel('Negative Prompt')).toHaveCount(0);
    await dialog.getByText('Options').click();
    await expect(dialog.locator('#modal-num-steps')).toHaveCount(0);
    await expect(dialog.locator('#modal-guidance')).toHaveCount(0);

    const refs = dialog.getByLabel('References (required, in order)');
    await refs.setInputFiles(file('dance.mp4', 'video/mp4'));
    await expect(dialog.getByText('dance.mp4')).toBeVisible();
    await refs.setInputFiles(file('outfit.png', 'image/png'));
    await expect(dialog.getByText('outfit.png')).toBeVisible();
    await refs.setInputFiles(file('voice.wav', 'audio/wav'));
    await expect(dialog.getByText('voice.wav')).toBeVisible();

    const items = dialog.getByTestId('reference-item');
    await expect(items).toHaveCount(3);
    await dialog.getByRole('button', { name: 'Move <Picture 1> up' }).click();
    await expect(items.nth(0)).toContainText('outfit.png');
    await expect(items.nth(0)).toContainText('<Picture 1>');
    await expect(items.nth(1)).toContainText('<Video 1>');
    // The image preview really loads from the upload preview endpoint.
    await expect
      .poll(() => items.nth(0).locator('img').evaluate((img: HTMLImageElement) => img.naturalWidth))
      .toBeGreaterThan(0);
    if (SCREENSHOTS) await dialog.screenshot({ path: `${SCREENSHOTS}/omni-form.png` });

    await dialog.getByRole('button', { name: 'Edit as raw prompt' }).click();
    await dialog.getByLabel('Prompt').fill('The dancer from <Video 1> performs in the outfit from <Picture 1>.');
    await dialog.getByRole('button', { name: 'Create Job' }).click();
    await expect(dialog).toBeHidden();

    const created = (await jobs(request)).find((j) => !before.has(j.id));
    expect(created).toBeDefined();
    expect(created!.model_id).toBe(OMNI_ID);
    expect(created!.references.map((r) => r.media_type)).toEqual(['image', 'video', 'audio']);
    expect(created!.references[0].source).toMatch(/outfit\.png$/);
    expect(created!.num_inference_steps).toBe(8);
    expect(created!.guidance_scale).toBe(1);

    const card = page.locator('article').filter({ hasText: created!.id.slice(0, 8) });
    await expect(card.getByText('8 forwards · VSA 0.9 · tile 128')).toBeVisible();
    await expect(card.getByText('3 references')).toBeVisible();

    // The mock host cannot run tile 128: the run fails before loading weights.
    await card.getByRole('button', { name: 'Start' }).click();
    await expect(card.getByRole('alert')).toContainText('Kernel / hardware error:', { timeout: 15_000 });
    await expect(card.getByRole('alert')).toContainText('sm_100a');
    if (SCREENSHOTS) await card.screenshot({ path: `${SCREENSHOTS}/omni-failed-card.png` });

    // A failed job stays editable; the new order is saved.
    await card.getByRole('button', { name: 'Edit' }).click();
    const edit = page.getByRole('dialog');
    await expect(edit.getByTestId('reference-item').nth(0)).toContainText('outfit.png');
    await edit.getByRole('button', { name: 'Move <Video 1> up' }).click();
    await edit.getByRole('button', { name: 'Save Changes' }).click();
    await expect(edit).toBeHidden();
    const edited = (await jobs(request)).find((j) => j.id === created!.id)!;
    expect(edited.references.map((r) => r.media_type)).toEqual(['video', 'image', 'audio']);
    expect(edited.num_inference_steps).toBe(8);

    // Duplicating copies the references in their order.
    await card.getByRole('button', { name: 'Duplicate' }).click();
    await expect
      .poll(async () => (await jobs(request)).filter((j) => !before.has(j.id)).length)
      .toBe(2);
    const dup = (await jobs(request)).find((j) => !before.has(j.id) && j.id !== created!.id)!;
    expect(dup.references).toEqual(edited.references);
    expect(dup.status).toBe('pending');
  });

  test('blocks a submission without an image or video reference', async ({ page, request }) => {
    const count = (await jobs(request)).length;
    const dialog = await openOmniForm(page);
    await dialog.getByLabel('Prompt').fill('Only a voice.');
    await dialog.getByLabel('References (required, in order)').setInputFiles(file('voice.wav', 'audio/wav'));
    await expect(dialog.getByText('Audio references must be paired with at least one image or video.')).toBeVisible();
    await dialog.getByRole('button', { name: 'Create Job' }).click();
    await expect(dialog).toBeVisible();
    expect((await jobs(request)).length).toBe(count);
  });
});
