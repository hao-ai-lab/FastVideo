import * as React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import { toast } from 'sonner';
import { createJob, getDatasets, getModels, uploadImage } from '@/lib/api';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';

vi.mock('sonner', () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  getModels: vi.fn(),
  getDatasets: vi.fn(),
  uploadImage: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

const MODELS = [
  { id: 'wan/t2v-1.3b', label: 'Wan T2V', type: 't2v' },
  { id: 'wan/t2v-14b', label: 'Wan T2V Large', type: 't2v' },
];

beforeEach(() => {
  // Reset the shared options store to a known baseline for test isolation.
  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue(MODELS);
  vi.mocked(getDatasets).mockResolvedValue([]);
  vi.mocked(uploadImage).mockResolvedValue({ path: '/uploads/x.png' });
  vi.mocked(createJob).mockResolvedValue({ id: 'job-1' } as never);
});

function renderModal(
  overrides: Partial<React.ComponentProps<typeof CreateJobModal>> = {},
) {
  const onClose = vi.fn();
  const onSuccess = vi.fn();
  render(
    <CreateJobModal
      isOpen
      onClose={onClose}
      onSuccess={onSuccess}
      jobType="inference"
      workloadType="t2v"
      {...overrides}
    />,
  );
  return { onClose, onSuccess };
}

describe('CreateJobModal', () => {
  it('shows a model loading error instead of an empty model list', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(getModels).mockRejectedValueOnce(new Error('network down'));

    renderModal();

    expect(
      await screen.findByText(/Models could not be loaded/),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Model')).toHaveAttribute(
      'aria-invalid',
      'true',
    );
  });

  it('keeps the form open and reports job creation failures', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(createJob).mockRejectedValueOnce(new Error('API rejected job'));
    const user = userEvent.setup();
    const { onClose, onSuccess } = renderModal();

    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    await user.type(screen.getByLabelText('Prompt'), 'a careful test prompt');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    expect(
      await screen.findByText(/API rejected job.*then try again/),
    ).toBeInTheDocument();
    expect(onSuccess).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('reports image upload failures next to the file input', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(uploadImage).mockRejectedValueOnce(new Error('Upload failed'));
    const user = userEvent.setup();
    renderModal({ workloadType: 'i2v' });

    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    const input = screen.getByLabelText('Image');
    await user.upload(
      input,
      new File(['image'], 'input.png', { type: 'image/png' }),
    );

    expect(
      await screen.findByText(/Upload failed.*Choose the image again/),
    ).toBeInTheDocument();
    expect(input).toHaveAttribute('aria-invalid', 'true');
  });

  it('renders the form fields for an inference job', async () => {
    renderModal();

    expect(
      await screen.findByText('New Inference Job (T2V)'),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Model')).toBeInTheDocument();
    expect(screen.getByLabelText('Prompt')).toBeInTheDocument();
    expect(screen.getByLabelText('Negative Prompt')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Create Job' }),
    ).toBeInTheDocument();

    // The model dropdown is populated once getModels resolves.
    expect(
      await screen.findByRole('option', {
        name: 'Wan T2V (wan/t2v-1.3b)',
      }),
    ).toBeInTheDocument();
  });

  it('seeds fields from the options store and submits an inference payload', async () => {
    // Non-default store values prove the open-time seeding effect ran (the
    // useState defaults are 50 / 480).
    defaultOptionsStore.set({
      options: { ...DEFAULT_OPTIONS, numInferenceSteps: 25, height: 720 },
    });

    const user = userEvent.setup();
    const { onClose, onSuccess } = renderModal();

    // Wait for models to load so the default model is selected.
    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });

    await user.type(
      screen.getByLabelText('Prompt'),
      'a raccoon in sunflowers',
    );
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload).toMatchObject({
      model_id: 'wan/t2v-1.3b',
      prompt: 'a raccoon in sunflowers',
      workload_type: 't2v',
      job_type: 'inference',
      num_inference_steps: 25,
      height: 720,
      num_frames: 81,
      width: 832,
      guidance_scale: 5,
      seed: 1024,
    });

    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('submits a dmd_t2v distillation payload including the DMD fields', async () => {
    vi.mocked(getDatasets).mockResolvedValue([
      { id: 'ds1', name: 'My Dataset', created_at: 0 },
    ]);

    const user = userEvent.setup();
    const { onSuccess } = renderModal({
      jobType: 'distillation',
      workloadType: 'dmd_t2v',
    });

    // Models + datasets load asynchronously on open. The model/dataset option
    // labels also appear in the Real/Fake Score Model and Validation Dataset
    // selects, so scope each wait to the relevant select.
    await within(screen.getByLabelText('Model')).findByRole('option', {
      name: 'Wan T2V (wan/t2v-1.3b)',
    });
    const datasetSelect = screen.getByLabelText('Dataset *');
    await within(datasetSelect).findByRole('option', { name: 'My Dataset' });

    await user.type(screen.getByLabelText('Description'), 'distill run');
    await user.selectOptions(datasetSelect, 'ds1');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload).toMatchObject({
      workload_type: 'dmd_t2v',
      job_type: 'distillation',
      // The dataset id is sent; the backend resolves it to the on-disk dir.
      data_path: 'ds1',
      lora_rank: 32,
      // DMD-specific fields added to CreateJobRequest for this modal.
      dmd_use_vsa: false,
      dmd_vsa_sparsity: 0.8,
      dmd_denoising_steps: '1000,757,522',
      real_score_guidance_scale: 3.5,
      generator_update_interval: 5,
      real_score_model_path: 'wan/t2v-1.3b',
      fake_score_model_path: 'wan/t2v-1.3b',
    });
    // Inference-only keys must be absent for a training job.
    expect(payload).not.toHaveProperty('num_inference_steps');

    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
  });

  it('shows the H3 speech-tag console only for an H3 model and inserts at the prompt cursor', async () => {
    vi.mocked(getModels).mockResolvedValue([
      ...MODELS,
      { id: 'minimax-h3/i2v', label: 'MiniMax H3', type: 'i2v' },
    ]);
    const user = userEvent.setup();
    renderModal({ workloadType: 'i2v' });

    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    expect(screen.queryByText(/Speech tags/)).not.toBeInTheDocument();

    await user.selectOptions(
      screen.getByLabelText('Model'),
      'minimax-h3/i2v',
    );
    expect(await screen.findByText(/Speech tags/)).toBeInTheDocument();

    const promptField = screen.getByLabelText('Prompt') as HTMLTextAreaElement;
    await user.type(promptField, 'Hello there.');
    promptField.setSelectionRange(5, 5);

    await user.click(screen.getByRole('button', { name: '<pause>' }));

    expect(promptField.value).toBe('Hello <pause> there.');
    // Focus and cursor return to the prompt field so tags can be chained.
    await waitFor(() => expect(promptField).toHaveFocus());
  });
});

const H3 = { id: 'minimax-h3/i2v', label: 'MiniMax H3', type: 'i2v' };

const BASE_PROMPT = [
  'For the target video, <Picture 1> is fully referenced.',
  '',
  'integrated_multimodal_description: [Shot 1] A close-up. He (S1) says: <d>[English] Sorry.</d> <cutoff>',
  '',
  'overall_soundscape: Room tone.',
  '',
  'non_diegetic_music: N/A',
].join('\n');

describe('CreateJobModal with a prefilled job', () => {
  beforeEach(() => {
    vi.mocked(getModels).mockResolvedValue([...MODELS, H3]);
    vi.mocked(toast.warning).mockClear();
  });

  it('fills the form from the job and creates a new job on submit', async () => {
    const user = userEvent.setup();
    const { onSuccess } = renderModal({
      workloadType: 'i2v',
      prefillJob: {
        id: 'imported-1',
        model_id: H3.id,
        name: 'from-file',
        prompt: BASE_PROMPT,
        workload_type: 'i2v',
        references: [{ source: '/data/inputs/face.png', media_type: 'image' }],
        num_frames: 243,
        seed: 7,
      },
    });

    await screen.findByRole('option', { name: /MiniMax H3/ });
    await waitFor(() => expect(screen.getByLabelText('Model')).toHaveValue(H3.id));
    expect(screen.getByLabelText('Name (optional)')).toHaveValue('from-file');
    expect(screen.getByLabelText('Prompt')).toHaveValue(BASE_PROMPT);
    expect(screen.getByText('face.png')).toBeInTheDocument();
    expect(screen.getByText('New Inference Job (I2V)')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0] as unknown as Record<string, unknown>;
    expect(payload).toMatchObject({
      model_id: H3.id,
      name: 'from-file',
      prompt: BASE_PROMPT,
      num_frames: 243,
      seed: 7,
      references: [{ source: '/data/inputs/face.png', media_type: 'image' }],
    });
  });

  it("warns when the file's model isn't offered for the workload", async () => {
    renderModal({
      workloadType: 't2v',
      prefillJob: { id: 'imported-2', model_id: 'not/a-real-model', prompt: 'p' },
    });
    await waitFor(() => expect(toast.warning).toHaveBeenCalled());
    expect(vi.mocked(toast.warning).mock.calls[0][0]).toContain('not/a-real-model');
    expect(screen.getByLabelText('Model')).toHaveValue(MODELS[0].id);
  });
});

describe('CreateJobModal shot list on a raw prompt', () => {
  beforeEach(() => {
    vi.mocked(getModels).mockResolvedValue([...MODELS, H3]);
  });

  async function openShotList(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByRole('button', { name: 'Open shot list' }));
    return screen.findByRole('dialog', { name: 'Shot list' });
  }

  it('edits dialogue inside integrated_multimodal_description and leaves the rest of the prompt alone', async () => {
    const user = userEvent.setup();
    renderModal({
      workloadType: 'i2v',
      prefillJob: {
        id: 'imported-3',
        model_id: H3.id,
        prompt: BASE_PROMPT,
        references: [{ source: '/data/face.png', media_type: 'image' }],
      },
    });
    await waitFor(() => expect(screen.getByLabelText('Model')).toHaveValue(H3.id));

    const dialog = await openShotList(user);
    expect(within(dialog).getAllByText(/integrated_multimodal_description/).length).toBeGreaterThan(0);
    expect(within(dialog).queryByText('Retention analysis')).not.toBeInTheDocument();

    const line = within(dialog).getByPlaceholderText('What the speaker says') as HTMLTextAreaElement;
    expect(line.value).toBe('Sorry.');
    await user.type(line, ' Really.');
    await user.click(within(dialog).getByRole('button', { name: 'Apply' }));

    await waitFor(() =>
      expect(screen.getByLabelText('Prompt')).toHaveValue(BASE_PROMPT.replace('Sorry.', 'Sorry. Really.')),
    );
  });

  it('treats a plain prompt as the shot text and writes shot markers back over it', async () => {
    const user = userEvent.setup();
    renderModal({ workloadType: 't2v' });
    await screen.findByRole('option', { name: /MiniMax H3/ });
    await user.selectOptions(screen.getByLabelText('Model'), H3.id);
    await user.type(screen.getByLabelText('Prompt'), 'A dock at dawn.');

    const dialog = await openShotList(user);
    await user.click(within(dialog).getByRole('button', { name: 'Apply' }));
    await waitFor(() => expect(screen.getByLabelText('Prompt')).toHaveValue('[Shot 1] A dock at dawn.'));
  });

  it('keeps retention analysis in step when the raw prompt has a detailed_description and retention_analysis', async () => {
    const user = userEvent.setup();
    const sections = [
      'subject_definitions:',
      '<Subject 1> is the courier.',
      '',
      'retention_analysis:',
      '<Subject 1> (appears in [Shot 1]): fully_preserved - jacket.',
      '',
      'detailed_description:',
      '[Shot 1] Wide shot. A rooftop with <Subject 1>.',
    ].join('\n');
    renderModal({ workloadType: 't2v', prefillJob: { id: 'imported-4', model_id: H3.id, prompt: sections } });
    await waitFor(() => expect(screen.getByLabelText('Model')).toHaveValue(H3.id));

    const dialog = await openShotList(user);
    expect(within(dialog).getByText('Retention analysis')).toBeInTheDocument();
    await user.click(within(dialog).getAllByRole('button', { name: /Add shot/ })[0]);
    await user.click(within(dialog).getAllByRole('button', { name: '<Subject 1>' })[1]);
    await user.click(within(dialog).getByRole('button', { name: 'Apply' }));

    const out = (screen.getByLabelText('Prompt') as HTMLTextAreaElement).value;
    expect(out).toContain('<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - jacket.');
    expect(out).toContain('subject_definitions:\n<Subject 1> is the courier.');
  });

  it('is not offered for a non-H3 model', async () => {
    renderModal({ workloadType: 't2v' });
    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    expect(screen.queryByRole('button', { name: 'Open shot list' })).not.toBeInTheDocument();
  });
});
