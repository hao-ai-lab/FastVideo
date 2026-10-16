import * as React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import {
  createJob,
  getDatasets,
  getModels,
  updateJob,
  uploadMedia,
} from '@/lib/api';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import { makeOmniModel } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  updateJob: vi.fn(),
  getModels: vi.fn(),
  getDatasets: vi.fn(),
  uploadImage: vi.fn(),
  uploadMedia: vi.fn(),
  getUploadPreviewUrl: (path: string) => `http://api/uploads/preview?path=${path}`,
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

const OMNI = makeOmniModel();
const WAN = { id: 'wan/i2v', label: 'Wan I2V' };
const H3 = {
  id: 'MiniMaxAI/MiniMax-H3',
  label: 'MiniMax H3',
  capabilities: {
    ...OMNI.capabilities!,
    references: { ...OMNI.capabilities!.references!, required: false },
    start_frame: true,
    end_frame: true,
    num_gpus: null,
  },
};

beforeEach(() => {
  defaultOptionsStore.set({
    options: { ...DEFAULT_OPTIONS, defaultModelIdI2v: OMNI.id },
  });
  vi.mocked(getModels).mockResolvedValue([WAN, H3, OMNI]);
  vi.mocked(getDatasets).mockResolvedValue([]);
  vi.mocked(createJob).mockResolvedValue({ id: 'job-1' } as never);
  vi.mocked(updateJob).mockResolvedValue({} as never);
  vi.mocked(uploadMedia).mockImplementation(async (file: File) => ({
    path: `/uploads/${file.name}`,
    media_type: file.name.endsWith('.mp4')
      ? 'video'
      : file.name.endsWith('.wav')
        ? 'audio'
        : 'image',
  }));
});

function renderModal(
  overrides: Partial<React.ComponentProps<typeof CreateJobModal>> = {},
) {
  render(
    <CreateJobModal
      isOpen
      onClose={vi.fn()}
      onSuccess={vi.fn()}
      jobType="inference"
      workloadType="i2v"
      {...overrides}
    />,
  );
}

async function addReference(user: ReturnType<typeof userEvent.setup>, name: string) {
  const input = screen.getByLabelText('References (required, in order)');
  await user.upload(input, new File(['x'], name));
  await screen.findByText(name);
}

describe('CreateJobModal with FastH3-Omni', () => {
  it('replaces locked option controls with the recipe and hides keyframe inputs', async () => {
    renderModal();
    expect(await screen.findByText('Recipe (locked to the checkpoint)')).toBeInTheDocument();
    expect(
      screen.getByText('8 fused blocks of a 32-interval grid = 8 transformer forwards'),
    ).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent(/sm_89.*fail before any weights load/);

    expect(screen.queryByLabelText('Image')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('End Frame (optional)')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Negative Prompt')).not.toBeInTheDocument();
    for (const id of ['modal-num-steps', 'modal-guidance', 'modal-vsa-sparsity', 'modal-fps',
      'modal-tp-size', 'modal-sp-size', 'modal-enable-torch-compile']) {
      expect(document.getElementById(id)).toBeNull();
    }
    // Frames and resolution stay editable, seeded from the recipe defaults.
    expect(document.getElementById('modal-num-frames')).toHaveValue(124);
    // GPU counts that divide the 56 heads only.
    const gpus = screen.getByLabelText('GPUs');
    expect(within(gpus).getAllByRole('option').map((o) => o.textContent)).toEqual(['1', '2', '4', '7', '8']);
  });

  it('needs an image or video reference before submitting', async () => {
    const user = userEvent.setup();
    renderModal();
    await screen.findByText('Recipe (locked to the checkpoint)');
    await user.type(screen.getByLabelText('Prompt'), 'p');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    expect(await screen.findByText(/needs at least one image or video reference/)).toBeInTheDocument();

    await addReference(user, 'voice.wav');
    expect(screen.getByText(/Audio references must be paired/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    expect(createJob).not.toHaveBeenCalled();
  });

  it('submits references in the order the user arranged, with the recipe values', async () => {
    const user = userEvent.setup();
    renderModal();
    await screen.findByText('Recipe (locked to the checkpoint)');
    await addReference(user, 'dance.mp4');
    await addReference(user, 'outfit.png');
    await addReference(user, 'voice.wav');

    const items = () => screen.getAllByTestId('reference-item').map((li) => li.textContent);
    expect(items()[0]).toContain('<Video 1>');
    await user.click(screen.getByRole('button', { name: 'Move <Picture 1> up' }));
    expect(items()[0]).toContain('outfit.png');
    expect(items()[0]).toContain('<Picture 1>');
    // Previews come from the upload preview endpoint.
    expect(screen.getAllByTestId('reference-item')[0].querySelector('img')).toHaveAttribute(
      'src',
      'http://api/uploads/preview?path=/uploads/outfit.png',
    );

    await user.click(screen.getByRole('button', { name: 'Edit as raw prompt' }));
    await user.type(screen.getByLabelText('Prompt'), 'The dancer wears the outfit.');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    expect(createJob).toHaveBeenCalledTimes(1);
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload.model_id).toBe(OMNI.id);
    expect(payload.references).toEqual([
      { source: '/uploads/outfit.png', media_type: 'image' },
      { source: '/uploads/dance.mp4', media_type: 'video' },
      { source: '/uploads/voice.wav', media_type: 'audio' },
    ]);
    expect(payload.num_inference_steps).toBe(8);
    expect(payload.guidance_scale).toBe(1);
    expect(payload.vsa_sparsity).toBe(0.9);
    expect(payload.fps).toBe(24);
    expect(payload.image_path).toBeUndefined();
  });

  it('restores generic defaults when switching away from the recipe model', async () => {
    const user = userEvent.setup();
    renderModal();
    await screen.findByText('Recipe (locked to the checkpoint)');
    await user.selectOptions(screen.getByLabelText('Model'), WAN.id);
    expect(screen.queryByText('Recipe (locked to the checkpoint)')).not.toBeInTheDocument();
    expect(document.getElementById('modal-num-steps')).toHaveValue(
      DEFAULT_OPTIONS.numInferenceSteps,
    );
    // Wan reports no capabilities: no references, the start image is back.
    expect(screen.queryByLabelText('References (Ref2VA)')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Image')).toBeInTheDocument();
  });

  it('keeps base MiniMax-H3 references optional with keyframes available', async () => {
    renderModal();
    await screen.findByText('Recipe (locked to the checkpoint)');
    await userEvent.setup().selectOptions(screen.getByLabelText('Model'), H3.id);
    expect(screen.getByLabelText('References (Ref2VA)')).toBeInTheDocument();
    expect(screen.getByLabelText('End Frame (optional)')).toBeInTheDocument();
  });

  it('edits an Omni job without reseeding, keeping its reference order', async () => {
    const user = userEvent.setup();
    renderModal({
      editingJob: {
        id: 'job-9',
        model_id: OMNI.id,
        prompt: 'p',
        workload_type: 'i2v',
        num_frames: 200,
        num_inference_steps: 8,
        guidance_scale: 1,
        references: [
          { source: '/uploads/b.png', media_type: 'image' },
          { source: '/uploads/a.mp4', media_type: 'video' },
        ],
      },
    });
    await screen.findByText('Recipe (locked to the checkpoint)');
    expect(document.getElementById('modal-num-frames')).toHaveValue(200);
    await user.click(screen.getByRole('button', { name: 'Move <Video 1> up' }));
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));
    const updates = vi.mocked(updateJob).mock.calls[0][1] as { references: unknown; num_frames: number };
    expect(updates.references).toEqual([
      { source: '/uploads/a.mp4', media_type: 'video' },
      { source: '/uploads/b.png', media_type: 'image' },
    ]);
    expect(updates.num_frames).toBe(200);
  });

  it('blocks an unusable export with its error', async () => {
    vi.mocked(getModels).mockResolvedValue([
      makeOmniModel({ recipe: null, error: 'Missing checkpoint component: /b/vae' }),
    ]);
    renderModal();
    expect(await screen.findByText(/This model is unavailable: Missing checkpoint component/)).toBeInTheDocument();
  });
});
