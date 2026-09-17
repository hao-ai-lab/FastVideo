import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import { getDatasets, getModels, updateJob } from '@/lib/api';
import type { JobLike } from '@/lib/jobToFields';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';
import { defaultOptionsStore } from '@/stores/defaultOptions';

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  updateJob: vi.fn(),
  getModels: vi.fn(),
  getDatasets: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

const trainingJob = {
  id: 'training-job',
  model_id: 'wan/model',
  name: 'saved job',
  prompt: 'saved description',
  data_path: '/datasets/train-original',
  validation_dataset_file: '/datasets/validation-original',
  max_train_steps: 2300,
  train_batch_size: 3,
  learning_rate: 0.0002,
  num_latent_t: 24,
  lora_rank: 64,
  dmd_use_vsa: true,
  dmd_vsa_sparsity: 0,
  dmd_denoising_steps: '900,500',
  real_score_guidance_scale: 2.5,
  generator_update_interval: 7,
  real_score_model_path: '/models/teacher',
  fake_score_model_path: '/models/critic',
};

beforeEach(() => {
  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue([{ id: 'wan/model', label: 'Wan' }]);
  vi.mocked(getDatasets).mockResolvedValue([
    { id: 'new-training', name: 'New training set', created_at: 0 },
    { id: 'new-validation', name: 'New validation set', created_at: 0 },
  ]);
  vi.mocked(updateJob).mockResolvedValue({ id: trainingJob.id } as never);
});

async function openEditor(workloadType = 'full_t2v', overrides: Partial<JobLike> = {}) {
  render(
    <CreateJobModal
      isOpen
      onClose={vi.fn()}
      onSuccess={vi.fn()}
      jobType={workloadType === 'dmd_t2v' ? 'distillation' : 'finetuning'}
      workloadType={workloadType}
      editingJob={{ ...trainingJob, workload_type: workloadType, ...overrides }}
    />,
  );
  await waitFor(() => expect(screen.getByLabelText('Model')).toBeEnabled());
  await waitFor(() => expect(screen.getByLabelText('Dataset *')).toBeEnabled());
}

describe('training job editing', () => {
  it.each(['full_t2v', 'lora_t2v', 'dmd_t2v'])(
    'preserves saved datasets and training settings when saving %s without changes',
    async (workloadType) => {
      const user = userEvent.setup();
      await openEditor(workloadType);

      expect(screen.getByLabelText('Dataset *')).toHaveValue(trainingJob.data_path);
      expect(screen.getByLabelText('Validation Dataset (optional)')).toHaveValue(
        trainingJob.validation_dataset_file,
      );
      await user.click(screen.getByRole('button', { name: 'Save Changes' }));

      await waitFor(() => expect(updateJob).toHaveBeenCalledTimes(1));
      const payload = vi.mocked(updateJob).mock.calls[0][1];
      expect(payload).toMatchObject({
        model_id: trainingJob.model_id,
        name: trainingJob.name,
        prompt: trainingJob.prompt,
        data_path: trainingJob.data_path,
        validation_dataset_file: trainingJob.validation_dataset_file,
        max_train_steps: trainingJob.max_train_steps,
        train_batch_size: trainingJob.train_batch_size,
        learning_rate: trainingJob.learning_rate,
        num_latent_t: trainingJob.num_latent_t,
        lora_rank: trainingJob.lora_rank,
      });
      if (workloadType === 'dmd_t2v') {
        expect(payload).toMatchObject({
          dmd_use_vsa: true,
          dmd_vsa_sparsity: 0,
          dmd_denoising_steps: '900,500',
          real_score_guidance_scale: 2.5,
          generator_update_interval: 7,
          real_score_model_path: '/models/teacher',
          fake_score_model_path: '/models/critic',
        });
      }
    },
  );

  it('allows changing both datasets to catalogue IDs', async () => {
    const user = userEvent.setup();
    await openEditor();
    await user.selectOptions(screen.getByLabelText('Dataset *'), 'new-training');
    await user.selectOptions(screen.getByLabelText('Validation Dataset (optional)'), 'new-validation');
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(updateJob).toHaveBeenCalledWith(trainingJob.id, expect.objectContaining({
      data_path: 'new-training', validation_dataset_file: 'new-validation',
    })));
  });

  it('sends an explicit empty value when clearing validation', async () => {
    const user = userEvent.setup();
    await openEditor();
    await user.selectOptions(screen.getByLabelText('Validation Dataset (optional)'), '');
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(updateJob).toHaveBeenCalledWith(trainingJob.id, expect.objectContaining({
      validation_dataset_file: '',
    })));
  });

  it('preserves a saved training model path outside the current catalogue', async () => {
    const user = userEvent.setup();
    await openEditor('dmd_t2v', { model_id: '/models/generator' });
    expect(screen.getByLabelText('Model')).toHaveValue('/models/generator');
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(updateJob).toHaveBeenCalledWith(trainingJob.id, expect.objectContaining({
      model_id: '/models/generator',
    })));
  });
});
