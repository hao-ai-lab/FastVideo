import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import WarmModelsPanel from './WarmModelsPanel';
import {
  getModels,
  listGenerators,
  preloadGenerator,
  unloadGenerator,
  type GeneratorInfo,
} from '@/lib/api';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import { toast } from 'sonner';

vi.mock('@/lib/api', () => ({
  getModels: vi.fn(),
  listGenerators: vi.fn(),
  preloadGenerator: vi.fn(),
  unloadGenerator: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

vi.mock('sonner', () => ({
  toast: { error: vi.fn() },
}));

const makeGenerator = (
  overrides: Partial<GeneratorInfo> = {},
): GeneratorInfo => ({
  state: 'ready',
  model_id: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
  workload_type: 't2v',
  num_gpus: 1,
  dit_cpu_offload: false,
  text_encoder_cpu_offload: false,
  vae_cpu_offload: false,
  image_encoder_cpu_offload: false,
  use_fsdp_inference: false,
  enable_torch_compile: false,
  vsa_sparsity: 0,
  tp_size: -1,
  sp_size: -1,
  error: null,
  ...overrides,
});

beforeEach(() => {
  // Reset the shared options store to a known baseline for test isolation.
  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue([
    { id: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers', label: 'Wan2.1 T2V 1.3B' },
  ]);
  vi.mocked(listGenerators).mockResolvedValue([]);
  vi.mocked(preloadGenerator).mockResolvedValue(
    makeGenerator({ state: 'loading' }),
  );
  vi.mocked(unloadGenerator).mockResolvedValue(undefined);
});

describe('WarmModelsPanel', () => {
  it('shows the empty slot when no model is loaded', async () => {
    render(<WarmModelsPanel />);

    expect(await screen.findByText('No model loaded')).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Unload' }),
    ).not.toBeInTheDocument();
  });

  it('renders the resident slot with its state and config summary', async () => {
    vi.mocked(listGenerators).mockResolvedValue([
      makeGenerator({ num_gpus: 8, enable_torch_compile: true }),
    ]);
    render(<WarmModelsPanel />);

    // Model label: last path segment with dashes/underscores as spaces.
    expect(
      await screen.findByText('Wan2.1 T2V 1.3B Diffusers'),
    ).toBeInTheDocument();
    expect(screen.getByText('ready')).toBeInTheDocument();
    expect(screen.getByText('8 GPU · compile')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Unload' })).toBeInTheDocument();
  });

  it('disables loading a new model while a load is in flight', async () => {
    vi.mocked(listGenerators).mockResolvedValue([
      makeGenerator({ state: 'loading' }),
    ]);
    render(<WarmModelsPanel />);

    expect(await screen.findByText('loading')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Load model' })).toBeDisabled();
    expect(
      screen.queryByRole('button', { name: 'Unload' }),
    ).not.toBeInTheDocument();
  });

  it('shows the error on a failed slot and keeps retry enabled', async () => {
    vi.mocked(listGenerators).mockResolvedValue([
      makeGenerator({ state: 'failed', error: 'CUDA out of memory' }),
    ]);
    render(<WarmModelsPanel />);

    expect(await screen.findByText('failed')).toHaveAttribute(
      'title',
      'CUDA out of memory',
    );
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Load model' })).toBeEnabled(),
    );
  });

  it('labels the load button as a swap when a different model is resident', async () => {
    vi.mocked(listGenerators).mockResolvedValue([
      makeGenerator({ model_id: 'FastVideo/FastHunyuan-diffusers' }),
    ]);
    render(<WarmModelsPanel />);

    expect(
      await screen.findByRole('button', { name: 'Load (replaces current)' }),
    ).toBeInTheDocument();
  });

  it('loads the selected model with the persisted default options', async () => {
    defaultOptionsStore.set({
      options: { ...DEFAULT_OPTIONS, numGpus: 4, enableTorchCompile: true },
    });
    const user = userEvent.setup();
    render(<WarmModelsPanel />);

    const button = await screen.findByRole('button', { name: 'Load model' });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    await waitFor(() =>
      expect(preloadGenerator).toHaveBeenCalledWith({
        model_id: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        workload_type: 't2v',
        num_gpus: 4,
        dit_cpu_offload: false,
        text_encoder_cpu_offload: false,
        vae_cpu_offload: false,
        image_encoder_cpu_offload: false,
        use_fsdp_inference: false,
        enable_torch_compile: true,
        vsa_sparsity: 0,
        tp_size: -1,
        sp_size: -1,
      }),
    );
    // The panel refetches so the new "loading" slot appears promptly.
    await waitFor(() => expect(listGenerators).toHaveBeenCalledTimes(2));
  });

  it('surfaces the backend detail when a load is rejected', async () => {
    vi.mocked(preloadGenerator).mockRejectedValue(
      new Error('a model load is already in progress'),
    );
    const user = userEvent.setup();
    render(<WarmModelsPanel />);

    const button = await screen.findByRole('button', { name: 'Load model' });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith('Model was not loaded', {
        description: 'a model load is already in progress',
      }),
    );
  });

  it('unloads the resident model with no payload', async () => {
    vi.mocked(listGenerators).mockResolvedValue([makeGenerator()]);
    const user = userEvent.setup();
    render(<WarmModelsPanel />);

    await user.click(await screen.findByRole('button', { name: 'Unload' }));

    await waitFor(() => expect(unloadGenerator).toHaveBeenCalledTimes(1));
    expect(vi.mocked(unloadGenerator).mock.calls[0]).toEqual([]);
    // The slot refreshes after the unload succeeds.
    await waitFor(() => expect(listGenerators).toHaveBeenCalledTimes(2));
  });
});
