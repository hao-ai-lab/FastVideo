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
  model_id: 'FastVideo/FastHunyuan-diffusers',
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
  it('renders one row per generator with its state and config summary', async () => {
    vi.mocked(listGenerators).mockResolvedValue([
      makeGenerator(),
      makeGenerator({
        state: 'loading',
        model_id: 'org/Big-Model',
        num_gpus: 8,
        enable_torch_compile: true,
      }),
      makeGenerator({
        state: 'failed',
        model_id: 'black-forest-labs/FLUX.1-schnell',
        error: 'CUDA out of memory',
      }),
    ]);
    render(<WarmModelsPanel />);

    // Model labels: last path segment with dashes/underscores as spaces.
    expect(
      await screen.findByText('FastHunyuan diffusers'),
    ).toBeInTheDocument();
    expect(screen.getByText('Big Model')).toBeInTheDocument();
    expect(screen.getByText('FLUX.1 schnell')).toBeInTheDocument();
    expect(screen.getByText('ready')).toBeInTheDocument();
    expect(screen.getByText('loading')).toBeInTheDocument();
    expect(screen.getByText('8 GPU · compile')).toBeInTheDocument();
    expect(screen.getByText('failed')).toHaveAttribute(
      'title',
      'CUDA out of memory',
    );
    // Only the ready row can be unloaded.
    expect(screen.getAllByRole('button', { name: 'Unload' })).toHaveLength(1);
  });

  it('preloads the selected model with the persisted default options', async () => {
    defaultOptionsStore.set({
      options: { ...DEFAULT_OPTIONS, numGpus: 4, enableTorchCompile: true },
    });
    const user = userEvent.setup();
    render(<WarmModelsPanel />);

    const button = await screen.findByRole('button', {
      name: 'Preload model',
    });
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
    // The panel refetches so the new "loading" row appears promptly.
    await waitFor(() => expect(listGenerators).toHaveBeenCalledTimes(2));
  });

  it('unloads a ready generator with its exact engine config', async () => {
    vi.mocked(listGenerators).mockResolvedValue([
      makeGenerator({ num_gpus: 2 }),
    ]);
    const user = userEvent.setup();
    render(<WarmModelsPanel />);

    await user.click(await screen.findByRole('button', { name: 'Unload' }));

    await waitFor(() =>
      expect(unloadGenerator).toHaveBeenCalledWith({
        model_id: 'FastVideo/FastHunyuan-diffusers',
        workload_type: 't2v',
        num_gpus: 2,
        dit_cpu_offload: false,
        text_encoder_cpu_offload: false,
        vae_cpu_offload: false,
        image_encoder_cpu_offload: false,
        use_fsdp_inference: false,
        enable_torch_compile: false,
        vsa_sparsity: 0,
        tp_size: -1,
        sp_size: -1,
      }),
    );
    // The list refreshes after the unload succeeds.
    await waitFor(() => expect(listGenerators).toHaveBeenCalledTimes(2));
  });
});
