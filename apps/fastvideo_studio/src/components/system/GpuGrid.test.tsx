import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import GpuGrid from './GpuGrid';
import { getGpus } from '@/lib/api';
import type { GpuSnapshot } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  getGpus: vi.fn(),
}));

const SNAPSHOT: GpuSnapshot = {
  available: true,
  error: null,
  gpus: [
    {
      index: 0,
      name: 'NVIDIA B200',
      utilization: 62,
      memory_used_mib: 61_440,
      memory_total_mib: 183_359,
      temperature_c: 41,
      power_watts: 312.4,
      power_limit_watts: 1000,
    },
    {
      index: 1,
      name: 'NVIDIA B200',
      utilization: 0,
      memory_used_mib: 1_024,
      memory_total_mib: 183_359,
      temperature_c: null,
      power_watts: null,
      power_limit_watts: null,
    },
  ],
};

beforeEach(() => {
  vi.mocked(getGpus).mockResolvedValue(SNAPSHOT);
});

describe('GpuGrid', () => {
  it('renders a card per GPU with utilization and memory', async () => {
    render(<GpuGrid />);

    expect(await screen.findAllByText('NVIDIA B200')).toHaveLength(2);
    expect(screen.getByText('GPU 0')).toBeInTheDocument();
    expect(screen.getByText('GPU 1')).toBeInTheDocument();
    expect(screen.getByText('62%')).toBeInTheDocument();
    expect(screen.getByText('60.0 GiB / 179.1 GiB')).toBeInTheDocument();
    // Optional sensors render only when present.
    expect(screen.getByText('41°C')).toBeInTheDocument();
    expect(screen.getByText('312 W / 1000 W')).toBeInTheDocument();
  });

  it('shows the backend-reported reason when telemetry is unavailable', async () => {
    vi.mocked(getGpus).mockResolvedValue({
      available: false,
      gpus: [],
      error: 'NVML Shared Library Not Found',
    });
    render(<GpuGrid />);
    expect(
      await screen.findByText(/GPU telemetry unavailable: NVML/),
    ).toBeInTheDocument();
  });

  it('explains when the API server is unreachable', async () => {
    vi.mocked(getGpus).mockRejectedValue(new Error('network down'));
    render(<GpuGrid />);
    expect(
      await screen.findByText(/Could not reach the API server/),
    ).toBeInTheDocument();
  });
});
