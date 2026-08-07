import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import GpusPage from './page';
import { getClusterStatus } from '@/lib/api';
import type { ClusterSnapshot } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  getClusterStatus: vi.fn(),
}));

const RAY_SNAPSHOT: ClusterSnapshot = {
  mode: 'ray',
  error: null,
  resources: { gpus_total: 8, gpus_available: 5 },
  nodes: [
    {
      hostname: 'node-a',
      ip: '10.0.0.10',
      is_this_host: true,
      cpus: 64,
      ray_gpus: 4,
      available: true,
      error: null,
      gpus: [
        {
          index: 0,
          name: 'NVIDIA B200',
          utilization: 62,
          memory_used_mib: 40_960,
          memory_total_mib: 81_920,
          temperature_c: 41,
          power_watts: 312.4,
          power_limit_watts: 1000,
        },
        {
          index: 1,
          name: 'NVIDIA B200',
          utilization: 0,
          memory_used_mib: 1_024,
          memory_total_mib: 81_920,
          temperature_c: null,
          power_watts: null,
          power_limit_watts: null,
        },
      ],
    },
    {
      hostname: 'node-b',
      ip: '10.0.0.11',
      is_this_host: false,
      cpus: 32,
      ray_gpus: 2,
      available: true,
      error: null,
      gpus: [
        {
          index: 0,
          name: 'NVIDIA B200',
          utilization: 90,
          memory_used_mib: 20_480,
          memory_total_mib: 81_920,
          temperature_c: 70,
          power_watts: 900,
          power_limit_watts: 1000,
        },
      ],
    },
  ],
};

const LOCAL_SNAPSHOT: ClusterSnapshot = {
  mode: 'local',
  error:
    'not connected to a ray cluster yet (load a model first); showing the API host only',
  resources: null,
  nodes: [
    {
      hostname: 'localhost',
      ip: null,
      is_this_host: true,
      cpus: null,
      ray_gpus: null,
      available: true,
      error: null,
      gpus: [
        {
          index: 0,
          name: 'NVIDIA RTX 5090',
          utilization: 12,
          memory_used_mib: 2_048,
          memory_total_mib: 32_768,
          temperature_c: 38,
          power_watts: 80,
          power_limit_watts: 575,
        },
      ],
    },
  ],
};

beforeEach(() => {
  vi.mocked(getClusterStatus).mockResolvedValue(RAY_SNAPSHOT);
});

describe('GpusPage', () => {
  it('renders the header with mode and GPU totals', async () => {
    render(<GpusPage />);

    expect(await screen.findByText('ray cluster')).toBeInTheDocument();
    expect(screen.getByText(/5 \/\s*8 GPUs available/)).toBeInTheDocument();
  });

  it('renders a section per node with host details and GPU rows', async () => {
    render(<GpusPage />);

    // Each hostname appears twice: once in the strip, once as a section.
    expect(await screen.findAllByText('node-a')).toHaveLength(2);
    expect(screen.getAllByText('node-b')).toHaveLength(2);
    expect(screen.getByText('10.0.0.10')).toBeInTheDocument();
    // Only node-a is the API host.
    expect(screen.getAllByText('API host')).toHaveLength(1);
    expect(screen.getByText(/64 CPUs · 4 ray GPUs/)).toBeInTheDocument();

    expect(screen.getAllByText('NVIDIA B200')).toHaveLength(3);
    expect(screen.getByText('GPU 1')).toBeInTheDocument();
    expect(screen.getByText('62%')).toBeInTheDocument();
    expect(
      screen.getByText('40960 / 81920 MiB (40.0 GiB / 80.0 GiB)'),
    ).toBeInTheDocument();
    // Optional sensors render only when present.
    expect(screen.getByText('41°C')).toBeInTheDocument();
    expect(screen.getByText('312 W / 1000 W')).toBeInTheDocument();
  });

  it('bars reflect utilization and VRAM values', async () => {
    render(<GpusPage />);
    await screen.findAllByText('node-a');

    const utilMeters = screen
      .getAllByRole('meter', { name: 'Utilization' })
      .map((m) => m.getAttribute('aria-valuenow'));
    expect(utilMeters).toEqual(['62', '0', '90']);

    const vramMeters = screen
      .getAllByRole('meter', { name: 'VRAM' })
      .map((m) => m.getAttribute('aria-valuenow'));
    // 40960/81920 = 50%, 1024/81920 ≈ 1%, 20480/81920 = 25%
    expect(vramMeters).toEqual(['50', '1', '25']);
  });

  it('renders the compact strip with per-GPU segments', async () => {
    render(<GpusPage />);
    await screen.findAllByText('node-a');

    const segments = screen.getAllByRole('img');
    expect(segments).toHaveLength(3);
    expect(segments[0]).toHaveAccessibleName(
      'GPU 0: 62% utilization, 40.0 GiB / 80.0 GiB VRAM',
    );
  });

  it('shows the informational banner and local mode', async () => {
    vi.mocked(getClusterStatus).mockResolvedValue(LOCAL_SNAPSHOT);
    render(<GpusPage />);

    expect(await screen.findByText('local host only')).toBeInTheDocument();
    expect(
      screen.getByText(/not connected to a ray cluster yet/),
    ).toBeInTheDocument();
    // No resources in local mode.
    expect(screen.queryByText(/GPUs available/)).not.toBeInTheDocument();
  });

  it('explains when the API server is unreachable', async () => {
    vi.mocked(getClusterStatus).mockRejectedValue(new Error('network down'));
    render(<GpusPage />);
    expect(
      await screen.findByText(/Could not reach the API server/),
    ).toBeInTheDocument();
  });
});
