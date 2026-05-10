import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const projectStorageMockState = vi.hoisted(() => ({
  saveProject: vi.fn(),
  saveProjectMetadata: vi.fn(),
  listProjects: vi.fn(async () => []),
  loadProjectClips: vi.fn(async () => []),
  deleteProject: vi.fn(async () => {}),
  pruneOldProjects: vi.fn(async () => {}),
  reset() {
    this.saveProject.mockReset();
    this.saveProjectMetadata.mockReset();
    this.listProjects.mockClear();
    this.loadProjectClips.mockClear();
    this.deleteProject.mockClear();
    this.pruneOldProjects.mockClear();
  },
}));

vi.mock('../lib/storyPresetsData', () => ({
  default: [
    {
      id: 'test_preset',
      label: 'Test Preset',
      segment_prompts: ['segment one', 'segment two'],
    },
  ],
}));

vi.mock('../lib/projectStorage', () => ({
  saveProject: projectStorageMockState.saveProject,
  saveProjectMetadata: projectStorageMockState.saveProjectMetadata,
  listProjects: projectStorageMockState.listProjects,
  loadProjectClips: projectStorageMockState.loadProjectClips,
  deleteProject: projectStorageMockState.deleteProject,
  pruneOldProjects: projectStorageMockState.pruneOldProjects,
}));

vi.mock('../lib/media/avPipeline', () => ({
  DEFAULT_AV_MIME: 'video/mp4',
  createAvPipeline: vi.fn(() => ({
    reset() {},
    enqueueChunk() {},
    ensurePipeline: async () => {},
    maybeStartPlayback() {},
    tryEndStream() {},
    setStreamCompleted() {},
    noteSegmentInit() {},
    noteSegmentComplete() {},
    markStreamStarting() {},
    hasArchivedChunks() {
      return false;
    },
    hasArchivedCompletedSegments() {
      return false;
    },
    buildArchivedStreamChunks() {
      return [];
    },
    buildArchivedSegmentSnapshots() {
      return [];
    },
    buildArchivedStreamBlob() {
      return new Blob([], { type: 'video/mp4' });
    },
    takeArchivedStreamChunks() {
      return [];
    },
    takeArchivedSegmentSnapshots() {
      return [];
    },
    takeArchivedStreamBlob() {
      return new Blob([], { type: 'video/mp4' });
    },
    usesNativePlaybackFallback() {
      return false;
    },
  })),
}));

import Page from './page';

describe('Page startup readiness UX', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    projectStorageMockState.reset();
    window.history.pushState({}, '', '/');
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('shows a clear notice when the backend is not reachable before session start', async () => {
    fetchMock.mockRejectedValue(new Error('connect ECONNREFUSED'));

    const user = userEvent.setup();
    render(<Page />);

    const promptInput = screen.getByRole('textbox', { name: 'Continuation prompt' });
    await user.type(promptInput, 'A fox surfing through neon rain');
    await user.click(screen.getByRole('button', { name: 'Generate' }));

    expect(
      await screen.findByText(
        'Dreamverse backend is not reachable. Start uv run dreamverse-server and wait for /readyz to return 200 before retrying.',
      ),
    ).toBeInTheDocument();
    expect(promptInput).toHaveValue('A fox surfing through neon rain');
  });

  it('shows a readiness notice when GPU workers are not ready yet', async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;

      if (url.endsWith('/healthz')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ status: 'ok', service: 'ltx2-streaming-backend' }),
        };
      }

      if (url.endsWith('/readyz')) {
        return {
          ok: false,
          status: 503,
          json: async () => ({
            detail: 'No ready GPU worker processes.',
          }),
        };
      }

      if (url.endsWith('/status')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            total_gpus: 1,
            available_gpus: 0,
            queue_size: 0,
            warmup_enabled: true,
            warmup_successful_gpus: 0,
            warmup_failed_gpus: 0,
          }),
        };
      }

      throw new Error(`Unhandled fetch request in test: ${url}`);
    });

    const user = userEvent.setup();
    render(<Page />);

    const promptInput = screen.getByRole('textbox', { name: 'Continuation prompt' });
    await user.type(promptInput, 'A fox surfing through neon rain');
    await user.click(screen.getByRole('button', { name: 'Generate' }));

    expect(
      await screen.findByText(
        'Dreamverse backend is running, but GPU workers are not ready yet. Wait for startup warmup to finish and retry.',
      ),
    ).toBeInTheDocument();
    expect(promptInput).toHaveValue('A fox surfing through neon rain');
  });
});
