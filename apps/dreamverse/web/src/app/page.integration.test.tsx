import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Server } from 'mock-socket';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const avPipelineMockState = vi.hoisted(() => ({
  useNativePlaybackFallback: false,
}));

const projectStorageMockState = vi.hoisted(() => {
  const state = {
    projects: [] as any[],
    clips: [] as any[],
    commitSave: async (project: any, clips: any[]) => {
      state.projects = [
        project,
        ...state.projects.filter((entry) => entry.id !== project.id),
      ];
      state.clips = [
        ...state.clips.filter((clip) => clip.projectId !== project.id),
        ...clips.map((clip) => ({ ...clip })),
      ];
    },
    saveProject: vi.fn(async (project: any, clips: any[]) => {
      await state.commitSave(project, clips);
    }),
    saveProjectMetadata: vi.fn(async (project: any) => {
      state.projects = [
        project,
        ...state.projects.filter((entry) => entry.id !== project.id),
      ];
    }),
    listProjects: vi.fn(async () => [...state.projects]),
    loadProjectClips: vi.fn(async (projectId: string) =>
      state.clips
        .filter((clip) => clip.projectId === projectId)
        .map((clip) => ({ ...clip })),
    ),
    deleteProject: vi.fn(async (projectId: string) => {
      state.projects = state.projects.filter((project) => project.id !== projectId);
      state.clips = state.clips.filter((clip) => clip.projectId !== projectId);
    }),
    pruneOldProjects: vi.fn(async () => {}),
    reset() {
      state.projects = [];
      state.clips = [];
      state.saveProject.mockReset();
      state.saveProject.mockImplementation(async (project: any, clips: any[]) => {
        await state.commitSave(project, clips);
      });
      state.saveProjectMetadata.mockReset();
      state.saveProjectMetadata.mockImplementation(async (project: any) => {
        state.projects = [
          project,
          ...state.projects.filter((entry) => entry.id !== project.id),
        ];
      });
      state.listProjects.mockClear();
      state.loadProjectClips.mockClear();
      state.deleteProject.mockClear();
      state.pruneOldProjects.mockClear();
    },
  };

  return state;
});

vi.mock('../lib/storyPresetsData', () => ({
  default: [
    {
      id: 'test_preset',
      label: 'Test Preset',
      segment_prompts: ['segment one', 'segment two'],
    },
  ],
}));

vi.mock('../lib/media/avPipeline', () => ({
  DEFAULT_AV_MIME: 'video/mp4',
  createAvPipeline: vi.fn(({ onPlaybackStarted = () => {} }) => {
    let archivedChunks: any[] = [];
    let archivedSegments: any[] = [];
    let activeSegmentKey = '';

    function cloneChunk(chunk: any) {
      if (chunk instanceof ArrayBuffer) {
        return chunk.slice(0);
      }
      if (ArrayBuffer.isView(chunk)) {
        return chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength);
      }
      return chunk;
    }

    return {
      reset: vi.fn(() => {
        archivedChunks = [];
        archivedSegments = [];
        activeSegmentKey = '';
      }),
      enqueueChunk: vi.fn((chunk: any) => {
        const clonedChunk = cloneChunk(chunk);
        archivedChunks = [...archivedChunks, clonedChunk];
        if (!activeSegmentKey) {
          const key = `implicit-${archivedSegments.length + 1}`;
          activeSegmentKey = key;
          archivedSegments = [
            ...archivedSegments,
            {
              key,
              segmentIdx: null,
              streamId: key,
              mime: 'video/mp4',
              completed: false,
              chunks: [],
            },
          ];
        }
        const target = archivedSegments.find((segment) => segment.key === activeSegmentKey);
        if (target) {
          target.chunks.push(clonedChunk);
        }
      }),
      ensurePipeline: vi.fn(async () => {}),
      maybeStartPlayback: vi.fn(() => {
        onPlaybackStarted();
      }),
      tryEndStream: vi.fn(() => {}),
      setStreamCompleted: vi.fn(() => {}),
      noteSegmentInit: vi.fn(({ segmentIdx = null, streamId = '', mime = 'video/mp4' } = {}) => {
        const normalizedStreamId = streamId || `stream-${archivedSegments.length + 1}`;
        const key = `${segmentIdx !== null ? segmentIdx : 'na'}:${normalizedStreamId}`;
        const existing = archivedSegments.find((segment) => segment.key === key);
        if (existing) {
          existing.completed = false;
          existing.mime = mime || existing.mime;
        } else {
          archivedSegments = [
            ...archivedSegments,
            {
              key,
              segmentIdx,
              streamId: normalizedStreamId,
              mime: mime || 'video/mp4',
              completed: false,
              chunks: [],
            },
          ];
        }
        activeSegmentKey = key;
      }),
      noteSegmentComplete: vi.fn(({ segmentIdx = null, streamId = '' } = {}) => {
        let target = null as any;
        if (streamId) {
          target = archivedSegments.find(
            (segment) => segment.streamId === streamId && (segmentIdx === null || segment.segmentIdx === segmentIdx),
          ) || null;
        }
        if (!target && activeSegmentKey) {
          target = archivedSegments.find((segment) => segment.key === activeSegmentKey) || null;
        }
        if (target) {
          target.completed = true;
          if (target.key === activeSegmentKey) {
            activeSegmentKey = '';
          }
        }
      }),
      markStreamStarting: vi.fn(() => {
        archivedChunks = [];
        archivedSegments = [];
        activeSegmentKey = '';
      }),
      hasArchivedChunks: vi.fn(() => {
        return archivedChunks.length > 0;
      }),
      hasArchivedCompletedSegments: vi.fn(() => {
        return archivedSegments.some((segment) => segment.completed && segment.chunks.length > 0);
      }),
      buildArchivedStreamChunks: vi.fn(() => archivedChunks.map((chunk) => {
        if (chunk instanceof ArrayBuffer) {
          return chunk.slice(0);
        }
        return chunk;
      })),
      buildArchivedSegmentSnapshots: vi.fn(({ includeInProgress = true } = {}) => archivedSegments
        .filter((segment) => segment.chunks.length > 0 && (includeInProgress || segment.completed))
        .map((segment) => ({
          ...segment,
          chunks: segment.chunks.map((chunk: any) => cloneChunk(chunk)),
        }))),
      buildArchivedStreamBlob: vi.fn(() => {
        return new Blob(archivedChunks, { type: 'video/mp4' });
      }),
      takeArchivedStreamChunks: vi.fn(() => {
        const chunks = archivedChunks.map((chunk) => {
          if (chunk instanceof ArrayBuffer) {
            return chunk.slice(0);
          }
          return chunk;
        });
        archivedChunks = [];
        archivedSegments = [];
        activeSegmentKey = '';
        return chunks;
      }),
      takeArchivedSegmentSnapshots: vi.fn(({ includeInProgress = true } = {}) => {
        const snapshots = archivedSegments
          .filter((segment) => segment.chunks.length > 0 && (includeInProgress || segment.completed))
          .map((segment) => ({
            ...segment,
            chunks: segment.chunks.map((chunk: any) => cloneChunk(chunk)),
          }));
        if (includeInProgress) {
          archivedSegments = [];
          activeSegmentKey = '';
        } else {
          archivedSegments = archivedSegments.filter((segment) => !segment.completed);
          if (activeSegmentKey && !archivedSegments.some((segment) => segment.key === activeSegmentKey)) {
            activeSegmentKey = '';
          }
        }
        return snapshots;
      }),
      takeArchivedStreamBlob: vi.fn(() => {
        const blob = new Blob(archivedChunks, { type: 'video/mp4' });
        archivedChunks = [];
        archivedSegments = [];
        activeSegmentKey = '';
        return blob;
      }),
      usesNativePlaybackFallback() {
        return avPipelineMockState.useNativePlaybackFallback;
      },
    };
  }),
}));

vi.mock('../lib/projectStorage', () => ({
  saveProject: projectStorageMockState.saveProject,
  saveProjectMetadata: projectStorageMockState.saveProjectMetadata,
  listProjects: projectStorageMockState.listProjects,
  loadProjectClips: projectStorageMockState.loadProjectClips,
  deleteProject: projectStorageMockState.deleteProject,
  pruneOldProjects: projectStorageMockState.pruneOldProjects,
}));

import Page from './page';
import { createAvPipeline } from '../lib/media/avPipeline';

function getWsUrl() {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${window.location.host}/ws`;
}

describe.skip('App websocket integration', () => {
  let server: Server;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    avPipelineMockState.useNativePlaybackFallback = false;
    projectStorageMockState.reset();
    window.history.pushState({}, '', '/');
    server = new Server(getWsUrl());
    fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
      if (url.endsWith('/healthz')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: 'ok',
            service: 'ltx2-streaming-backend',
          }),
        };
      }
      if (url.endsWith('/readyz')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: 'ready',
            service: 'ltx2-streaming-backend',
            ready_gpu_workers: 1,
            total_gpus: 1,
            available_gpus: 1,
            warmup_successful_gpus: 1,
            warmup_failed_gpus: 0,
            queue_size: 0,
          }),
        };
      }
      if (url.endsWith('/status')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            total_gpus: 1,
            available_gpus: 1,
            queue_size: 0,
            warmup_enabled: true,
            warmup_successful_gpus: 1,
            warmup_failed_gpus: 0,
          }),
        };
      }
      throw new Error(`Unhandled fetch request in test: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    server.stop();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('renders the streaming chat workspace and hides advanced panels', async () => {
    render(<Page />);

    expect(await screen.findByRole('heading', { name: 'Realtime streaming video workspace' }))
      .toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'History' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Collapse history sidebar' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Main chat' })).toBeInTheDocument();
    expect(screen.getByLabelText('Story preset')).toBeInTheDocument();
    expect(screen.getByLabelText('Continuation prompt')).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Guide next segment' })).not.toBeInTheDocument();
    expect(screen.getByText('Video replies')).toBeInTheDocument();
    expect(screen.queryByText('Session setup')).not.toBeInTheDocument();

    expect(screen.queryByText('Live Prompt Input')).not.toBeInTheDocument();
    expect(screen.queryByText('Prompt Window')).not.toBeInTheDocument();
    expect(screen.queryByText('Editable Prompt Segments')).not.toBeInTheDocument();
    expect(screen.queryByText('Depth Mode: System Prompt Editor')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Prompt count')).not.toBeInTheDocument();
  });

  it('collapses and re-expands the history sidebar', async () => {
    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Collapse history sidebar' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Expand history sidebar' })).toBeInTheDocument();
      expect(screen.queryByRole('heading', { name: 'History' })).not.toBeInTheDocument();
      expect(screen.getByText('Clips')).toBeInTheDocument();
      expect(screen.getByText('Prompts')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Expand history sidebar' }));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'History' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Collapse history sidebar' })).toBeInTheDocument();
    });
  });

  it('keeps the session timer visible and pauses session playback while viewing a saved project', async () => {
    projectStorageMockState.projects = [
      {
        id: 'saved_project_1',
        label: 'Saved Project',
        originalLabel: 'Saved History Project',
        presetId: 'saved_preset',
        createdAt: Date.now() - 60_000,
        lastThumbnail: null,
        promptEvents: [],
      },
    ];
    projectStorageMockState.clips = [
      {
        id: 'saved_clip_1',
        projectId: 'saved_project_1',
        label: 'Saved Clip',
        prompt: 'saved prompt',
        mime: 'video/mp4',
        blob: new Blob([new Uint8Array([9, 9, 9])], { type: 'video/mp4' }),
        createdAt: Date.now() - 60_000,
      },
    ];

    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    await waitFor(() => {
      expect(screen.getByText(/Time left:/i)).toBeInTheDocument();
    });

    const sessionVideos = Array.from(container.querySelectorAll('video'));
    const liveSessionVideo = sessionVideos[1] as HTMLVideoElement;
    expect(liveSessionVideo).toBeTruthy();

    const playbackState = { paused: false };
    Object.defineProperty(liveSessionVideo, 'paused', {
      configurable: true,
      get: () => playbackState.paused,
    });
    Object.defineProperty(liveSessionVideo, 'ended', {
      configurable: true,
      get: () => false,
    });
    Object.defineProperty(liveSessionVideo, 'readyState', {
      configurable: true,
      get: () => 4,
    });

    const pauseSpy = vi.spyOn(liveSessionVideo, 'pause').mockImplementation(() => {
      playbackState.paused = true;
    });
    const playSpy = vi.spyOn(liveSessionVideo, 'play').mockImplementation(async () => {
      playbackState.paused = false;
    });

    await user.click(screen.getByRole('button', { name: 'Toggle sidebar' }));
    await user.click(await screen.findByRole('button', { name: 'Saved History Project' }));

    await waitFor(() => {
      expect(screen.getByText('View-only project')).toBeInTheDocument();
    });

    expect(screen.getByText(/Time left:/i)).toBeInTheDocument();
    expect(pauseSpy).toHaveBeenCalledTimes(1);
    expect(playbackState.paused).toBe(true);

    playSpy.mockClear();

    await user.click(screen.getByRole('button', { name: 'Back' }));

    await waitFor(() => {
      expect(screen.queryByText('View-only project')).not.toBeInTheDocument();
    });

    expect(playSpy).toHaveBeenCalledTimes(1);
    expect(playbackState.paused).toBe(false);
  });

  it('does not resume session playback after leaving a saved project if it was already paused', async () => {
    projectStorageMockState.projects = [
      {
        id: 'saved_project_1',
        label: 'Saved Project',
        originalLabel: 'Saved History Project',
        presetId: 'saved_preset',
        createdAt: Date.now() - 60_000,
        lastThumbnail: null,
        promptEvents: [],
      },
    ];
    projectStorageMockState.clips = [
      {
        id: 'saved_clip_1',
        projectId: 'saved_project_1',
        label: 'Saved Clip',
        prompt: 'saved prompt',
        mime: 'video/mp4',
        blob: new Blob([new Uint8Array([9, 9, 9])], { type: 'video/mp4' }),
        createdAt: Date.now() - 60_000,
      },
    ];

    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    const sessionVideos = Array.from(container.querySelectorAll('video'));
    const liveSessionVideo = sessionVideos[1] as HTMLVideoElement;
    expect(liveSessionVideo).toBeTruthy();

    const playbackState = { paused: true };
    Object.defineProperty(liveSessionVideo, 'paused', {
      configurable: true,
      get: () => playbackState.paused,
    });
    Object.defineProperty(liveSessionVideo, 'ended', {
      configurable: true,
      get: () => false,
    });
    Object.defineProperty(liveSessionVideo, 'readyState', {
      configurable: true,
      get: () => 4,
    });

    const pauseSpy = vi.spyOn(liveSessionVideo, 'pause').mockImplementation(() => {
      playbackState.paused = true;
    });
    const playSpy = vi.spyOn(liveSessionVideo, 'play').mockImplementation(async () => {
      playbackState.paused = false;
    });

    await user.click(screen.getByRole('button', { name: 'Toggle sidebar' }));
    await user.click(await screen.findByRole('button', { name: 'Saved History Project' }));

    await waitFor(() => {
      expect(screen.getByText('View-only project')).toBeInTheDocument();
    });

    expect(pauseSpy).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Back' }));

    await waitFor(() => {
      expect(screen.queryByText('View-only project')).not.toBeInTheDocument();
    });

    expect(playSpy).not.toHaveBeenCalled();
    expect(playbackState.paused).toBe(true);
  });

  it('keeps the active prompt window collapsed by default and expands it on toggle', async () => {
    const user = userEvent.setup();
    render(<Page />);

    expect(await screen.findByRole('button', { name: 'Show prompts' })).toBeInTheDocument();
    expect(
      screen.getByText(
        '2 prompts in the active window. Expand when you want to inspect the full rollout context.',
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText('Window 1')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Show prompts' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Hide prompts' })).toBeInTheDocument();
      expect(screen.getByText('Window 1')).toBeInTheDocument();
      expect(screen.getByText('Window 2')).toBeInTheDocument();
    });
  });

  it('sends session_init_v2 payload when Generate is clicked', async () => {
    const outbound: any[] = [];
    server.on('connection', (socket) => {
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    const generateButton = await screen.findByRole('button', { name: 'Generate' });
    await waitFor(() => expect(generateButton).toBeEnabled());
    await user.click(generateButton);

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'session_init_v2')).toBe(true);
    });

    const initMessage = outbound.find((message) => message.type === 'session_init_v2');
    expect(initMessage.preset_id).toBe('test_preset');
    expect(initMessage.curated_prompts).toEqual(['segment one', 'segment two']);
    expect(initMessage.enhancement_enabled).toBe(true);
    expect(initMessage.auto_extension_enabled).toBe(false);
    expect(initMessage.loop_generation_enabled).toBe(false);
    expect(initMessage.initial_rollout_prompt).toBe('');
  });

  it('starts a streaming session from a custom initial prompt without using curated prompts', async () => {
    const outbound: any[] = [];
    server.on('connection', (socket) => {
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(
      continuationInput,
      'A lone astronaut walks through a flooded moonbase corridor lit by failing red alarms',
    );

    const generateButton = await screen.findByRole('button', { name: 'Generate' });
    await waitFor(() => expect(generateButton).toBeEnabled());
    await user.click(generateButton);

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'session_init_v2')).toBe(true);
    });

    const initMessage = outbound.find((message) => message.type === 'session_init_v2');
    expect(initMessage.preset_id).toBe('custom_editable');
    expect(initMessage.preset_label).toBe('Custom rollout');
    expect(initMessage.curated_prompts).toEqual([]);
    expect(initMessage.initial_rollout_prompt)
      .toBe(
        'A lone astronaut walks through a flooded moonbase corridor lit by failing red alarms',
      );
  });

  it('shows a specific notice when a second websocket from the same IP is rejected', async () => {
    const outbound: any[] = [];
    const sockets: any[] = [];
    server.on('connection', (socket) => {
      sockets.push(socket);
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });

      if (sockets.length === 2) {
        socket.send(JSON.stringify({
          type: 'error',
          error_code: 'ip_session_limit',
          message: 'Only one active websocket session is allowed per IP. Close the other session and retry.',
        }));
        socket.close();
      }
    });

    const user = userEvent.setup();
    render(<Page />);
    render(<Page />);

    const generateButtons = await screen.findAllByRole('button', { name: 'Generate' });
    await user.click(generateButtons[0]);

    await waitFor(() => {
      expect(sockets).toHaveLength(1);
      expect(outbound.filter((message) => message.type === 'session_init_v2')).toHaveLength(1);
    });

    await user.click(generateButtons[1]);

    await waitFor(() => {
      expect(sockets).toHaveLength(2);
      expect(outbound.filter((message) => message.type === 'session_init_v2')).toHaveLength(2);
    });

    await waitFor(() => {
      expect(screen.getByText(
        'Only one active websocket session is allowed per IP. Close the other session and click Run to retry.',
      )).toBeInTheDocument();
    });
  });

  it('shows a clear notice when the backend is not reachable before session start', async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
      if (url.endsWith('/healthz')) {
        throw new Error('network down');
      }
      throw new Error(`Unhandled fetch request in test: ${url}`);
    });

    const user = userEvent.setup();
    render(<Page />);

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(continuationInput, 'A neon city skyline in the rain');
    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    expect(await screen.findByText(
      'Dreamverse backend is not reachable. Start uv run dreamverse-server and wait for /readyz to return 200 before retrying.',
    )).toBeInTheDocument();
    expect(screen.getByLabelText('Continuation prompt')).toHaveValue('A neon city skyline in the rain');
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
          json: async () => ({ status: 'ok' }),
        };
      }
      if (url.endsWith('/readyz')) {
        return {
          ok: false,
          status: 503,
          json: async () => ({ detail: 'No ready GPU worker processes.' }),
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

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(continuationInput, 'A cathedral drifting through clouds');
    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    expect(await screen.findByText(
      'Dreamverse backend is running, but GPU workers are not ready yet. Wait for startup warmup to finish and retry.',
    )).toBeInTheDocument();
    expect(screen.getByLabelText('Continuation prompt')).toHaveValue('A cathedral drifting through clouds');
  });

  it('submits rewrite requests from the chat composer', async () => {
    const outbound: any[] = [];
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await waitFor(() => expect(continuationInput).toBeEnabled());
    await user.type(
      continuationInput,
      'Make the mood more ominous and push closer to the subject',
    );

    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'rewrite_seed_prompts')).toBe(true);
    });

    const rewriteMessage = outbound.find((message) => message.type === 'rewrite_seed_prompts');
    expect(rewriteMessage.rewrite_instruction)
      .toBe('Make the mood more ominous and push closer to the subject');
    expect(Array.isArray(rewriteMessage.prompt_window_prompts)).toBe(true);
  });

  it('sends rewrite requests from the continuation composer', async () => {
    const outbound: any[] = [];
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await waitFor(() => expect(continuationInput).toBeEnabled());
    await user.type(continuationInput, 'A dramatic reveal');

    const rewriteButton = screen.getByRole('button', { name: 'Rewrite rollout' });
    await waitFor(() => expect(rewriteButton).toBeEnabled());
    await user.click(rewriteButton);

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'rewrite_seed_prompts')).toBe(true);
    });

    const rewriteMessage = outbound.find((message) => message.type === 'rewrite_seed_prompts');
    expect(rewriteMessage.rewrite_instruction).toBe('A dramatic reveal');
    expect(Array.isArray(rewriteMessage.prompt_window_prompts)).toBe(true);
  });

  it('uses the selected archived version as the next rewrite source', async () => {
    const outbound: any[] = [];
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'segment two',
      source: 'curated',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([4, 5, 6]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(continuationInput, 'Turn it into a desert');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    await waitFor(() => {
      expect(outbound.filter((message) => message.type === 'rewrite_seed_prompts')).toHaveLength(1);
    });

    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_started' }));
    clientSocket.send(JSON.stringify({
      type: 'seed_prompts_updated',
      reason: 'rewrite',
      prompts: [
        'desert one',
        'desert two',
      ],
    }));
    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_complete' }));
    clientSocket.send(JSON.stringify({ type: 'seed_prompts_reset_applied' }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_stream_start',
      total_segments: 2,
      loop_generation_enabled: false,
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'desert one',
      source: 'rewrite',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([7, 8, 9]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'desert two',
      source: 'rewrite',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([10, 11, 12]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await screen.findByText('Turn it into a desert');

    await user.click(screen.getByText('Original'));

    const freshContinuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(freshContinuationInput, 'Add snowfall');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    await waitFor(() => {
      expect(outbound.filter((message) => message.type === 'rewrite_seed_prompts')).toHaveLength(2);
    });

    const secondRewrite = outbound
      .filter((message) => message.type === 'rewrite_seed_prompts')
      .at(-1);
    expect(secondRewrite.prompt_window_prompts.slice(0, 2)).toEqual([
      'segment one',
      'segment two',
    ]);
  });

  it('highlights the selected edit history item and rewrites from that selected version', async () => {
    const outbound: any[] = [];
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'segment two',
      source: 'curated',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([4, 5, 6]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(continuationInput, 'Turn it into a desert');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    await waitFor(() => {
      expect(outbound.filter((message) => message.type === 'rewrite_seed_prompts')).toHaveLength(1);
    });

    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_started' }));
    clientSocket.send(JSON.stringify({
      type: 'seed_prompts_updated',
      reason: 'rewrite',
      prompts: [
        'desert one',
        'desert two',
      ],
    }));
    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_complete' }));
    clientSocket.send(JSON.stringify({ type: 'seed_prompts_reset_applied' }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_stream_start',
      total_segments: 2,
      loop_generation_enabled: false,
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'desert one',
      source: 'rewrite',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([7, 8, 9]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'desert two',
      source: 'rewrite',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([10, 11, 12]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await screen.findByText('Turn it into a desert');

    const secondContinuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(secondContinuationInput, 'Add snowfall');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    await waitFor(() => {
      expect(outbound.filter((message) => message.type === 'rewrite_seed_prompts')).toHaveLength(2);
    });

    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_started' }));
    clientSocket.send(JSON.stringify({
      type: 'seed_prompts_updated',
      reason: 'rewrite',
      prompts: [
        'snow one',
        'snow two',
      ],
    }));
    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_complete' }));
    clientSocket.send(JSON.stringify({ type: 'seed_prompts_reset_applied' }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_stream_start',
      total_segments: 2,
      loop_generation_enabled: false,
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'snow one',
      source: 'rewrite',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([13, 14, 15]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'snow two',
      source: 'rewrite',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([16, 17, 18]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    const currentHistoryRow = await screen.findByText('Add snowfall');
    await waitFor(() => {
      expect(currentHistoryRow.closest('[data-selected]')).toHaveAttribute('data-selected', 'true');
    });

    const olderHistoryRow = screen.getByText('Turn it into a desert');
    await user.click(olderHistoryRow);

    await waitFor(() => {
      expect(olderHistoryRow.closest('[data-selected]')).toHaveAttribute('data-selected', 'true');
      expect(currentHistoryRow.closest('[data-selected]')).toHaveAttribute('data-selected', 'false');
    });

    const thirdContinuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(thirdContinuationInput, 'Make it rainy');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    await waitFor(() => {
      expect(outbound.filter((message) => message.type === 'rewrite_seed_prompts')).toHaveLength(3);
    });

    const thirdRewrite = outbound
      .filter((message) => message.type === 'rewrite_seed_prompts')
      .at(-1);
    expect(thirdRewrite.prompt_window_prompts.slice(0, 2)).toEqual([
      'desert one',
      'desert two',
    ]);
  });

  it('submits the rewrite-only continuation composer with Enter', async () => {
    const outbound: any[] = [];
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await waitFor(() => expect(continuationInput).toBeEnabled());
    await user.type(continuationInput, 'A dramatic reveal{Enter}');

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'rewrite_seed_prompts')).toBe(true);
    });

    const rewriteMessage = outbound.find((message) => message.type === 'rewrite_seed_prompts');
    expect(rewriteMessage.rewrite_instruction).toBe('A dramatic reveal');
    expect(continuationInput).toHaveValue('');
  });

  it('submits rewrite requests with Enter from the composer', async () => {
    const outbound: any[] = [];
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await waitFor(() => expect(continuationInput).toBeEnabled());
    await user.type(
      continuationInput,
      'Make the mood more ominous and push closer to the subject{Enter}',
    );

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'rewrite_seed_prompts')).toBe(true);
    });

    const rewriteMessage = outbound.find((message) => message.type === 'rewrite_seed_prompts');
    expect(rewriteMessage.rewrite_instruction)
      .toBe('Make the mood more ominous and push closer to the subject');
    expect(continuationInput).toHaveValue('');
  });

  it('keeps the completed 30s rollout in the main player and adds one video-reply card', async () => {
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'segment two',
      source: 'curated',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([4, 5, 6]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(container.querySelectorAll('.gallery-card')).toHaveLength(1);
    });
    expect(container.querySelector('.stage-copy h2')?.textContent).toBe('Test Preset');
    expect(screen.getByRole('button', { name: /Test Preset/i })).toBeInTheDocument();
    expect(container.querySelector('.gallery-card.is-active')).toBeNull();
  });

  it('keeps prior clips available while a rewrite-driven restart is underway', async () => {
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'segment two',
      source: 'curated',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([4, 5, 6]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(container.querySelectorAll('.gallery-card')).toHaveLength(1);
    });

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(continuationInput, 'The camera cuts to a rooftop chase');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    clientSocket.send(JSON.stringify({
      type: 'rewrite_seed_prompts_started',
    }));
    clientSocket.send(JSON.stringify({
      type: 'seed_prompts_updated',
      reason: 'rewrite',
      prompts: [
        'The camera cuts to a rooftop chase',
        'A wider aerial view of the rooftop pursuit',
      ],
    }));
    clientSocket.send(JSON.stringify({
      type: 'rewrite_seed_prompts_complete',
    }));
    clientSocket.send(JSON.stringify({
      type: 'seed_prompts_reset_applied',
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_stream_start',
      total_segments: 2,
      loop_generation_enabled: false,
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'The camera cuts to a rooftop chase',
      source: 'rewrite',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([4, 5, 6]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));

    await waitFor(() => {
      expect(container.querySelectorAll('.gallery-card')).toHaveLength(1);
    });
    expect(container.textContent).toContain('The camera cuts to a rooftop chase');
    expect(container.querySelector('.stage-copy h2')?.textContent).toBe('Cuts 2');

    const galleryLabelsBeforeSelection = Array.from<Element>(
      container.querySelectorAll('.gallery-card .gallery-card-label'),
    ).map((node: Element) => node.textContent || '');
    expect(galleryLabelsBeforeSelection).toEqual(['Test Preset']);

    await user.click(screen.getByRole('button', { name: /Test Preset/i }));

    await waitFor(() => {
      expect(container.querySelector('.stage-copy h2')?.textContent).toBe('Test Preset');
    });

    const galleryLabelsAfterSelection = Array.from<Element>(
      container.querySelectorAll('.gallery-card .gallery-card-label'),
    ).map((node: Element) => node.textContent || '');
    expect(galleryLabelsAfterSelection).toEqual(['Test Preset']);
  });

  it('shows a timeout popup with repo and blog links when the session expires', async () => {
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));
    clientSocket.send(JSON.stringify({
      type: 'session_timeout',
    }));

    const dialog = await screen.findByRole('dialog', { name: 'Session ended' });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText(/5-minute session limit/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open repo' })).toHaveAttribute('href', 'https://github.com/hao-ai-lab/FastVideo');
    expect(screen.getByRole('link', { name: 'Read blog' })).toHaveAttribute('href', 'https://hao-ai-lab.github.io/blogs/fastvideo/');
  });

  it('recreates the live media pipeline when a rewrite restarts mid-generation', async () => {
    const outbound: any[] = [];
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_stream_start',
      total_segments: 2,
      generation_segment_cap: 6,
      loop_generation_enabled: false,
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'segment two',
      source: 'curated',
      seed_prompt_index: 1,
    }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Rewrite rollout' })).toBeEnabled();
    });

    const continuationInput = await screen.findByLabelText('Continuation prompt');
    await user.type(continuationInput, 'Restart from a stormy street chase');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'rewrite_seed_prompts')).toBe(true);
    });

    const livePipeline = (createAvPipeline as ReturnType<typeof vi.fn>).mock.results[0]?.value;
    expect(livePipeline).toBeTruthy();
    const resetsBeforeRestart = livePipeline.reset.mock.calls.length;

    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_started' }));
    clientSocket.send(JSON.stringify({
      type: 'seed_prompts_updated',
      reason: 'rewrite',
      prompts: [
        'Restart from a stormy street chase',
        'The chase cuts through a wet neon alley',
      ],
    }));
    clientSocket.send(JSON.stringify({ type: 'rewrite_seed_prompts_complete' }));
    clientSocket.send(JSON.stringify({
      type: 'seed_prompts_reset_applied',
      reason: 'rewrite_during_generation',
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_stream_start',
      total_segments: 2,
      generation_segment_cap: 6,
      loop_generation_enabled: false,
    }));

    await waitFor(() => {
      expect(livePipeline.reset.mock.calls.length).toBeGreaterThan(resetsBeforeRestart);
    });
    expect(livePipeline.markStreamStarting).not.toHaveBeenCalled();
  });

  it('switches the stage to the archived final clip when segment cap is reached', async () => {
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 1,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    expect(screen.queryByText(/Reached max number of segments supported/i)).not.toBeInTheDocument();
    clientSocket.send(JSON.stringify({
      type: 'generation_cap_reached',
      cap_segments: 1,
      generated_segments: 1,
      message: 'Reached max number of segments supported (1). Click Restart to continue.',
    }));

    await waitFor(() => {
      expect(container.querySelectorAll('.gallery-card')).toHaveLength(1);
    });

    expect(container.querySelector('.stage-copy h2')?.textContent).toBe('Test Preset');
    expect(container.querySelector('.gallery-card.is-active')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Restart' })).not.toBeInTheDocument();
    expect(screen.queryByText(/Reached max number of segments supported/i)).not.toBeInTheDocument();
  });

  it('replays completed 30s video replies from the archived blob when available', async () => {
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'segment two',
      source: 'curated',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([4, 5, 6]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(container.querySelectorAll('.gallery-card')).toHaveLength(1);
    });

    const archivedReplayPipeline = (createAvPipeline as ReturnType<typeof vi.fn>).mock.results[1]?.value;
    expect(archivedReplayPipeline).toBeTruthy();
    expect(archivedReplayPipeline.ensurePipeline).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /Test Preset/i }));

    await waitFor(() => {
      expect(container.querySelector('video')?.getAttribute('src')).toBe('blob:mock-url');
    });
    expect(archivedReplayPipeline.ensurePipeline).not.toHaveBeenCalled();
    expect(archivedReplayPipeline.enqueueChunk).not.toHaveBeenCalled();
  });

  it('keeps the archived rollout selected while the next rollout starts on Apple fallback', async () => {
    avPipelineMockState.useNativePlaybackFallback = true;

    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: 'Generate' }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 2,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 2,
      total_segments: 2,
      prompt: 'segment two',
      source: 'curated',
      seed_prompt_index: 1,
    }));
    clientSocket.send(new Uint8Array([4, 5, 6]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 2 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(container.querySelector('.gallery-card.is-active')).not.toBeNull();
    });
    expect(container.querySelector('.stage-copy h2')?.textContent).toBe('Test Preset');

    const promptInput = await screen.findByLabelText('Continuation prompt');
    await user.type(promptInput, 'The hero escapes into the rain');
    await user.click(screen.getByRole('button', { name: 'Rewrite rollout' }));

    clientSocket.send(JSON.stringify({
      type: 'rewrite_seed_prompts_started',
    }));
    clientSocket.send(JSON.stringify({
      type: 'ltx2_stream_start',
      total_segments: 2,
      loop_generation_enabled: false,
    }));
    clientSocket.send(JSON.stringify({
      type: 'media_init',
      mime: 'video/mp4',
      stream_id: 'rewrite_stream',
    }));

    await waitFor(() => {
      expect(container.querySelector('.gallery-card.is-active')).not.toBeNull();
    });
    expect(container.querySelector('.stage-copy h2')?.textContent).toBe('Test Preset');
  });

  it('saves the first completed clip when the user leaves the session without edits', async () => {
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    const { container } = render(<Page />);

    await user.click(await screen.findByRole('button', { name: /Test Preset/i }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 1,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(container.querySelectorAll('.gallery-card')).toHaveLength(1);
    });

    await user.click(screen.getByLabelText('Leave'));

    await waitFor(() => {
      expect(projectStorageMockState.saveProject).toHaveBeenCalled();
    });

    expect(projectStorageMockState.projects).toHaveLength(1);
    expect(projectStorageMockState.clips).toHaveLength(1);
    expect(projectStorageMockState.clips[0].projectId).toBe(projectStorageMockState.projects[0].id);
    expect(projectStorageMockState.clips[0].blob).toBeInstanceOf(Blob);
    expect(projectStorageMockState.clips[0].label).toBe('Test Preset');
  });

  it('starts the next project on the same websocket after New project and keeps both archives', async () => {
    const sockets: any[] = [];
    const outbound: any[] = [];
    server.on('connection', (socket) => {
      sockets.push(socket);
      socket.on('message', (rawMessage) => {
        outbound.push(JSON.parse(rawMessage as string));
      });
    });

    let releaseFirstSave = () => {};
    let firstSaveBlocked = false;
    projectStorageMockState.saveProject.mockImplementation(async (project: any, clips: any[]) => {
      if (!firstSaveBlocked) {
        firstSaveBlocked = true;
        await new Promise<void>((resolve) => {
          releaseFirstSave = resolve;
        });
      }
      await projectStorageMockState.commitSave(project, clips);
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: /Test Preset/i }));

    await waitFor(() => {
      expect(sockets[0]).toBeTruthy();
    });

    sockets[0].send(JSON.stringify({
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 90,
    }));
    sockets[0].send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 1,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    sockets[0].send(new Uint8Array([1, 2, 3]).buffer);
    sockets[0].send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    sockets[0].send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(screen.getByLabelText('Leave')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'New project' }));

    expect(screen.getByLabelText('Leave')).toBeInTheDocument();
    expect(outbound.some((message) => message.type === 'end_project_keep_session')).toBe(true);
    expect(sockets).toHaveLength(1);

    sockets[0].send(JSON.stringify({ type: 'ltx2_stream_complete' }));
    sockets[0].send(JSON.stringify({ type: 'project_idle' }));

    releaseFirstSave();

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Test Preset/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /Test Preset/i }));

    await waitFor(() => {
      expect(outbound.some((message) => message.type === 'project_init_v1')).toBe(true);
    });

    expect(sockets).toHaveLength(1);

    sockets[0].send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 1,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    sockets[0].send(new Uint8Array([4, 5, 6]).buffer);
    sockets[0].send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    sockets[0].send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(screen.getByLabelText('Leave')).toBeInTheDocument();
    });

    await user.click(screen.getByLabelText('Leave'));

    await waitFor(() => {
      expect(projectStorageMockState.projects).toHaveLength(2);
    });

    expect(new Set(projectStorageMockState.projects.map((project) => project.id)).size).toBe(2);
    expect(new Set(projectStorageMockState.clips.map((clip) => clip.projectId)).size).toBe(2);
    expect(outbound.filter((message) => message.type === 'session_init_v2')).toHaveLength(1);
    expect(outbound.filter((message) => message.type === 'project_init_v1')).toHaveLength(1);
    const clipCountsByProject = projectStorageMockState.clips.reduce((counts: Record<string, number>, clip) => {
      counts[clip.projectId] = (counts[clip.projectId] || 0) + 1;
      return counts;
    }, {});
    expect(Object.values(clipCountsByProject).sort()).toEqual([1, 1]);
  });

  it('prunes older archives and retries when project save hits storage pressure', async () => {
    projectStorageMockState.projects = [
      { id: 'old-1', label: 'Old 1', presetId: '', originalLabel: 'Old 1', createdAt: 400, lastThumbnail: null, promptEvents: [] },
      { id: 'old-2', label: 'Old 2', presetId: '', originalLabel: 'Old 2', createdAt: 300, lastThumbnail: null, promptEvents: [] },
      { id: 'old-3', label: 'Old 3', presetId: '', originalLabel: 'Old 3', createdAt: 200, lastThumbnail: null, promptEvents: [] },
      { id: 'old-4', label: 'Old 4', presetId: '', originalLabel: 'Old 4', createdAt: 100, lastThumbnail: null, promptEvents: [] },
    ];
    projectStorageMockState.clips = projectStorageMockState.projects.map((project, index) => ({
      id: `clip-${index + 1}`,
      projectId: project.id,
      label: project.label,
      prompt: project.label,
      mime: 'video/mp4',
      blob: new Blob([String(index + 1)], { type: 'video/mp4' }),
      createdAt: project.createdAt,
    }));

    let failedOnce = false;
    projectStorageMockState.saveProject.mockImplementation(async (project: any, clips: any[]) => {
      if (!failedOnce) {
        failedOnce = true;
        throw new DOMException('Quota exceeded', 'QuotaExceededError');
      }
      await projectStorageMockState.commitSave(project, clips);
    });

    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: /Test Preset/i }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 1,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([7, 8, 9]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(screen.getByLabelText('Leave')).toBeInTheDocument();
    });

    await user.click(screen.getByLabelText('Leave'));

    await waitFor(() => {
      expect(projectStorageMockState.saveProject).toHaveBeenCalledTimes(2);
    });

    expect(projectStorageMockState.deleteProject).toHaveBeenCalled();
    expect(projectStorageMockState.projects.some((project) => project.id === 'old-4')).toBe(false);
    expect(projectStorageMockState.projects.some((project) => project.label === 'Test Preset')).toBe(true);
  });

  it('preserves the existing archive when later saves fall back to metadata only', async () => {
    let clientSocket: any;
    server.on('connection', (socket) => {
      clientSocket = socket;
    });

    const user = userEvent.setup();
    render(<Page />);

    await user.click(await screen.findByRole('button', { name: /Test Preset/i }));

    await waitFor(() => {
      expect(clientSocket).toBeTruthy();
    });

    clientSocket.send(JSON.stringify({
      type: 'ltx2_segment_start',
      segment_idx: 1,
      total_segments: 1,
      prompt: 'segment one',
      source: 'curated',
      seed_prompt_index: 0,
    }));
    clientSocket.send(new Uint8Array([1, 2, 3]).buffer);
    clientSocket.send(JSON.stringify({ type: 'media_segment_complete', segment_idx: 1 }));
    clientSocket.send(JSON.stringify({ type: 'ltx2_stream_complete' }));

    await waitFor(() => {
      expect(projectStorageMockState.projects).toHaveLength(1);
      expect(projectStorageMockState.clips).toHaveLength(1);
    });

    const projectId = projectStorageMockState.projects[0].id;
    projectStorageMockState.saveProject.mockImplementation(async () => {
      throw new DOMException('Quota exceeded', 'QuotaExceededError');
    });

    await user.click(screen.getByLabelText('Leave'));

    await waitFor(() => {
      expect(projectStorageMockState.saveProjectMetadata).toHaveBeenCalled();
    });

    expect(projectStorageMockState.projects).toHaveLength(1);
    expect(projectStorageMockState.projects[0].id).toBe(projectId);
    expect(projectStorageMockState.clips.filter((clip) => clip.projectId === projectId)).toHaveLength(1);
  });

});
