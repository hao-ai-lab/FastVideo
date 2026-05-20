import { applySessionUiMessage } from '../lib/sessionState';
import { createManagedStore, type ManagedStore } from './createManagedStore';

export interface SessionState {
  connected: boolean;
  connecting: boolean;
  sessionStarted: boolean;
  sessionTimeout: number | null;
  timeLeft: number | null;
  queuePosition: number;
  gpuAssigned: boolean;
  enhancementEnabled: boolean;
  promptExtensionError: string;
  autoExtensionEnabled: boolean;
  autoExtensionTimeoutHint: string;
  loopGenerationEnabled: boolean;
  generationPaused: boolean;
  livePromptDraft: string;
  sessionNotice: string;
  generationCapReached: boolean;
  generationSegmentCap: number;
  generatedSegmentCount: number;
  preservePlaybackOnClose: boolean;
  livePromptRewriteMode: boolean;
  sessionExpired: boolean;
  projectResetPending: boolean;
}

const DEFAULT_SESSION_STATE: SessionState = {
  connected: false,
  connecting: false,
  sessionStarted: false,
  sessionTimeout: null,
  timeLeft: null,
  queuePosition: 0,
  gpuAssigned: false,
  enhancementEnabled: true,
  promptExtensionError: '',
  autoExtensionEnabled: false,
  autoExtensionTimeoutHint: '',
  loopGenerationEnabled: false,
  generationPaused: false,
  livePromptDraft: '',
  sessionNotice: '',
  generationCapReached: false,
  generationSegmentCap: 0,
  generatedSegmentCount: 0,
  preservePlaybackOnClose: false,
  livePromptRewriteMode: false,
  sessionExpired: false,
  projectResetPending: false,
};

export type SessionStore = ManagedStore<SessionState> & {
  reset: (overrides?: Partial<SessionState>) => SessionState;
  applyServerUiMessage: (data: unknown) => SessionState;
};

export function createSessionStore(
  initialState: Partial<SessionState> = {},
): SessionStore {
  const store = createManagedStore<SessionState>({
    ...DEFAULT_SESSION_STATE,
    ...initialState,
  } as SessionState);

  return {
    ...store,
    reset(overrides: Partial<SessionState> = {}) {
      return store.set({
        ...DEFAULT_SESSION_STATE,
        ...(initialState as SessionState),
        ...overrides,
      } as SessionState);
    },
    applyServerUiMessage(data: unknown) {
      return store.update((state) => applySessionUiMessage(state, data));
    },
  };
}

export { DEFAULT_SESSION_STATE };
