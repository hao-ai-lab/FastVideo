import {
  prependPromptEvent,
  updatePromptEvent,
} from '../lib/promptEvents';
import { createManagedStore, type ManagedStore } from './createManagedStore';

export interface RewriteState {
  rewritingSeedPrompts: boolean;
  promptEvents: Record<string, unknown>[];
  lastPromptSource: string;
  pendingSegmentSource: Record<string, unknown> | null;
}

const DEFAULT_REWRITE_STATE: RewriteState = {
  rewritingSeedPrompts: false,
  promptEvents: [],
  lastPromptSource: '',
  pendingSegmentSource: null,
};

export interface SegmentPromptSourcePayload {
  source?: string;
  seed_prompt_index?: number;
  loop_iteration?: number;
  prompt_id?: string;
}

export type RewriteStore = ManagedStore<RewriteState> & {
  reset: (overrides?: Partial<RewriteState>) => RewriteState;
  resetSessionState: () => RewriteState;
  trackPromptEvent: (
    promptId: string,
    update: Record<string, unknown>,
  ) => RewriteState;
  addPromptEvent: (
    event: Record<string, unknown>,
  ) => RewriteState;
  applySegmentPromptSource: (
    payload: SegmentPromptSourcePayload | null,
  ) => RewriteState;
};

export function createRewriteStore(
  initialState: Partial<RewriteState> = {},
): RewriteStore {
  const store = createManagedStore<RewriteState>({
    ...DEFAULT_REWRITE_STATE,
    ...initialState,
  } as RewriteState);

  return {
    ...store,
    reset(overrides: Partial<RewriteState> = {}) {
      return store.set({
        ...DEFAULT_REWRITE_STATE,
        ...initialState,
        ...overrides,
      } as RewriteState);
    },
    resetSessionState() {
      return store.patch({
        rewritingSeedPrompts: false,
        promptEvents: [],
        lastPromptSource: '',
        pendingSegmentSource: null,
      });
    },
    trackPromptEvent(
      promptId: string,
      update: Record<string, unknown>,
    ) {
      return store.update((state) => ({
        ...state,
        promptEvents: updatePromptEvent(
          state.promptEvents,
          promptId,
          update,
        ),
      }));
    },
    addPromptEvent(event: Record<string, unknown>) {
      return store.update((state) => ({
        ...state,
        promptEvents: prependPromptEvent(
          state.promptEvents,
          event,
        ),
      }));
    },
    applySegmentPromptSource(
      payload: SegmentPromptSourcePayload | null,
    ) {
      return store.patch({
        lastPromptSource: payload?.source || '',
        pendingSegmentSource: {
          source: payload?.source || 'unknown',
          seedPromptIndex: Number.isInteger(
            payload?.seed_prompt_index,
          )
            ? (payload!.seed_prompt_index as number)
            : null,
          loopIteration: Number.isInteger(payload?.loop_iteration)
            ? (payload!.loop_iteration as number)
            : null,
          promptId: payload?.prompt_id || null,
        },
      });
    },
  };
}

export { DEFAULT_REWRITE_STATE };
