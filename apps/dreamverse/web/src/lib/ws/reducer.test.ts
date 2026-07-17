import { describe, expect, it } from 'vitest';

import { applyNormalizedSocketEvent, resolveSessionErrorMessage } from './reducer';
import { createSessionStore } from '../../stores/session';
import { createRewriteStore } from '../../stores/rewrite';
import { createStreamStore } from '../../stores/stream';
import { createUiStore } from '../../stores/ui';
import { createPromptWindowStore } from '../../stores/promptWindow';

describe('resolveSessionErrorMessage', () => {
  it('returns a dedicated message for IP session limit errors', () => {
    expect(resolveSessionErrorMessage({
      error_code: 'ip_session_limit',
      message: 'ignored',
    })).toBe(
      'Only one active websocket session is allowed per IP. '
      + 'Close the other session and click Run to retry.',
    );
  });

  it('falls back to payload message for other server errors', () => {
    expect(resolveSessionErrorMessage({
      message: 'Backend replica unavailable. Rejoin session.',
    })).toBe('Backend replica unavailable. Rejoin session.');
  });
});

function buildContext(overrides: Record<string, unknown> = {}) {
  const sessionStore = createSessionStore();
  const rewriteStore = createRewriteStore();
  const streamStore = createStreamStore();
  const uiStore = createUiStore();
  const promptWindowStore = createPromptWindowStore();
  const avPipeline = {
    reset: () => {},
    setStreamCompleted: () => {},
    noteSegmentInit: () => {},
    noteSegmentComplete: () => {},
    maybeStartPlayback: () => {},
    ensurePipeline: async () => {},
  };
  return {
    sessionStore,
    promptWindowStore,
    rewriteStore,
    streamStore,
    uiStore,
    avPipeline,
    tick: async () => {},
    defaultAvMime: 'video/mp4',
    fixedRewriteModel: 'model',
    parseLatencyMs: () => null,
    formatPromptWindowEventText: () => '',
    makePromptId: () => 'generated-id',
    buildClipLabel: () => 'clip',
    startSessionCountdown: () => {},
    clearCountdownInterval: () => {},
    resetTtffTimer: () => {},
    startTtffTimer: () => {},
    preserveArchivedPlaybackSelection: false,
    finalizeStreamCompletion: async () => {},
    ...overrides,
  };
}

describe('steering generatingNextScene flow', () => {
  it('sets generatingNextScene on prompt/sources_resumed in manual mode', async () => {
    const context = buildContext();
    await applyNormalizedSocketEvent(
      { type: 'prompt/sources_resumed', payload: { segment_idx: 2 } },
      context,
    );
    expect(context.sessionStore.get().generatingNextScene).toBe(true);
    expect(context.sessionStore.get().waitingForSegmentPrompt).toBe(false);
  });

  it('does NOT set generatingNextScene on session/auto_extension_updated', async () => {
    const context = buildContext();
    await applyNormalizedSocketEvent(
      { type: 'session/auto_extension_updated', payload: { enabled: true } },
      context,
    );
    expect(context.sessionStore.get().generatingNextScene).toBe(false);
    expect(context.sessionStore.get().waitingForSegmentPrompt).toBe(false);
  });

  it('clears generatingNextScene when segment media arrives', async () => {
    const context = buildContext();
    context.sessionStore.patch({ generatingNextScene: true });
    await applyNormalizedSocketEvent(
      {
        type: 'stream/media_init',
        payload: { segment_idx: 2, stream_id: 's', mime: 'video/mp4' },
      },
      context,
    );
    expect(context.sessionStore.get().generatingNextScene).toBe(false);
  });

  it('clears generatingNextScene and returns to waiting on prompt/sources_blocked', async () => {
    const context = buildContext();
    context.sessionStore.patch({ generatingNextScene: true });
    await applyNormalizedSocketEvent(
      { type: 'prompt/sources_blocked', payload: { segment_idx: 3 } },
      context,
    );
    expect(context.sessionStore.get().generatingNextScene).toBe(false);
    expect(context.sessionStore.get().waitingForSegmentPrompt).toBe(true);
  });

  it('clears generatingNextScene when the opening prompt falls back', async () => {
    const context = buildContext();
    context.sessionStore.patch({ generatingNextScene: true });
    await applyNormalizedSocketEvent(
      {
        type: 'prompt/fallback_used',
        payload: { prompt_id: 'p1', prompt: '', source: 'user_enhancement_failed' },
      },
      context,
    );
    expect(context.sessionStore.get().generatingNextScene).toBe(false);
    expect(context.sessionStore.get().waitingForSegmentPrompt).toBe(true);
  });
});

describe('opening prompt id tracking', () => {
  it('routes prompt lifecycle updates to the frontend-recorded opening event', async () => {
    // The frontend records the opening scene under its own prompt id and sends it
    // as initial_rollout_prompt_id; the backend echoes it in status updates.
    const context = buildContext();
    context.rewriteStore.addPromptEvent({
      promptId: 'opening-id',
      status: 'rewrite_requested',
      source: 'user_rewrite',
      text: 'a castle at dawn',
      steeringUserPrompt: true,
      rawText: 'a castle at dawn',
    });

    await applyNormalizedSocketEvent(
      { type: 'prompt/enhancing', payload: { prompt_id: 'opening-id' } },
      context,
    );
    let opening = (context.rewriteStore.get().promptEvents as Record<string, any>[])
      .find((e) => e.promptId === 'opening-id');
    expect(opening?.status).toBe('enhancing');

    await applyNormalizedSocketEvent(
      {
        type: 'prompt/fallback_used',
        payload: { prompt_id: 'opening-id', prompt: '', source: 'user_enhancement_failed' },
      },
      context,
    );
    opening = (context.rewriteStore.get().promptEvents as Record<string, any>[])
      .find((e) => e.promptId === 'opening-id');
    expect(opening?.status).toBe('ready_fallback');
    // A failed opening is dropped from the steering scene list instead of
    // lingering as a ghost "Scene 1".
    expect(opening?.steeringFailed).toBe(true);
  });

  it('marks a prompt-scoped session/error (e.g. safety block) as steeringFailed', async () => {
    const context = buildContext();
    context.rewriteStore.addPromptEvent({
      promptId: 'blocked-id',
      status: 'queued',
      source: 'user_raw',
      text: 'a blocked prompt',
      steeringUserPrompt: true,
      rawText: 'a blocked prompt',
    });

    await applyNormalizedSocketEvent(
      {
        type: 'session/error',
        payload: { message: 'Prompt blocked by safety filter.', prompt_id: 'blocked-id' },
      },
      context,
    );
    const blocked = (context.rewriteStore.get().promptEvents as Record<string, any>[])
      .find((e) => e.promptId === 'blocked-id');
    expect(blocked?.steeringFailed).toBe(true);
  });
});
