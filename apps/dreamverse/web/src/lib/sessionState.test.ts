import { describe, expect, it } from 'vitest';

import { applySessionUiMessage } from './sessionState';

interface SessionUiState {
  autoExtensionEnabled: boolean;
  gpuAssigned: boolean;
  queuePosition: number;
  sessionTimeout: number | null;
  timeLeft: number | null;
}

const initialState: SessionUiState = {
  autoExtensionEnabled: false,
  gpuAssigned: false,
  queuePosition: 0,
  sessionTimeout: null,
  timeLeft: null,
};

describe('applySessionUiMessage', () => {
  it('updates queue position from queue_status', () => {
    const next = applySessionUiMessage(initialState, {
      type: 'queue_status',
      position: 3,
    });

    expect(next.queuePosition).toBe(3);
    expect(next.gpuAssigned).toBe(false);
  });

  it('updates state on gpu_assigned', () => {
    const next = applySessionUiMessage(initialState, {
      type: 'gpu_assigned',
      gpu_id: 0,
      session_timeout: 120,
    });

    expect(next.gpuAssigned).toBe(true);
    expect(next.queuePosition).toBe(0);
    expect(next.sessionTimeout).toBe(120);
    expect(next.timeLeft).toBe(120);
  });

  it('updates auto extension toggle', () => {
    const next = applySessionUiMessage(initialState, {
      type: 'auto_extension_updated',
      enabled: 1,
    });

    expect(next.autoExtensionEnabled).toBe(true);
  });

  it('handles session timeout state', () => {
    const state: SessionUiState = {
      ...initialState,
      gpuAssigned: true,
      timeLeft: 15,
    };
    const next = applySessionUiMessage(state, {
      type: 'session_timeout',
    });

    expect(next.gpuAssigned).toBe(false);
    expect(next.timeLeft).toBe(0);
  });
});
