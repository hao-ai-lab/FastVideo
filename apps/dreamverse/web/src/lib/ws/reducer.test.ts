import { describe, expect, it } from 'vitest';

import { resolveSessionErrorMessage } from './reducer';

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
