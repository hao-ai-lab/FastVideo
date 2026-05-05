import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { toast } from 'sonner';

import SpeechToTextButton from './SpeechToTextButton';
import {
  getWhisperApiKey,
  startWhisperStt,
} from '@/lib/whisperStt';

vi.mock('@/lib/whisperStt', () => ({
  getWhisperApiKey: vi.fn(),
  startWhisperStt: vi.fn(),
}));

vi.mock('sonner', () => ({
  toast: {
    error: vi.fn(),
    dismiss: vi.fn(),
  },
}));

describe('SpeechToTextButton', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows a toast error when speech input is not configured', async () => {
    const user = userEvent.setup();
    vi.mocked(getWhisperApiKey).mockReturnValue('');

    render(<SpeechToTextButton onTranscript={vi.fn()} />);

    await user.click(
      screen.getByRole('button', { name: 'Start speech input' }),
    );

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        'Speech input is not configured.',
        { id: 'speech-to-text-error' },
      );
    });
    expect(startWhisperStt).not.toHaveBeenCalled();
  });

  it('transcribes speech when stop is clicked', async () => {
    const user = userEvent.setup();
    const onTranscript = vi.fn();
    const stop = vi.fn().mockResolvedValue('hello world');

    vi.mocked(getWhisperApiKey).mockReturnValue('sk-key');
    vi.mocked(startWhisperStt).mockResolvedValue({
      active: true,
      stop,
    });

    render(<SpeechToTextButton onTranscript={onTranscript} />);

    await user.click(
      screen.getByRole('button', { name: 'Start speech input' }),
    );

    await waitFor(() => {
      expect(startWhisperStt).toHaveBeenCalledTimes(1);
    });

    await user.click(
      screen.getByRole('button', { name: 'Stop speech input' }),
    );

    await waitFor(() => {
      expect(stop).toHaveBeenCalledTimes(1);
      expect(onTranscript).toHaveBeenCalledWith('hello world');
    });
  });

  it('does not call onTranscript when stop returns no text', async () => {
    const user = userEvent.setup();
    const onTranscript = vi.fn();
    const stop = vi.fn().mockResolvedValue(undefined);

    vi.mocked(getWhisperApiKey).mockReturnValue('sk-key');
    vi.mocked(startWhisperStt).mockResolvedValue({
      active: true,
      stop,
    });

    render(<SpeechToTextButton onTranscript={onTranscript} />);

    await user.click(
      screen.getByRole('button', { name: 'Start speech input' }),
    );

    await waitFor(() => {
      expect(startWhisperStt).toHaveBeenCalledTimes(1);
    });

    await user.click(
      screen.getByRole('button', { name: 'Stop speech input' }),
    );

    await waitFor(() => {
      expect(stop).toHaveBeenCalledTimes(1);
    });
    expect(onTranscript).not.toHaveBeenCalled();
  });
});
