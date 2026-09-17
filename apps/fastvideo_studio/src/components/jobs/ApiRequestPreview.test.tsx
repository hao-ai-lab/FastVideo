import * as React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import ApiRequestPreview from './ApiRequestPreview';
import { createJobRequest } from '@/lib/apiRequest';
import { getApiBaseUrl } from '@/lib/api';

vi.mock('@/lib/api', () => ({ getApiBaseUrl: vi.fn(() => 'http://localhost:8189/api') }));

afterEach(() => vi.unstubAllGlobals());

describe('ApiRequestPreview', () => {
  it('refreshes edited form values and copies without submitting a request', async () => {
    const user = userEvent.setup();
    const clipboard = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const initial = createJobRequest({ model_id: 'wan/test', prompt: 'first prompt', num_frames: 81 });
    const { rerender } = render(<ApiRequestPreview request={initial} />);

    await user.click(screen.getByRole('button', { name: 'Show cURL' }));
    expect(screen.getByLabelText('cURL command')).toHaveTextContent('first prompt');
    const next = createJobRequest({ model_id: 'wan/test', prompt: 'edited prompt', num_frames: 60 });
    rerender(<ApiRequestPreview request={next} />);
    expect(screen.getByRole('status')).toHaveTextContent('Settings changed');
    expect(screen.getByRole('button', { name: 'Copy cURL' })).toBeDisabled();

    vi.mocked(getApiBaseUrl).mockReturnValueOnce('https://new-backend.test/api');
    await user.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(screen.getByLabelText('cURL command')).toHaveTextContent('edited prompt');
    expect(screen.getByLabelText('cURL command')).toHaveTextContent('"num_frames": 60');
    expect(screen.getByLabelText('cURL command')).toHaveTextContent('https://new-backend.test/api');
    await user.click(screen.getByRole('button', { name: 'Copy cURL' }));
    await waitFor(() => expect(clipboard).toHaveBeenCalledWith(screen.getByLabelText('cURL command').textContent));
    expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('provides a manual copy fallback if clipboard access fails', async () => {
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, 'writeText').mockRejectedValue(new Error('denied'));
    render(<ApiRequestPreview request={createJobRequest({ model_id: 'wan/test', prompt: 'test' })} />);
    await user.click(screen.getByRole('button', { name: 'Show cURL' }));
    await user.click(screen.getByRole('button', { name: 'Copy cURL' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('copy it manually');
  });
});
