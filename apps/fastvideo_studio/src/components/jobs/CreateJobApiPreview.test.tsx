import * as React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import { createJob, getDatasets, getModels, updateJob } from '@/lib/api';
import { createJobRequest, requestToCurl, updateJobRequest } from '@/lib/apiRequest';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';
import { defaultOptionsStore } from '@/stores/defaultOptions';

vi.mock('@/lib/api', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/lib/api')>(),
  createJob: vi.fn(),
  updateJob: vi.fn(),
  getModels: vi.fn(),
  getDatasets: vi.fn(),
  getApiBaseUrl: vi.fn(() => 'https://configured.test/api'),
}));

beforeEach(() => {
  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue([{ id: 'wan/test', label: 'Wan' }]);
  vi.mocked(getDatasets).mockResolvedValue([]);
  vi.mocked(createJob).mockResolvedValue({ id: 'new-job' } as never);
  vi.mocked(updateJob).mockResolvedValue({});
});

describe('CreateJobModal API integration', () => {
  it.each([false, true])('exports exactly the submitted payload (edit=%s)', async (editing) => {
    const user = userEvent.setup();
    render(<CreateJobModal
      isOpen
      onClose={vi.fn()}
      onSuccess={vi.fn()}
      jobType="inference"
      workloadType="t2v"
      editingJob={editing ? { id: 'pending-job', model_id: 'wan/test', prompt: 'old prompt' } : undefined}
    />);
    await screen.findByRole('option', { name: 'Wan (wan/test)' });
    await user.click(screen.getByRole('button', { name: 'Show cURL' }));
    await user.clear(screen.getByLabelText('Prompt'));
    await user.type(screen.getByLabelText('Prompt'), 'revised prompt');
    await user.click(screen.getByRole('button', { name: 'Refresh' }));
    const exported = screen.getByLabelText('cURL command').textContent;
    expect(createJob).not.toHaveBeenCalled();
    expect(updateJob).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: editing ? 'Save Changes' : 'Create Job' }));
    if (editing) {
      await waitFor(() => expect(updateJob).toHaveBeenCalledTimes(1));
      const [id, payload] = vi.mocked(updateJob).mock.calls[0];
      expect(exported).toBe(requestToCurl('https://configured.test/api', updateJobRequest(id, payload)));
    } else {
      await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
      const [payload] = vi.mocked(createJob).mock.calls[0];
      expect(exported).toBe(requestToCurl('https://configured.test/api', createJobRequest(payload)));
    }
  });
});
