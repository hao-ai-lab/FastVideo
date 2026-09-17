import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import { createJob, getModels } from '@/lib/api';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';
import { defaultOptionsStore } from '@/stores/defaultOptions';

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(), updateJob: vi.fn(), getModels: vi.fn(),
  getDatasets: vi.fn().mockResolvedValue([]), uploadImage: vi.fn(), uploadMedia: vi.fn(),
  getSettings: vi.fn(), updateSettings: vi.fn(),
}));

beforeEach(() => {
  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue([{ id: 'wan/t2v-1.3b', label: 'Wan T2V' }]);
  vi.mocked(createJob).mockResolvedValue({ id: 'test-job' } as never);
});

describe('inference options', () => {
  it('includes the last typed numeric value when Create Job causes blur', async () => {
    const user = userEvent.setup();
    render(<CreateJobModal isOpen onClose={vi.fn()} onSuccess={vi.fn()} jobType="inference" workloadType="t2v" />);
    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    await user.type(screen.getByLabelText('Prompt'), 'a test video');
    const frames = screen.getByRole('spinbutton', { name: 'Frames value' });
    await user.clear(frames);
    await user.type(frames, '60');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    expect(vi.mocked(createJob).mock.calls[0][0]).toMatchObject({ num_frames: 60 });
    expect(screen.getByText('Output').closest('details')).toHaveAttribute('open');
    expect(screen.getByText('Distributed').closest('details')).not.toHaveAttribute('open');
  });

  it('keeps both the slider and numeric entry disabled in read-only views', async () => {
    render(<CreateJobModal isOpen readOnly onClose={vi.fn()} onSuccess={vi.fn()} jobType="inference" workloadType="t2v" />);
    const frames = screen.getByRole('spinbutton', { name: 'Frames value' });
    expect(frames).toBeDisabled();
    const slider = screen.getByRole('slider', { name: 'Frames' });
    expect(slider).toHaveAttribute('data-disabled');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(frames).toHaveValue(81);
    expect(screen.queryByRole('button', { name: 'Create Job' })).not.toBeInTheDocument();
  });

  it('omits video-only controls from image inference', async () => {
    render(<CreateJobModal isOpen onClose={vi.fn()} onSuccess={vi.fn()} jobType="inference" workloadType="t2i" />);
    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    expect(screen.queryByRole('spinbutton', { name: 'Frames value' })).not.toBeInTheDocument();
    expect(screen.queryByRole('spinbutton', { name: 'FPS value' })).not.toBeInTheDocument();
  });
});
