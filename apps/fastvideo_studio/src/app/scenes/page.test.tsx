import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ScenesPage from './page';
import { getJobsList } from '@/lib/api';
import { makeJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getJobsList: vi.fn(),
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
}));

vi.mock('@/components/jobs/CreateJobModal', () => ({
  default: (props: { readOnly?: boolean; editingJob?: { name?: string } | null; onClose: () => void }) => (
    <div role="dialog" data-readonly={String(!!props.readOnly)}>
      {props.editingJob?.name}
      <button onClick={props.onClose}>close</button>
    </div>
  ),
}));

const wolf = (n: number, over: Parameters<typeof makeJob>[0] = {}) =>
  makeJob({
    id: `wolf-${n}`,
    name: `wolf-lunch-clip-${String(n).padStart(2, '0')}`,
    created_at: n,
    num_frames: 240,
    fps: 24,
    prompt: [
      'detailed_description:',
      `[Shot 1] Wide shot. Clip ${n} opens.`,
      `[Shot 2] At 00:02.000, the shot cuts to a close-up shot. (S1) <d>[English] Line from clip ${n}.</d>`,
    ].join('\n'),
    ...over,
  });

beforeEach(() => {
  vi.mocked(getJobsList).mockReset();
});

describe('ScenesPage', () => {
  it('groups clips into a scene and lists every shot in order across them', async () => {
    vi.mocked(getJobsList).mockResolvedValue([wolf(2), wolf(3), wolf(1)]);
    render(<ScenesPage />);

    expect(await screen.findByRole('heading', { name: 'wolf-lunch' })).toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledWith('inference');
    expect(screen.getByText(/3 clips · 6 shots · about 0:30/)).toBeInTheDocument();

    const rows = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent);
    expect(rows).toEqual(['wolf-lunch-clip-01', 'wolf-lunch-clip-02', 'wolf-lunch-clip-03']);
    for (const n of [1, 2, 3]) {
      expect(screen.getByText(new RegExp(`S1: “Line from clip ${n}\\.”`))).toBeInTheDocument();
    }
  });

  it('numbers shots straight through the scene, not per clip', async () => {
    vi.mocked(getJobsList).mockResolvedValue([wolf(1), wolf(2)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const second = screen.getByLabelText('Shots in wolf-lunch-clip-02');
    const badges = [...second.querySelectorAll('span.rounded-full.bg-secondary')].map((b) => b.textContent);
    expect(badges.slice(0, 2)).toEqual(['3', '4']);
  });

  it('puts scene-wide times on each shot', async () => {
    vi.mocked(getJobsList).mockResolvedValue([wolf(1), wolf(2)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const second = screen.getByLabelText('Shots in wolf-lunch-clip-02');
    // clip 2 starts at 10 s (clip 1 is 240 frames at 24 fps), and its second shot cuts in 2 s later
    expect(within(second).getByText('00:10.000')).toBeInTheDocument();
    expect(within(second).getByText('00:12.000')).toBeInTheDocument();
  });

  it('switches between scenes', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([
      wolf(1),
      makeJob({ id: 'r1', name: 'rooftop-1', created_at: 99, prompt: '[Shot 1] Wide shot. A roof.' }),
    ]);
    render(<ScenesPage />);

    expect(await screen.findByRole('heading', { name: 'rooftop' })).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Scene'), 'wolf-lunch');
    expect(await screen.findByRole('heading', { name: 'wolf-lunch' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'rooftop' })).not.toBeInTheDocument();
  });

  it('hides the scene picker when there is only one scene', async () => {
    vi.mocked(getJobsList).mockResolvedValue([wolf(1), wolf(2)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });
    expect(screen.queryByLabelText('Scene')).not.toBeInTheDocument();
  });

  it('shows status counts and marks a continuation clip', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      wolf(1, { status: 'completed' }),
      wolf(2, { status: 'pending', prompt: '[Shot 1] Close-up shot. Continuing the same shot on <Subject 1>, still speaking.' }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });
    expect(screen.getAllByText('1 completed').length).toBeGreaterThan(0);
    expect(screen.getAllByText('1 pending').length).toBeGreaterThan(0);
    expect(screen.getByText('continues previous shot')).toBeInTheDocument();
  });

  it('previews a completed clip on demand', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([
      wolf(1, { status: 'completed', output_path: '/out/a.mp4' }),
      wolf(2, { status: 'pending' }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    expect(screen.getAllByRole('button', { name: 'Preview' })).toHaveLength(1);
    expect(screen.queryByLabelText(/Generated video/)).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Preview' }));
    expect(screen.getByLabelText(/Generated video for wolf-lunch-clip-01/)).toHaveAttribute(
      'src',
      'http://test.local/api/jobs/wolf-1/video',
    );
    await user.click(screen.getByRole('button', { name: 'Hide video' }));
    expect(screen.queryByLabelText(/Generated video/)).not.toBeInTheDocument();
  });

  it('opens a pending clip for editing and a finished one read-only', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([wolf(1, { status: 'completed' }), wolf(2, { status: 'pending' })]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    let dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('data-readonly', 'false');
    expect(dialog).toHaveTextContent('wolf-lunch-clip-02');
    await user.click(within(dialog).getByRole('button', { name: 'close' }));

    await user.click(screen.getByRole('button', { name: 'View' }));
    dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('data-readonly', 'true');
    expect(dialog).toHaveTextContent('wolf-lunch-clip-01');
  });

  it('timeline segments jump to their clip', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([wolf(1), wolf(2)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const scroll = vi.spyOn(Element.prototype, 'scrollIntoView');
    await user.click(screen.getByRole('button', { name: /Go to clip 2: wolf-lunch-clip-02/ }));
    expect(scroll).toHaveBeenCalled();
    expect((scroll.mock.instances.at(-1) as unknown as Element).id).toBe('clip-wolf-2');
  });

  it('shows an empty state when there are no jobs', async () => {
    vi.mocked(getJobsList).mockResolvedValue([]);
    render(<ScenesPage />);
    expect(await screen.findByText(/No inference jobs yet/)).toBeInTheDocument();
  });

  it('shows an error with a retry that reloads', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockRejectedValueOnce(new Error('API down')).mockResolvedValue([wolf(1)]);
    render(<ScenesPage />);

    expect(await screen.findByRole('alert')).toHaveTextContent('API down');
    await user.click(screen.getByRole('button', { name: 'Try Again' }));
    expect(await screen.findByRole('heading', { name: 'wolf-lunch' })).toBeInTheDocument();
    await waitFor(() => expect(getJobsList).toHaveBeenCalledTimes(2));
  });

  it('says so when a clip has no shots', async () => {
    vi.mocked(getJobsList).mockResolvedValue([makeJob({ id: 'e', name: 'empty-1', prompt: '' })]);
    render(<ScenesPage />);
    expect(await screen.findByText(/prompt has no shots/)).toBeInTheDocument();
  });

  describe('re-runs of the same clip', () => {
    it('shows one clip with a take picker instead of a row per re-run', async () => {
      vi.mocked(getJobsList).mockResolvedValue([
        wolf(1, { id: 't1', created_at: 1, status: 'completed' }),
        wolf(1, { id: 't2', created_at: 2, status: 'failed' }),
        wolf(1, { id: 't3', created_at: 3, status: 'completed' }),
        wolf(2, { id: 'n', created_at: 4, status: 'pending' }),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      expect(screen.getAllByRole('heading', { level: 3 })).toHaveLength(2);
      expect(screen.getByText(/2 clips · 4 shots/)).toBeInTheDocument();
      const picker = screen.getByLabelText('Take for clip 1') as HTMLSelectElement;
      expect(picker.options).toHaveLength(3);
      // newest completed take is shown by default (not the failed one in the middle)
      expect(picker.value).toBe('t3');
      expect(screen.queryByLabelText('Take for clip 2')).not.toBeInTheDocument();
    });

    it('switching takes changes the clip shown and its status', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([
        wolf(1, { id: 't1', created_at: 1, status: 'completed', prompt: '[Shot 1] Wide shot. First take.' }),
        wolf(1, { id: 't2', created_at: 2, status: 'failed', prompt: '[Shot 1] Wide shot. Second take.' }),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(screen.getByText('First take.')).toBeInTheDocument();

      await user.selectOptions(screen.getByLabelText('Take for clip 1'), 't2');
      expect(await screen.findByText('Second take.')).toBeInTheDocument();
      expect(screen.queryByText('First take.')).not.toBeInTheDocument();
      expect(screen.getAllByText('1 failed').length).toBeGreaterThan(0);
    });
  });

  describe('sticky scene header', () => {
    type Callback = (entries: { isIntersecting: boolean }[]) => void;
    let callback: Callback | null = null;
    const disconnect = vi.fn();

    beforeEach(() => {
      callback = null;
      disconnect.mockClear();
      vi.stubGlobal(
        'IntersectionObserver',
        class {
          constructor(cb: Callback) {
            callback = cb;
          }
          observe() {}
          disconnect = disconnect;
        },
      );
    });
    afterEach(() => vi.unstubAllGlobals());

    const header = () => screen.getByRole('heading', { name: 'wolf-lunch' }).closest('[data-stuck]') as HTMLElement;

    it('is sticky to the top of the scrolling area', async () => {
      vi.mocked(getJobsList).mockResolvedValue([wolf(1), wolf(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      expect(header()).toHaveClass('sticky', 'top-0');
      // the timeline bar lives in the sticky block, so it follows the scroll
      expect(within(header()).getByRole('group', { name: 'Scene timeline' })).toBeInTheDocument();
    });

    it('adds a shadow only once it has stuck', async () => {
      vi.mocked(getJobsList).mockResolvedValue([wolf(1), wolf(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      expect(header()).toHaveAttribute('data-stuck', 'false');
      expect(header()).not.toHaveClass('shadow-md');

      // the sentinel above the header has scrolled out of view
      act(() => callback?.([{ isIntersecting: false }]));
      expect(header()).toHaveAttribute('data-stuck', 'true');
      expect(header()).toHaveClass('shadow-md');

      // ...and back into view at the top of the page
      act(() => callback?.([{ isIntersecting: true }]));
      expect(header()).toHaveAttribute('data-stuck', 'false');
      expect(header()).not.toHaveClass('shadow-md');
    });

    it('stops observing when the scene board goes away', async () => {
      vi.mocked(getJobsList).mockResolvedValue([wolf(1)]);
      const { unmount } = render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      unmount();
      expect(disconnect).toHaveBeenCalled();
    });
  });
});
