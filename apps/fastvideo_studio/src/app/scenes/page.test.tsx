import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ScenesPage from './page';
import { toast } from 'sonner';
import { dequeueJob, downloadMergedVideo, getJobsList, mergeScene, queueJobs, restoreJobVideo, trimJob, updateJob } from '@/lib/api';
import { downloadBlob } from '@/lib/utils';
import { makeJob } from '@/test/factories';

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

vi.mock('@/lib/api', () => ({
  getJobsList: vi.fn(),
  queueJobs: vi.fn(),
  dequeueJob: vi.fn(),
  mergeScene: vi.fn(),
  downloadMergedVideo: vi.fn(),
  getMergedVideoUrl: (name: string) => `http://test.local/api/merged/${name}`,
  updateJob: vi.fn(),
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
  trimJob: vi.fn(),
  restoreJobVideo: vi.fn(),
}));

vi.mock('@/lib/utils', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/utils')>()),
  downloadBlob: vi.fn(),
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

import { attachLastFrame } from '@/lib/continuation';

const sixSection = (n: number) =>
  [
    'subject_definitions:',
    '<Subject 1> is a man.',
    '',
    'summary:',
    '[reference generation] A man.',
    '',
    'retention_analysis:',
    '<Subject 1> (appears in [Shot 1]): fully_preserved - suit.',
    '',
    'detailed_description:',
    `[Shot 1] Wide shot. Clip ${n}.`,
    '',
    'overall_soundscape:',
    'Quiet.',
    '',
    'non_diegetic_music:',
    'N/A',
  ].join('\n');

/** Clip n's prompt after 'Use last frame' rewrote it (built by the same function the page uses). */
const attachedPrompt = (n: number) => {
  const images = [
    { source: '/refs/veteran.png', media_type: 'image' },
    { source: '/refs/young.png', media_type: 'image' },
  ];
  const r = attachLastFrame(sixSection(n), images, 'job-last-frame:wolf-1');
  if (!r.ok) throw new Error(r.error);
  return r.prompt;
};

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
    // The observer is created in an effect, which can land just after the heading does.
    const observerReady = () => waitFor(() => expect(callback).not.toBeNull());

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

      await observerReady();
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
      await observerReady();
      unmount();
      expect(disconnect).toHaveBeenCalled();
    });
  });

  describe('using the last frame of the previous clip', () => {
    const FRAME = 'job-last-frame:wolf-1';
    const IMAGES = [
      { source: '/refs/veteran.png', media_type: 'image' },
      { source: '/refs/young.png', media_type: 'image' },
    ];
    const done = (over: Parameters<typeof wolf>[1] = {}) =>
      wolf(1, { status: 'completed', output_path: '/out/a.mp4', prompt: sixSection(1), ...over });
    const next = (over: Parameters<typeof wolf>[1] = {}) =>
      wolf(2, { status: 'pending', prompt: sixSection(2), references: IMAGES, ...over });
    const button = () => screen.getByRole('checkbox', { name: /Use last frame of clip 1/ });

    beforeEach(() => {
      vi.mocked(updateJob).mockReset().mockResolvedValue({});
      vi.mocked(toast.success).mockClear();
      vi.mocked(toast.error).mockClear();
    });

    it('is offered on a clip that can still be edited', async () => {
      vi.mocked(getJobsList).mockResolvedValue([done(), next()]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      expect(button()).toBeEnabled();
      // the first clip has nothing before it
      expect(screen.getAllByRole('checkbox', { name: /Use last frame/ })).toHaveLength(1);
      expect(button()).not.toBeChecked();
    });

    it('is not offered on a clip that has already run', async () => {
      vi.mocked(getJobsList).mockResolvedValue([done(), next({ status: 'completed', output_path: '/out/b.mp4' })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(screen.queryByRole('checkbox', { name: /Use last frame/ })).not.toBeInTheDocument();
    });

    it("attaches the frame as a reference the server resolves, rewrites the prompt, and reloads", async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([done(), next()]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.click(button());

      await waitFor(() => expect(updateJob).toHaveBeenCalledTimes(1));
      const [jobId, updates] = vi.mocked(updateJob).mock.calls[0] as [string, { references: unknown[]; prompt: string }];
      expect(jobId).toBe('wolf-2');
      expect(updates.references).toEqual([...IMAGES, { source: FRAME, media_type: 'image' }]);
      expect(updates.prompt).toContain('[reference generation + keyframe completion]');
      expect(updates.prompt).toContain('<Picture 3> is the first frame of [Shot 1]');
      expect(updates.prompt).toContain('[Shot 1] Wide shot. The shot begins from <Picture 3>. Clip 2.');

      await waitFor(() => expect(toast.success).toHaveBeenCalled());
      expect(vi.mocked(toast.success).mock.calls[0][0]).toContain('last frame of clip 1');
      await waitFor(() => expect(getJobsList).toHaveBeenCalledTimes(2));
    });

    it('works before the previous clip has finished', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([done({ status: 'pending', output_path: null }), next()]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      expect(button()).toBeEnabled();
      expect(button().closest('label')?.getAttribute('title')).toMatch(/after clip 1 has finished/);
      await user.click(button());
      await waitFor(() => expect(updateJob).toHaveBeenCalled());
    });

    it('shows progress while the job is being updated', async () => {
      const user = userEvent.setup();
      let release: (v: unknown) => void = () => {};
      vi.mocked(updateJob).mockReturnValue(new Promise((r) => (release = r)));
      vi.mocked(getJobsList).mockResolvedValue([done(), next()]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.click(button());
      expect(screen.getByRole('checkbox', { name: 'Attaching…' })).toBeDisabled();

      release({});
      await waitFor(() => expect(toast.success).toHaveBeenCalled());
    });

    it("explains why it is unavailable for a prompt it can't rewrite", async () => {
      vi.mocked(getJobsList).mockResolvedValue([done(), next({ prompt: 'A plain prompt.' })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(button()).toBeDisabled();
      expect(button().closest('label')?.getAttribute('title')).toMatch(/six-section/);
    });

    it.each([
      ['a reference the server resolves', FRAME],
      ['a frame already saved to disk', '/data/uploads/last_frames/last_frame_wolf-1.png'],
    ])('shows the box checked when the frame is attached (%s)', async (_label, source) => {
      vi.mocked(getJobsList).mockResolvedValue([done(), next({ references: [...IMAGES, { source, media_type: 'image' }] })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(button()).toBeChecked();
    });

    it("shows a frame from another take as attached instead of stacking a second one", async () => {
      const other = 'job-last-frame:some-other-take';
      vi.mocked(getJobsList).mockResolvedValue([done(), next({ references: [...IMAGES, { source: other, media_type: 'image' }] })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(screen.getByRole('checkbox', { name: "Starts from another take's last frame" })).toBeChecked();
    });

    describe('unchecking it', () => {
      const attached = (source = FRAME, extra: { source: string; media_type: string }[] = []) =>
        next({
          prompt: attachedPrompt(2),
          references: [...IMAGES, { source, media_type: 'image' }, ...extra],
        });
      const updates = () => vi.mocked(updateJob).mock.calls[0][1] as { references: unknown[]; prompt: string };

      it('removes the frame and restores the prompt', async () => {
        const user = userEvent.setup();
        vi.mocked(getJobsList).mockResolvedValue([done(), attached()]);
        render(<ScenesPage />);
        await screen.findByRole('heading', { name: 'wolf-lunch' });

        await user.click(button());

        await waitFor(() => expect(updateJob).toHaveBeenCalledTimes(1));
        expect(vi.mocked(updateJob).mock.calls[0][0]).toBe('wolf-2');
        expect(updates().references).toEqual(IMAGES);
        expect(updates().prompt).toBe(sixSection(2));
        await waitFor(() => expect(toast.success).toHaveBeenCalled());
        await waitFor(() => expect(getJobsList).toHaveBeenCalledTimes(2));
      });

      it('also removes a frame saved to disk', async () => {
        const user = userEvent.setup();
        vi.mocked(getJobsList).mockResolvedValue([done(), attached('/data/uploads/last_frames/last_frame_wolf-1.png')]);
        render(<ScenesPage />);
        await screen.findByRole('heading', { name: 'wolf-lunch' });
        await user.click(button());
        await waitFor(() => expect(updateJob).toHaveBeenCalled());
        expect(updates().references).toEqual(IMAGES);
      });

      it("removes another take's frame too", async () => {
        const user = userEvent.setup();
        vi.mocked(getJobsList).mockResolvedValue([done(), attached('job-last-frame:some-other-take')]);
        render(<ScenesPage />);
        await screen.findByRole('heading', { name: 'wolf-lunch' });
        await user.click(screen.getByRole('checkbox', { name: "Starts from another take's last frame" }));
        await waitFor(() => expect(updateJob).toHaveBeenCalled());
        expect(updates().references).toEqual(IMAGES);
      });

      it('is undone exactly by checking it and then unchecking it', async () => {
        const user = userEvent.setup();
        vi.mocked(getJobsList).mockResolvedValue([done(), next()]);
        render(<ScenesPage />);
        await screen.findByRole('heading', { name: 'wolf-lunch' });

        // Once saved, the reload shows the clip as the first click left it.
        const attachedClip = attached();
        vi.mocked(getJobsList).mockResolvedValue([done(), attachedClip]);
        await user.click(button());
        await waitFor(() => expect(updateJob).toHaveBeenCalledTimes(1));
        expect(updates().prompt).toBe(attachedClip.prompt); // same rewrite the page's own helper produces

        await user.click(await screen.findByRole('checkbox', { name: /Use last frame of clip 1/, checked: true }));
        await waitFor(() => expect(updateJob).toHaveBeenCalledTimes(2));
        const restored = vi.mocked(updateJob).mock.calls[1][1] as { prompt: string; references: unknown[] };
        expect(restored.prompt).toBe(sixSection(2));
        expect(restored.references).toEqual(IMAGES);
      });

      it('says why when other images were added after the frame, and changes nothing', async () => {
        const user = userEvent.setup();
        vi.mocked(getJobsList).mockResolvedValue([
          done(),
          attached(FRAME, [{ source: '/refs/extra.png', media_type: 'image' }]),
        ]);
        render(<ScenesPage />);
        await screen.findByRole('heading', { name: 'wolf-lunch' });
        await user.click(button());
        await waitFor(() => expect(toast.error).toHaveBeenCalled());
        expect(vi.mocked(toast.error).mock.calls[0][0]).toMatch(/Other images were added/);
        expect(updateJob).not.toHaveBeenCalled();
      });
    });

    it('reports a failure to update the job', async () => {
      const user = userEvent.setup();
      vi.mocked(updateJob).mockRejectedValue(new Error('Job is running; only pending jobs can be edited.'));
      vi.mocked(getJobsList).mockResolvedValue([done(), next()]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.click(button());

      await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Job is running; only pending jobs can be edited.'));
      expect(toast.success).not.toHaveBeenCalled();
    });

    it('uses the clip that is shown when a different take is chosen', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([
        done({ id: 'take-a', created_at: 1 }),
        done({ id: 'take-b', created_at: 2 }),
        next(),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.selectOptions(screen.getByLabelText('Take for clip 1'), 'take-a');
      await user.click(button());
      await waitFor(() => expect(updateJob).toHaveBeenCalled());
      const updates = vi.mocked(updateJob).mock.calls[0][1] as { references: { source: string }[] };
      expect(updates.references.at(-1)?.source).toBe('job-last-frame:take-a');
    });
  });

  describe('queueing', () => {
    const clip = (n: number, over: Parameters<typeof wolf>[1] = {}) =>
      wolf(n, { status: 'pending', prompt: sixSection(n), ...over });
    const openDialog = async (user: ReturnType<typeof userEvent.setup>) => {
      await user.click(screen.getByRole('button', { name: 'Queue scene' }));
      return screen.findByRole('dialog');
    };

    beforeEach(() => {
      vi.mocked(updateJob).mockReset().mockResolvedValue({});
      vi.mocked(queueJobs).mockReset().mockResolvedValue([]);
      vi.mocked(dequeueJob).mockReset().mockResolvedValue(makeJob());
      vi.mocked(toast.success).mockClear();
      vi.mocked(toast.error).mockClear();
    });

    it('queues the whole scene in order and leaves the clips as they are', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([clip(3), clip(1), clip(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      expect(within(dialog).getByText('3 clips will be queued and run one at a time, in order.')).toBeInTheDocument();
      expect(within(dialog).getByRole('checkbox')).not.toBeChecked();
      await user.click(within(dialog).getByRole('button', { name: 'Queue 3 clips' }));

      await waitFor(() => expect(queueJobs).toHaveBeenCalledWith(['wolf-1', 'wolf-2', 'wolf-3']));
      expect(updateJob).not.toHaveBeenCalled();
      expect(toast.success).toHaveBeenCalledWith('Queued 3 clips.');
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('links each clip to the frame before it when asked to, before queueing', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([clip(3), clip(1), clip(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      await user.click(within(dialog).getByRole('checkbox'));
      await user.click(within(dialog).getByRole('button', { name: 'Queue 3 clips' }));

      await waitFor(() => expect(queueJobs).toHaveBeenCalledWith(['wolf-1', 'wolf-2', 'wolf-3']));
      expect(updateJob).toHaveBeenCalledTimes(2);
      const refs = (n: number) =>
        (vi.mocked(updateJob).mock.calls.find((c) => c[0] === `wolf-${n}`)?.[1] as { references: { source: string }[] })
          .references.map((r) => r.source);
      expect(refs(2)).toEqual(['job-last-frame:wolf-1']);
      expect(refs(3)).toEqual(['job-last-frame:wolf-2']);
      // linked before queued, so the server never sees a queued clip without its link
      expect(vi.mocked(updateJob).mock.invocationCallOrder.at(-1)!).toBeLessThan(
        vi.mocked(queueJobs).mock.invocationCallOrder[0],
      );
    });

    it('keeps the links you set on individual clips, and adds no others unless asked', async () => {
      const user = userEvent.setup();
      const link = [{ source: 'job-last-frame:wolf-1', media_type: 'image' }];
      vi.mocked(getJobsList).mockResolvedValue([clip(1), clip(2, { references: link }), clip(3)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      expect(within(dialog).getByText(/Clip 2 already starts from a previous frame/)).toBeInTheDocument();
      await user.click(within(dialog).getByRole('button', { name: 'Queue 3 clips' }));

      await waitFor(() => expect(queueJobs).toHaveBeenCalledWith(['wolf-1', 'wolf-2', 'wolf-3']));
      expect(updateJob).not.toHaveBeenCalled();
    });

    it('skips clips that already ran and says so', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([
        clip(1, { status: 'completed', output_path: '/o/1.mp4' }),
        clip(2),
        clip(3, { status: 'running' }),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      await user.click(within(dialog).getByRole('checkbox'));
      expect(within(dialog).getByText(/Skipping clip 1 \(already finished\), clip 3 \(already running\)/)).toBeInTheDocument();
      await user.click(within(dialog).getByRole('button', { name: 'Queue 1 clip' }));

      await waitFor(() => expect(queueJobs).toHaveBeenCalledWith(['wolf-2']));
      // clip 2 still follows the finished clip 1's last frame
      expect(vi.mocked(updateJob).mock.calls[0][0]).toBe('wolf-2');
    });

    it("mentions clips whose prompt can't be linked once linking is on, and still queues them", async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([clip(1), clip(2), clip(3, { prompt: 'A plain prompt.' })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      expect(within(dialog).queryByText(/can't be linked automatically/)).not.toBeInTheDocument();
      await user.click(within(dialog).getByRole('checkbox'));
      expect(within(dialog).getByText(/Clip 3 can't be linked automatically/)).toBeInTheDocument();
      await user.click(within(dialog).getByRole('button', { name: 'Queue 3 clips' }));
      await waitFor(() => expect(queueJobs).toHaveBeenCalledWith(['wolf-1', 'wolf-2', 'wolf-3']));
      expect(updateJob).toHaveBeenCalledTimes(1); // only clip 2
    });

    it('queues nothing, and says which clip, when a link cannot be saved', async () => {
      const user = userEvent.setup();
      vi.mocked(updateJob).mockRejectedValue(new Error('Job is running'));
      vi.mocked(getJobsList).mockResolvedValue([clip(1), clip(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      await user.click(within(dialog).getByRole('checkbox'));
      await user.click(within(dialog).getByRole('button', { name: 'Queue 2 clips' }));

      await waitFor(() => expect(toast.error).toHaveBeenCalled());
      expect(vi.mocked(toast.error).mock.calls[0][0]).toContain('Clip 2: Job is running. Nothing was queued.');
      expect(queueJobs).not.toHaveBeenCalled();
    });

    it("reports the server's refusal", async () => {
      const user = userEvent.setup();
      vi.mocked(queueJobs).mockRejectedValue(new Error('Job already completed. Delete and re-create to run again.'));
      vi.mocked(getJobsList).mockResolvedValue([clip(1)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      await user.click(within(dialog).getByRole('button', { name: 'Queue 1 clip' }));
      await waitFor(() =>
        expect(toast.error).toHaveBeenCalledWith('Job already completed. Delete and re-create to run again.'),
      );
    });

    it('changes nothing when cancelled', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([clip(1), clip(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const dialog = await openDialog(user);
      await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(queueJobs).not.toHaveBeenCalled();
      expect(updateJob).not.toHaveBeenCalled();
    });

    it('is unavailable when every clip has already run or is queued', async () => {
      vi.mocked(getJobsList).mockResolvedValue([
        clip(1, { status: 'completed', output_path: '/o/1.mp4' }),
        clip(2, { status: 'queued' }),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(screen.getByRole('button', { name: 'Queue scene' })).toBeDisabled();
    });

    it('queues a single clip', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([clip(1), clip(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.click(within(document.getElementById('clip-wolf-2') as HTMLElement).getByRole('button', { name: 'Queue' }));

      await waitFor(() => expect(queueJobs).toHaveBeenCalledWith(['wolf-2']));
      await waitFor(() => expect(getJobsList).toHaveBeenCalledTimes(2));
    });

    it('takes a queued clip back out of the queue', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([clip(1, { status: 'running' }), clip(2, { status: 'queued' })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const row = document.getElementById('clip-wolf-2') as HTMLElement;
      expect(within(row).queryByRole('button', { name: 'Queue' })).not.toBeInTheDocument();
      await user.click(within(row).getByRole('button', { name: 'Remove from queue' }));
      await waitFor(() => expect(dequeueJob).toHaveBeenCalledWith('wolf-2'));
    });

    it('shows what a queued clip is waiting for', async () => {
      vi.mocked(getJobsList).mockResolvedValue([
        clip(1, { status: 'running' }),
        clip(2, { status: 'queued', references: [{ source: 'job-last-frame:wolf-1', media_type: 'image' }] }),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(screen.getByText('waiting for clip 1')).toBeInTheDocument();
      expect(screen.getByText("starts from clip 1's last frame")).toBeInTheDocument();
    });

    it('stops waiting once the previous clip has finished', async () => {
      vi.mocked(getJobsList).mockResolvedValue([
        clip(1, { status: 'completed', output_path: '/o/1.mp4' }),
        clip(2, { status: 'queued', references: [{ source: 'job-last-frame:wolf-1', media_type: 'image' }] }),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(screen.queryByText('waiting for clip 1')).not.toBeInTheDocument();
    });

    it('refreshes on its own while something is queued or running', async () => {
      const setIntervalSpy = vi.spyOn(globalThis, 'setInterval');
      vi.mocked(getJobsList).mockResolvedValue([clip(1, { status: 'running' }), clip(2, { status: 'queued' })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      const poll = setIntervalSpy.mock.calls.find(([, ms]) => ms === 3000)?.[0] as () => void;
      expect(poll).toBeTypeOf('function');
      vi.mocked(getJobsList).mockResolvedValue([
        clip(1, { status: 'completed', output_path: '/o/1.mp4' }),
        clip(2, { status: 'running' }),
      ]);
      await act(async () => poll());
      await waitFor(() => expect(within(document.getElementById('clip-wolf-2') as HTMLElement).getByText('running')).toBeInTheDocument());
      setIntervalSpy.mockRestore();
    });

    it("doesn't poll when nothing is in progress", async () => {
      const setIntervalSpy = vi.spyOn(globalThis, 'setInterval');
      vi.mocked(getJobsList).mockResolvedValue([clip(1), clip(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(setIntervalSpy.mock.calls.some(([, ms]) => ms === 3000)).toBe(false);
      setIntervalSpy.mockRestore();
    });
  });

  describe('merging a scene', () => {
    const finished = (n: number, over: Parameters<typeof wolf>[1] = {}) =>
      wolf(n, { status: 'completed', output_path: `/o/${n}.mp4`, ...over });
    const mergeButton = () => screen.getByRole('button', { name: /Merge scene|Merging/ });

    beforeEach(() => {
      vi.mocked(mergeScene).mockReset().mockResolvedValue({ filename: 'wolf-lunch-20260927-100000.mp4', clips: 3, seconds: 33 });
      vi.mocked(downloadMergedVideo).mockReset().mockResolvedValue(new Blob(['x']));
      vi.mocked(downloadBlob).mockReset();
      vi.mocked(toast.error).mockClear();
    });

    it('joins the clips in scene order and shows the result', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([finished(3), finished(1), finished(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      expect(mergeButton()).toBeEnabled();
      await user.click(mergeButton());

      await waitFor(() => expect(mergeScene).toHaveBeenCalledWith(['wolf-1', 'wolf-2', 'wolf-3'], 'wolf-lunch'));
      const dialog = await screen.findByRole('dialog');
      expect(within(dialog).getByText('wolf-lunch merged')).toBeInTheDocument();
      expect(within(dialog).getByText('3 clips joined in order · 0:33')).toBeInTheDocument();
      expect(within(dialog).getByLabelText('Merged video of wolf-lunch')).toHaveAttribute(
        'src',
        'http://test.local/api/merged/wolf-lunch-20260927-100000.mp4',
      );
    });

    it('merges the take that is shown for a clip', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([
        finished(1, { id: 'take-a', created_at: 1 }),
        finished(1, { id: 'take-b', created_at: 2 }),
        finished(2),
      ]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.selectOptions(screen.getByLabelText('Take for clip 1'), 'take-a');
      await user.click(mergeButton());
      await waitFor(() => expect(mergeScene).toHaveBeenCalledWith(['take-a', 'wolf-2'], 'wolf-lunch'));
    });

    it('waits until every clip has finished, and says how many have not', async () => {
      vi.mocked(getJobsList).mockResolvedValue([finished(1), wolf(2, { status: 'running' }), wolf(3, { status: 'queued' })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(mergeButton()).toBeDisabled();
      expect(mergeButton()).toHaveAttribute('title', "2 clips haven't finished yet");
    });

    it('says so for a single unfinished clip', async () => {
      vi.mocked(getJobsList).mockResolvedValue([finished(1), wolf(2, { status: 'failed' })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(mergeButton()).toHaveAttribute('title', "1 clip hasn't finished yet");
    });

    it('needs a video on every clip, not just a completed status', async () => {
      vi.mocked(getJobsList).mockResolvedValue([finished(1), finished(2, { output_path: null })]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      expect(mergeButton()).toBeDisabled();
    });

    it("reports the server's refusal and shows nothing", async () => {
      const user = userEvent.setup();
      vi.mocked(mergeScene).mockRejectedValue(new Error("'wolf-lunch-clip-02' hasn't finished (running), so the scene can't be merged yet."));
      vi.mocked(getJobsList).mockResolvedValue([finished(1), finished(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.click(mergeButton());
      await waitFor(() => expect(toast.error).toHaveBeenCalledWith(expect.stringContaining("hasn't finished (running)")));
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      expect(mergeButton()).toBeEnabled(); // can try again
    });

    it('shows progress while merging', async () => {
      const user = userEvent.setup();
      let release: (v: { filename: string; clips: number; seconds: number }) => void = () => {};
      vi.mocked(mergeScene).mockReturnValue(new Promise((r) => (release = r)));
      vi.mocked(getJobsList).mockResolvedValue([finished(1), finished(2)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });

      await user.click(mergeButton());
      expect(screen.getByRole('button', { name: 'Merging…' })).toBeDisabled();
      release({ filename: 'a.mp4', clips: 2, seconds: 22 });
      expect(await screen.findByRole('dialog')).toBeInTheDocument();
    });

    it('downloads the merged video', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([finished(1), finished(2), finished(3)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      await user.click(mergeButton());

      await user.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Download' }));
      await waitFor(() => expect(downloadBlob).toHaveBeenCalled());
      expect(downloadMergedVideo).toHaveBeenCalledWith('wolf-lunch-20260927-100000.mp4');
      expect(vi.mocked(downloadBlob).mock.calls[0][1]).toBe('wolf-lunch-20260927-100000.mp4');
    });

    it('reports a failed download', async () => {
      const user = userEvent.setup();
      vi.mocked(downloadMergedVideo).mockRejectedValue(new Error('Failed to download the merged video'));
      vi.mocked(getJobsList).mockResolvedValue([finished(1), finished(2), finished(3)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      await user.click(mergeButton());

      await user.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Download' }));
      await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Failed to download the merged video'));
      expect(downloadBlob).not.toHaveBeenCalled();
    });

    it('closes', async () => {
      const user = userEvent.setup();
      vi.mocked(getJobsList).mockResolvedValue([finished(1), finished(2), finished(3)]);
      render(<ScenesPage />);
      await screen.findByRole('heading', { name: 'wolf-lunch' });
      await user.click(mergeButton());

      // the footer button and the corner X both close it
      const dialog = await screen.findByRole('dialog');
      await user.click(within(dialog).getAllByRole('button', { name: 'Close' })[0]);
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

      await user.click(mergeButton());
      await user.click(within(await screen.findByRole('dialog')).getAllByRole('button', { name: 'Close' }).at(-1)!);
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });
  });
});

describe('editing a clip\'s video', () => {
  const finished = (n: number, over: Parameters<typeof wolf>[1] = {}) =>
    wolf(n, { status: 'completed', output_path: `/o/${n}.mp4`, num_frames: 240, fps: 24, ...over });
  const openEdit = async (user: ReturnType<typeof userEvent.setup>, id = 'wolf-1') => {
    await user.click(within(document.getElementById(`clip-${id}`) as HTMLElement).getByRole('button', { name: 'Edit video' }));
    return screen.findByRole('dialog');
  };
  const applyButton = (dialog: HTMLElement) => within(dialog).getByRole('button', { name: /Apply|Applying/ });
  // section is 1-based, matching the "Section N" label shown in the dialog.
  const sectionSlider = (dialog: HTMLElement, section: number, label: string) =>
    within(dialog).getByRole('slider', { name: `Section ${section} ${label}` });
  const press = (el: HTMLElement, key: string, times = 1) => {
    for (let i = 0; i < times; i++) fireEvent.keyDown(el, { key });
  };
  const neutralSection = { end_seconds: null, brightness: 0, contrast: 1, saturation: 1 };

  beforeEach(() => {
    vi.mocked(trimJob).mockReset();
    vi.mocked(restoreJobVideo).mockReset();
    vi.mocked(toast.success).mockClear();
    vi.mocked(toast.error).mockClear();
  });

  it('is only offered on a clip with a finished video', async () => {
    vi.mocked(getJobsList).mockResolvedValue([finished(1), wolf(2, { status: 'running' })]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });
    expect(within(document.getElementById('clip-wolf-1') as HTMLElement).getByRole('button', { name: 'Edit video' })).toBeInTheDocument();
    expect(within(document.getElementById('clip-wolf-2') as HTMLElement).queryByRole('button', { name: 'Edit video' })).not.toBeInTheDocument();
  });

  it('opens with the full current range and one neutral section, and shows the current duration', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    expect(within(dialog).getByText(/Currently 0:10\./)).toBeInTheDocument();
    expect(within(dialog).getByLabelText('Start (seconds)')).toHaveValue(0);
    expect(within(dialog).getByLabelText('End (seconds, optional)')).toHaveValue(10);
    expect(within(dialog).getByText('Section 1 · 0:00–end')).toBeInTheDocument();
    expect(sectionSlider(dialog, 1, 'Brightness')).toHaveAttribute('aria-valuenow', '0');
    expect(sectionSlider(dialog, 1, 'Contrast')).toHaveAttribute('aria-valuenow', '1');
    expect(sectionSlider(dialog, 1, 'Saturation')).toHaveAttribute('aria-valuenow', '1');
    expect(within(dialog).queryByRole('button', { name: 'Merge with next' })).not.toBeInTheDocument();
  });

  it('reopening a previously edited clip shows the range and sections that produced it, not neutral', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([
      finished(1, {
        edit_start_seconds: 1,
        edit_end_seconds: 3,
        edit_segments: [{ end_seconds: null, brightness: 20, contrast: 1.4, saturation: 0.5 }],
      }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    expect(within(dialog).getByLabelText('Start (seconds)')).toHaveValue(1);
    expect(within(dialog).getByLabelText('End (seconds, optional)')).toHaveValue(3);
    expect(sectionSlider(dialog, 1, 'Brightness')).toHaveAttribute('aria-valuenow', '20');
    expect(sectionSlider(dialog, 1, 'Contrast')).toHaveAttribute('aria-valuenow', '1.4');
    expect(sectionSlider(dialog, 1, 'Saturation')).toHaveAttribute('aria-valuenow', '0.5');
  });

  it('reopening a multi-section edit shows every section, independently', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([
      finished(1, {
        edit_segments: [
          { end_seconds: 4, brightness: 20, contrast: 1, saturation: 1 },
          { end_seconds: null, brightness: 0, contrast: 1.4, saturation: 1 },
        ],
      }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    expect(within(dialog).getByText('Section 1 · 0:00–0:04')).toBeInTheDocument();
    expect(within(dialog).getByText('Section 2 · 0:04–end')).toBeInTheDocument();
    expect(sectionSlider(dialog, 1, 'Brightness')).toHaveAttribute('aria-valuenow', '20');
    expect(sectionSlider(dialog, 2, 'Contrast')).toHaveAttribute('aria-valuenow', '1.4');
  });

  it('restoring resets the fields to one neutral section and the full length', async () => {
    const user = userEvent.setup();
    vi.mocked(restoreJobVideo).mockResolvedValue(finished(1, { num_frames: 240 })); // back to the original 10s, neutral edit_*
    vi.mocked(getJobsList).mockResolvedValue([
      finished(1, {
        num_frames: 48,
        edit_start_seconds: 1,
        edit_end_seconds: 3,
        edit_segments: [{ end_seconds: null, brightness: 20, contrast: 1.4, saturation: 0.5 }],
      }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    expect(sectionSlider(dialog, 1, 'Brightness')).toHaveAttribute('aria-valuenow', '20');

    await user.click(within(dialog).getByRole('button', { name: 'Restore original' }));

    await waitFor(() => expect(sectionSlider(dialog, 1, 'Brightness')).toHaveAttribute('aria-valuenow', '0'));
    expect(sectionSlider(dialog, 1, 'Contrast')).toHaveAttribute('aria-valuenow', '1');
    expect(sectionSlider(dialog, 1, 'Saturation')).toHaveAttribute('aria-valuenow', '1');
    expect(within(dialog).getByLabelText('Start (seconds)')).toHaveValue(0);
    expect(within(dialog).getByLabelText('End (seconds, optional)')).toHaveValue(10);
  });

  it('applies the chosen range and reports it', async () => {
    const user = userEvent.setup();
    vi.mocked(trimJob).mockResolvedValue({ ...finished(1), output_path: '/o/edited.mp4', num_frames: 48 });
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.clear(within(dialog).getByLabelText('Start (seconds)'));
    await user.type(within(dialog).getByLabelText('Start (seconds)'), '1');
    await user.clear(within(dialog).getByLabelText('End (seconds, optional)'));
    await user.type(within(dialog).getByLabelText('End (seconds, optional)'), '3');
    await user.click(applyButton(dialog));

    await waitFor(() =>
      expect(trimJob).toHaveBeenCalledWith('wolf-1', { startSeconds: 1, endSeconds: 3, segments: [neutralSection] }),
    );
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Applied: 0:01–0:03.'));
  });

  it('applies chosen color values alongside the range, in one call', async () => {
    const user = userEvent.setup();
    vi.mocked(trimJob).mockResolvedValue(finished(1));
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    press(sectionSlider(dialog, 1, 'Brightness'), 'ArrowRight', 4); // step 5: 0 -> 20
    press(sectionSlider(dialog, 1, 'Contrast'), 'ArrowRight', 4); // step 0.1: 1 -> 1.4
    press(sectionSlider(dialog, 1, 'Saturation'), 'ArrowLeft', 5); // step 0.1: 1 -> 0.5
    await user.click(applyButton(dialog));

    await waitFor(() =>
      expect(trimJob).toHaveBeenCalledWith('wolf-1', {
        startSeconds: 0,
        endSeconds: 10,
        segments: [{ end_seconds: null, brightness: 20, contrast: 1.4, saturation: 0.5 }],
      }),
    );
  });

  it('splitting at the current time creates two independently graded sections and applies both', async () => {
    const user = userEvent.setup();
    vi.mocked(trimJob).mockResolvedValue(finished(1));
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]); // 240 frames @ 24fps = 10s
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    const video = dialog.querySelector('video') as HTMLVideoElement;
    video.currentTime = 4;
    await user.click(within(dialog).getByRole('button', { name: 'Split at current time' }));

    expect(within(dialog).getByText('Section 1 · 0:00–0:04')).toBeInTheDocument();
    expect(within(dialog).getByText('Section 2 · 0:04–end')).toBeInTheDocument();

    press(sectionSlider(dialog, 1, 'Brightness'), 'ArrowRight', 4); // -> 20
    press(sectionSlider(dialog, 2, 'Contrast'), 'ArrowRight', 4); // -> 1.4
    await user.click(applyButton(dialog));

    await waitFor(() =>
      expect(trimJob).toHaveBeenCalledWith('wolf-1', {
        startSeconds: 0,
        endSeconds: 10,
        segments: [
          { end_seconds: 4, brightness: 20, contrast: 1, saturation: 1 },
          { end_seconds: null, brightness: 0, contrast: 1.4, saturation: 1 },
        ],
      }),
    );
  });

  it('splitting too close to an existing edge is refused', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    const video = dialog.querySelector('video') as HTMLVideoElement;
    video.currentTime = 0; // right at the (only) section's start edge
    await user.click(within(dialog).getByRole('button', { name: 'Split at current time' }));

    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    expect(within(dialog).queryByText(/Section 2/)).not.toBeInTheDocument();
  });

  it('merging a section with the next removes the cut and keeps the first section\'s grade', async () => {
    const user = userEvent.setup();
    vi.mocked(trimJob).mockResolvedValue(finished(1));
    vi.mocked(getJobsList).mockResolvedValue([
      finished(1, {
        edit_segments: [
          { end_seconds: 4, brightness: 20, contrast: 1, saturation: 1 },
          { end_seconds: null, brightness: 0, contrast: 1.4, saturation: 1 },
        ],
      }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.click(within(dialog).getByRole('button', { name: 'Merge with next' }));

    expect(within(dialog).queryByText(/Section 2/)).not.toBeInTheDocument();
    expect(within(dialog).getByText('Section 1 · 0:00–end')).toBeInTheDocument();
    expect(sectionSlider(dialog, 1, 'Brightness')).toHaveAttribute('aria-valuenow', '20'); // the first section's grade won

    await user.click(applyButton(dialog));
    await waitFor(() =>
      expect(trimJob).toHaveBeenCalledWith('wolf-1', {
        startSeconds: 0,
        endSeconds: 10,
        segments: [{ end_seconds: null, brightness: 20, contrast: 1, saturation: 1 }],
      }),
    );
  });

  it('stepping frame by frame moves the preview by exactly one frame, clamped to the clip', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]); // 240 frames @ 24fps = 10s
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    const video = dialog.querySelector('video') as HTMLVideoElement;
    Object.defineProperty(video, 'readyState', { value: 4, configurable: true });
    video.currentTime = 5;

    await user.click(within(dialog).getByRole('button', { name: /Next frame/ }));
    expect(video.currentTime).toBeCloseTo(5 + 1 / 24, 5);

    await user.click(within(dialog).getByRole('button', { name: /Previous frame/ }));
    await user.click(within(dialog).getByRole('button', { name: /Previous frame/ }));
    expect(video.currentTime).toBeCloseTo(5 - 1 / 24, 5);

    video.currentTime = 0;
    await user.click(within(dialog).getByRole('button', { name: /Previous frame/ }));
    expect(video.currentTime).toBe(0); // clamped, doesn't go negative

    video.currentTime = 10;
    await user.click(within(dialog).getByRole('button', { name: /Next frame/ }));
    expect(video.currentTime).toBe(10); // clamped to the clip's length
  });

  it('the frame readout updates as the preview moves', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]); // 240 frames @ 24fps = 10s
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    expect(within(dialog).getByText('Frame 0 · 0.00s')).toBeInTheDocument();

    const video = dialog.querySelector('video') as HTMLVideoElement;
    video.currentTime = 5;
    fireEvent(video, new Event('timeupdate')); // what a real browser fires as currentTime changes

    expect(within(dialog).getByText('Frame 120 · 5.00s')).toBeInTheDocument();
  });

  it('previewing a section seeks to its start immediately once metadata is loaded', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([
      finished(1, {
        edit_segments: [
          { end_seconds: 4, brightness: 20, contrast: 1, saturation: 1 },
          { end_seconds: null, brightness: 0, contrast: 1.4, saturation: 1 },
        ],
      }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    const video = dialog.querySelector('video') as HTMLVideoElement;
    Object.defineProperty(video, 'readyState', { value: 4, configurable: true }); // HAVE_ENOUGH_DATA

    await user.click(within(dialog).getAllByRole('button', { name: 'Preview' })[1]); // section 2: 0:04-end
    expect(video.currentTime).toBe(4); // starts at 4

    await user.click(within(dialog).getAllByRole('button', { name: 'Preview' })[0]); // section 1: 0:00-0:04
    expect(video.currentTime).toBe(0); // starts at 0
  });

  it('previewing a section before metadata has loaded defers the seek instead of dropping it', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([
      finished(1, {
        edit_segments: [
          { end_seconds: 4, brightness: 20, contrast: 1, saturation: 1 },
          { end_seconds: null, brightness: 0, contrast: 1.4, saturation: 1 },
        ],
      }),
    ]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    const video = dialog.querySelector('video') as HTMLVideoElement;
    Object.defineProperty(video, 'readyState', { value: 0, configurable: true }); // HAVE_NOTHING -- not loaded yet

    await user.click(within(dialog).getAllByRole('button', { name: 'Preview' })[1]); // section 2 starts at 4
    expect(video.currentTime).toBe(0); // not applied yet -- would be silently dropped by a real browser here

    Object.defineProperty(video, 'readyState', { value: 1, configurable: true });
    fireEvent(video, new Event('loadedmetadata'));
    expect(video.currentTime).toBe(4); // applied once metadata is actually available
  });

  it('leaving the end blank keeps to the end of the clip', async () => {
    const user = userEvent.setup();
    vi.mocked(trimJob).mockResolvedValue(finished(1));
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.clear(within(dialog).getByLabelText('Start (seconds)'));
    await user.type(within(dialog).getByLabelText('Start (seconds)'), '2');
    await user.clear(within(dialog).getByLabelText('End (seconds, optional)'));
    await user.click(applyButton(dialog));

    await waitFor(() =>
      expect(trimJob).toHaveBeenCalledWith('wolf-1', { startSeconds: 2, endSeconds: undefined, segments: [neutralSection] }),
    );
  });

  it('disables Apply for an invalid range and explains why', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.clear(within(dialog).getByLabelText('Start (seconds)'));
    await user.type(within(dialog).getByLabelText('Start (seconds)'), '5');
    await user.clear(within(dialog).getByLabelText('End (seconds, optional)'));
    await user.type(within(dialog).getByLabelText('End (seconds, optional)'), '3');

    expect(within(dialog).getByText('The end must come after the start.')).toBeInTheDocument();
    expect(applyButton(dialog)).toBeDisabled();
    expect(trimJob).not.toHaveBeenCalled();
  });

  it('reports a negative start instead of the generic message', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.clear(within(dialog).getByLabelText('Start (seconds)'));
    await user.type(within(dialog).getByLabelText('Start (seconds)'), '-1');
    expect(within(dialog).getByText('The start must be 0 or later.')).toBeInTheDocument();
  });

  it.each([
    ['Brightness', -100, 100],
    ['Contrast', 0, 3],
    ['Saturation', 0, 3],
  ])('cannot push the %s slider past its bounds', async (label, min, max) => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    const slider = sectionSlider(dialog, 1, label);
    press(slider, 'End');
    expect(slider).toHaveAttribute('aria-valuenow', String(max));
    press(slider, 'Home');
    expect(slider).toHaveAttribute('aria-valuenow', String(min));
  });

  it('restores the original video and reports it', async () => {
    const user = userEvent.setup();
    vi.mocked(restoreJobVideo).mockResolvedValue(finished(1));
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.click(within(dialog).getByRole('button', { name: 'Restore original' }));

    await waitFor(() => expect(restoreJobVideo).toHaveBeenCalledWith('wolf-1'));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Restored the original video.'));
  });

  it('reports a failed edit and leaves the dialog open to retry', async () => {
    const user = userEvent.setup();
    vi.mocked(trimJob).mockRejectedValue(new Error('That range keeps no frames.'));
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.click(applyButton(dialog));

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('That range keeps no frames.'));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('closes, and refetches jobs after a change but not after just closing', async () => {
    const user = userEvent.setup();
    vi.mocked(getJobsList).mockResolvedValue([finished(1)]);
    render(<ScenesPage />);
    await screen.findByRole('heading', { name: 'wolf-lunch' });

    const dialog = await openEdit(user);
    await user.click(within(dialog).getAllByRole('button', { name: 'Close' })[0]);
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(getJobsList).toHaveBeenCalledTimes(1); // just the initial load
  });
});
