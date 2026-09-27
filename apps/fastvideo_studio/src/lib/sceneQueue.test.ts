import { describe, expect, it } from 'vitest';

import { deferredLastFrameSource } from '@/lib/continuation';
import { planSceneQueue } from '@/lib/sceneQueue';
import { buildScenes, type SceneClip } from '@/lib/scenes';
import { makeJob } from '@/test/factories';

const SIX_SECTION = [
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
  '[Shot 1] Wide shot. A man.',
  '',
  'overall_soundscape:',
  'Quiet.',
  '',
  'non_diegetic_music:',
  'N/A',
].join('\n');

const clips = (...overrides: Parameters<typeof makeJob>[0][]): SceneClip[] => {
  const jobs = overrides.map((over, i) =>
    makeJob({
      id: `c${i + 1}`,
      name: `s-clip-${i + 1}`,
      created_at: i + 1,
      status: 'pending',
      prompt: SIX_SECTION,
      ...over,
    }),
  );
  return buildScenes(jobs)[0].clips;
};

const ids = (list: { job: { id: string } }[]) => list.map((c) => c.job.id);

describe('planSceneQueue', () => {
  it('queues every clip that can run, in scene order', () => {
    const plan = planSceneQueue(clips({}, {}, {}), { chain: false });
    expect(ids(plan.queue)).toEqual(['c1', 'c2', 'c3']);
    expect(plan.skipped).toEqual([]);
  });

  it.each([
    ['completed', 'already finished'],
    ['running', 'already running'],
    ['queued', 'already queued'],
  ])('leaves %s clips alone', (status, reason) => {
    const plan = planSceneQueue(clips({}, { status }, {}), { chain: false });
    expect(ids(plan.queue)).toEqual(['c1', 'c3']);
    expect(plan.skipped.map((s) => [s.clip.job.id, s.reason])).toEqual([['c2', reason]]);
  });

  it('requeues failed and stopped clips', () => {
    const plan = planSceneQueue(clips({ status: 'failed' }, { status: 'stopped' }), { chain: false });
    expect(ids(plan.queue)).toEqual(['c1', 'c2']);
  });

  describe('chaining clips through last frames', () => {
    it('links each clip after the first to the one before it', () => {
      const plan = planSceneQueue(clips({}, {}, {}), { chain: true });
      expect(plan.chain.map((c) => [c.clip.job.id, c.previous.job.id])).toEqual([
        ['c2', 'c1'],
        ['c3', 'c2'],
      ]);
    });

    it('links nothing when not asked to', () => {
      expect(planSceneQueue(clips({}, {}), { chain: false }).chain).toEqual([]);
    });

    it('links a clip to a previous clip that has already finished', () => {
      const plan = planSceneQueue(clips({ status: 'completed' }, {}), { chain: true });
      expect(plan.chain.map((c) => c.previous.job.id)).toEqual(['c1']);
      expect(ids(plan.queue)).toEqual(['c2']);
    });

    it('does not link clips that are not being queued', () => {
      const plan = planSceneQueue(clips({}, { status: 'completed' }, {}), { chain: true });
      expect(plan.chain.map((c) => c.clip.job.id)).toEqual(['c3']);
    });

    it('keeps a frame the clip already opens on', () => {
      const own = [{ source: deferredLastFrameSource('c1'), media_type: 'image' }];
      const stale = [{ source: '/uploads/last_frames/last_frame_old-take.png', media_type: 'image' }];
      const plan = planSceneQueue(clips({}, { references: own }, { references: stale }), { chain: true });
      expect(plan.chain).toEqual([]);
      expect(plan.cannotChain).toEqual([]);
    });

    it("reports clips whose prompt can't be rewritten instead of linking them", () => {
      const plan = planSceneQueue(clips({}, { prompt: 'A plain prompt.' }), { chain: true });
      expect(plan.chain).toEqual([]);
      expect(plan.cannotChain).toHaveLength(1);
      expect(plan.cannotChain[0].clip.job.id).toBe('c2');
      expect(plan.cannotChain[0].reason).toMatch(/six-section/);
      expect(ids(plan.queue)).toEqual(['c1', 'c2']); // still queued, just not linked
    });
  });
});
