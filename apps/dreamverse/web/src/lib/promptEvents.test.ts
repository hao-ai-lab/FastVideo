import { describe, expect, it } from 'vitest';

import {
  prependPromptEvent,
  updatePromptEvent,
} from './promptEvents';

interface PromptEvent {
  promptId: string;
  status: string;
  source?: string;
}

describe('updatePromptEvent', () => {
  it('updates only the matching prompt event', () => {
    const events: PromptEvent[] = [
      { promptId: 'a', status: 'submitted' },
      { promptId: 'b', status: 'submitted' },
    ];

    const updated = updatePromptEvent(events, 'b', {
      status: 'ready',
      source: 'enhanced',
    });

    expect(updated).toEqual([
      { promptId: 'a', status: 'submitted' },
      { promptId: 'b', status: 'ready', source: 'enhanced' },
    ]);
  });
});

describe('prependPromptEvent', () => {
  it('prepends new event', () => {
    const events: PromptEvent[] = [{ promptId: 'a', status: 'submitted' }];
    const next = prependPromptEvent(events, {
      promptId: 'b',
      status: 'submitted',
    });

    expect(next[0].promptId).toBe('b');
    expect(next[1].promptId).toBe('a');
  });

  it('caps list length to 24 entries', () => {
    const events: PromptEvent[] = Array.from({ length: 24 }, (_, i: number) => ({
      promptId: `p-${i}`,
      status: 'submitted',
    }));

    const next = prependPromptEvent(events, {
      promptId: 'new',
      status: 'submitted',
    });

    expect(next).toHaveLength(24);
    expect(next[0].promptId).toBe('new');
    expect(next.some((item: any) => item.promptId === 'p-23')).toBe(false);
  });

  it('never drops steering scene events when capping', () => {
    // 30 scenes interleaved with 30 other events — well past the cap.
    let events: Record<string, any>[] = [];
    for (let i = 0; i < 30; i += 1) {
      events = prependPromptEvent(events, {
        promptId: `scene-${i}`,
        status: 'submitted',
        steeringUserPrompt: true,
        rawText: `scene ${i}`,
      });
      events = prependPromptEvent(events, {
        promptId: `other-${i}`,
        status: 'submitted',
      });
    }

    const scenes = events.filter((e) => e.steeringUserPrompt);
    expect(scenes).toHaveLength(30);
    // Oldest-first scene order (and therefore numbering) is stable and complete.
    expect(scenes[scenes.length - 1].promptId).toBe('scene-0');
    expect(scenes[0].promptId).toBe('scene-29');
    // Non-scene events are still capped, oldest dropped first.
    const others = events.filter((e) => !e.steeringUserPrompt);
    expect(others.length).toBeLessThanOrEqual(24);
    expect(others.some((e) => e.promptId === 'other-0')).toBe(false);
    expect(others[0].promptId).toBe('other-29');
  });
});
