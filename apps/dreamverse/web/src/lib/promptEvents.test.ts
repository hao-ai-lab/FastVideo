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
});
