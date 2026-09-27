'use client';

import * as React from 'react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { attachedLastFrames } from '@/lib/continuation';
import { planSceneQueue, QUEUEABLE_STATUSES, type SceneQueuePlan } from '@/lib/sceneQueue';
import type { Scene, SceneClip } from '@/lib/scenes';

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
const clipNumbers = (clips: SceneClip[]) => clips.map((c) => c.position).join(', ');

export interface SceneQueueDialogProps {
  scene: Scene | null;
  onConfirm: (plan: SceneQueuePlan) => Promise<void>;
  onClose: () => void;
}

/**
 * Confirm queueing a whole scene: which clips are queued, and whether each
 * starts from the previous clip's last frame. Nothing is sent until "Queue".
 */
export function SceneQueueDialog({ scene, onConfirm, onClose }: SceneQueueDialogProps) {
  // Off by default: the per-clip "Use last frame" boxes are the source of truth, and
  // this would add links to every clip that doesn't have one.
  const [chain, setChain] = React.useState(false);
  const [busy, setBusy] = React.useState(false);

  // Each opening starts from the default choice.
  React.useEffect(() => {
    if (scene) setChain(false);
  }, [scene]);

  const plan = React.useMemo(() => (scene ? planSceneQueue(scene.clips, { chain }) : null), [scene, chain]);
  const alreadyLinked = React.useMemo(
    () =>
      scene
        ? scene.clips.filter(
            (c) => QUEUEABLE_STATUSES.includes(c.job.status) && attachedLastFrames(c.job.references).length > 0,
          )
        : [],
    [scene],
  );
  // Whether the option would do anything is independent of the checkbox.
  const linkable = React.useMemo(
    () => (scene ? planSceneQueue(scene.clips, { chain: true }) : null),
    [scene],
  );

  async function confirm() {
    if (!plan) return;
    setBusy(true);
    try {
      await onConfirm(plan);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={!!scene} onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Queue {scene?.title}</DialogTitle>
          <DialogDescription>
            {plan && plan.queue.length > 0
              ? `${plural(plan.queue.length, 'clip')} will be queued and run one at a time, in order.`
              : 'There is nothing in this scene to queue.'}
          </DialogDescription>
        </DialogHeader>

        {plan && linkable && (
          <div className="flex flex-col gap-3 text-sm">
            {alreadyLinked.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Clip {clipNumbers(alreadyLinked)} already start{alreadyLinked.length === 1 ? 's' : ''} from a previous
                frame (as you set on the clip{alreadyLinked.length === 1 ? '' : 's'}). Other clips are queued as they are.
              </p>
            )}
            {linkable.chain.length > 0 && (
              <label className="flex items-start gap-2">
                <input
                  type="checkbox"
                  checked={chain}
                  onChange={(e) => setChain(e.target.checked)}
                  className="mt-1 size-4"
                />
                <span>
                  Start each clip from the previous clip&apos;s last frame
                  <span className="block text-xs text-muted-foreground">
                    Also links {plural(linkable.chain.length, 'clip')} that don&apos;t have it (
                    {clipNumbers(linkable.chain.map((c) => c.clip))}), rewriting their prompts. Each waits for the clip
                    before it to finish.
                  </span>
                </span>
              </label>
            )}
            {chain && plan.cannotChain.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Clip {clipNumbers(plan.cannotChain.map((c) => c.clip))} can&apos;t be linked automatically (its prompt
                isn&apos;t in the six-section format) and will start without the previous frame.
              </p>
            )}
            {plan.skipped.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Skipping{' '}
                {plan.skipped.map((s) => `clip ${s.clip.position} (${s.reason})`).join(', ')}.
              </p>
            )}
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button type="button" onClick={confirm} disabled={busy || !plan || plan.queue.length === 0}>
            {busy ? 'Queueing…' : plan ? `Queue ${plural(plan.queue.length, 'clip')}` : 'Queue'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
