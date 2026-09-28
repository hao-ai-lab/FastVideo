'use client';

import * as React from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { getJobVideoUrl, restoreJobVideo, trimJob } from '@/lib/api';
import { clipSeconds, formatClock } from '@/lib/scenes';
import type { Job } from '@/lib/types';

export interface TrimClipDialogProps {
  /** the clip to edit, or null when closed */
  clip: Job | null;
  /**
   * Cache-busts the preview after an edit (the video keeps the same URL, only
   * the file changes). Owned by the caller, not this component -- a version
   * counter kept here gets reset by the very re-render `onChanged` causes
   * (a fresh `clip` object comes back from the edit), before the browser
   * ever sees the new URL.
   */
  videoVersion: number;
  onClose: () => void;
  /** called with the job as the edit/restore left it, so the caller can refresh */
  onChanged: (job: Job) => void;
}

const NEUTRAL = { brightness: '0', contrast: '1', saturation: '1' };
// Mirrors trim.py's BRIGHTNESS_RANGE / CONTRAST_RANGE / SATURATION_RANGE.
const BRIGHTNESS_BOUNDS = [-100, 100] as const;
const CONTRAST_BOUNDS = [0, 3] as const;
const SATURATION_BOUNDS = [0, 3] as const;

function outOfRange(value: number, [low, high]: readonly [number, number]): boolean {
  return !Number.isFinite(value) || value < low || value > high;
}

/**
 * Cut a finished clip to a chosen in/out range and adjust its brightness,
 * contrast and saturation, or undo every change back to what H3 originally
 * generated. The server always keeps the untouched original and re-renders
 * from it with the full set of values shown here, so range and color are
 * applied together and neither is lost by changing the other -- as long as
 * both are set the way you want them before you click Apply. The fields keep
 * whatever you last applied while this dialog stays open for the same clip,
 * so a second, smaller color tweak doesn't also reset the range. Opening it
 * fresh for a clip (or reopening it later) shows the clip's current length as
 * the range (so leaving color alone and applying keeps a previous trim), but
 * color always starts at neutral, since the server doesn't remember which values
 * produced the current video.
 */
export function TrimClipDialog({ clip, videoVersion, onClose, onChanged }: TrimClipDialogProps) {
  const duration = clip ? clipSeconds(clip) : 0;
  const [start, setStart] = React.useState('0');
  const [end, setEnd] = React.useState('');
  const [brightness, setBrightness] = React.useState(NEUTRAL.brightness);
  const [contrast, setContrast] = React.useState(NEUTRAL.contrast);
  const [saturation, setSaturation] = React.useState(NEUTRAL.saturation);
  const [busy, setBusy] = React.useState<'apply' | 'restore' | null>(null);

  // Reset the fields only when a *different* clip opens -- onChanged() re-renders
  // this dialog with a fresh job object for the same edit, which must not reset
  // what the user is about to apply next.
  const openClipId = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (clip && clip.id !== openClipId.current) {
      openClipId.current = clip.id;
      setStart('0');
      setEnd(clipSeconds(clip).toFixed(2));
      setBrightness(NEUTRAL.brightness);
      setContrast(NEUTRAL.contrast);
      setSaturation(NEUTRAL.saturation);
    } else if (!clip) {
      openClipId.current = null;
    }
  }, [clip]);

  // Sets the *initial* muted state only, imperatively, so it doesn't fight the
  // user's own unmute via the native controls: `muted` as a plain JSX prop is
  // controlled, and React re-applies it on every re-render of this dialog
  // (e.g. after Apply calls onChanged with a fresh job object), silently
  // re-muting a video the user just unmuted.
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  React.useEffect(() => {
    if (videoRef.current) videoRef.current.muted = true;
  }, []);

  if (!clip) {
    return <Dialog open={false} onOpenChange={() => {}} />;
  }

  const startSeconds = Number(start);
  const endSeconds = end.trim() === '' ? undefined : Number(end);
  const brightnessValue = Number(brightness);
  const contrastValue = Number(contrast);
  const saturationValue = Number(saturation);

  const rangeError =
    !Number.isFinite(startSeconds) || startSeconds < 0
      ? 'The start must be 0 or later.'
      : endSeconds !== undefined && (!Number.isFinite(endSeconds) || endSeconds <= startSeconds)
        ? 'The end must come after the start.'
        : null;
  const colorError = outOfRange(brightnessValue, BRIGHTNESS_BOUNDS)
    ? `Brightness must be between ${BRIGHTNESS_BOUNDS[0]} and ${BRIGHTNESS_BOUNDS[1]}.`
    : outOfRange(contrastValue, CONTRAST_BOUNDS)
      ? `Contrast must be between ${CONTRAST_BOUNDS[0]} and ${CONTRAST_BOUNDS[1]}.`
      : outOfRange(saturationValue, SATURATION_BOUNDS)
        ? `Saturation must be between ${SATURATION_BOUNDS[0]} and ${SATURATION_BOUNDS[1]}.`
        : null;
  const invalid = !!(rangeError || colorError);

  async function apply() {
    if (!clip || invalid) return;
    setBusy('apply');
    try {
      const updated = await trimJob(clip.id, {
        startSeconds,
        endSeconds,
        brightness: brightnessValue,
        contrast: contrastValue,
        saturation: saturationValue,
      });
      toast.success(`Applied: ${formatClock(startSeconds)}–${endSeconds === undefined ? 'end' : formatClock(endSeconds)}.`);
      onChanged(updated);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't edit the clip.");
    } finally {
      setBusy(null);
    }
  }

  async function restore() {
    if (!clip) return;
    setBusy('restore');
    try {
      const updated = await restoreJobVideo(clip.id);
      toast.success('Restored the original video.');
      onChanged(updated);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't restore the original video.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Edit {clip.name?.trim() || clip.id.slice(0, 8)}</DialogTitle>
          <DialogDescription>
            Currently {formatClock(duration)}. Editing keeps the untouched original, so this can be adjusted or undone.
          </DialogDescription>
        </DialogHeader>

        <video
          ref={videoRef}
          src={getJobVideoUrl(clip.id, videoVersion)}
          aria-label={`Preview of ${clip.name ?? clip.id}`}
          className="max-h-64 w-full rounded-md border border-border bg-black"
          controls
          playsInline
          preload="metadata"
        />

        <div className="flex flex-col gap-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Range</p>
          <div className="flex items-end gap-3">
            <div className="flex flex-1 flex-col gap-1 text-sm">
              <Label htmlFor="trim-start">Start (seconds)</Label>
              <Input
                id="trim-start"
                type="number"
                min={0}
                step={0.1}
                value={start}
                onChange={(e) => setStart(e.target.value)}
                disabled={!!busy}
              />
            </div>
            <div className="flex flex-1 flex-col gap-1 text-sm">
              <Label htmlFor="trim-end">End (seconds, optional)</Label>
              <Input
                id="trim-end"
                type="number"
                min={0}
                step={0.1}
                placeholder="end of clip"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
                disabled={!!busy}
              />
            </div>
          </div>
          {rangeError && <p className="text-sm text-destructive">{rangeError}</p>}
        </div>

        <div className="flex flex-col gap-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Color</p>
          <div className="flex items-end gap-3">
            <div className="flex flex-1 flex-col gap-1 text-sm">
              <Label htmlFor="grade-brightness">Brightness</Label>
              <Input
                id="grade-brightness"
                type="number"
                min={BRIGHTNESS_BOUNDS[0]}
                max={BRIGHTNESS_BOUNDS[1]}
                step={5}
                value={brightness}
                onChange={(e) => setBrightness(e.target.value)}
                disabled={!!busy}
              />
            </div>
            <div className="flex flex-1 flex-col gap-1 text-sm">
              <Label htmlFor="grade-contrast">Contrast</Label>
              <Input
                id="grade-contrast"
                type="number"
                min={CONTRAST_BOUNDS[0]}
                max={CONTRAST_BOUNDS[1]}
                step={0.1}
                value={contrast}
                onChange={(e) => setContrast(e.target.value)}
                disabled={!!busy}
              />
            </div>
            <div className="flex flex-1 flex-col gap-1 text-sm">
              <Label htmlFor="grade-saturation">Saturation</Label>
              <Input
                id="grade-saturation"
                type="number"
                min={SATURATION_BOUNDS[0]}
                max={SATURATION_BOUNDS[1]}
                step={0.1}
                value={saturation}
                onChange={(e) => setSaturation(e.target.value)}
                disabled={!!busy}
              />
            </div>
          </div>
          {colorError && <p className="text-sm text-destructive">{colorError}</p>}
        </div>

        <DialogFooter className="justify-between sm:justify-between">
          <Button type="button" variant="outline" onClick={restore} disabled={!!busy}>
            {busy === 'restore' ? 'Restoring…' : 'Restore original'}
          </Button>
          <div className="flex gap-2">
            <Button type="button" variant="outline" onClick={onClose} disabled={!!busy}>
              Close
            </Button>
            <Button type="button" onClick={apply} disabled={!!busy || invalid}>
              {busy === 'apply' ? 'Applying…' : 'Apply'}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
