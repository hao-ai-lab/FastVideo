'use client';

import * as React from 'react';
import { toast } from 'sonner';

import { BeforeAfterGrade } from '@/components/scenes/BeforeAfterGrade';
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
import { Slider } from '@/components/ui/slider';
import { getJobVideoUrl, restoreJobVideo, trimJob } from '@/lib/api';
import { BRIGHTNESS_BOUNDS, CONTRAST_BOUNDS, NEUTRAL_GRADE, SATURATION_BOUNDS } from '@/lib/grading';
import { clipSeconds, DEFAULT_FPS, formatClock } from '@/lib/scenes';
import type { GradeSegment, Job } from '@/lib/types';
import { cn } from '@/lib/utils';

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

/** A section as edited locally -- same shape as GradeSegment plus a stable React key. */
interface EditableSection extends GradeSegment {
  key: string;
}

let sectionSeq = 0;
function makeSectionKey(): string {
  return `section-${Date.now()}-${sectionSeq++}`;
}

const NEUTRAL_SECTION: Omit<EditableSection, 'key'> = { end_seconds: null, ...NEUTRAL_GRADE };

/** The clip's last-applied sections, or one neutral section covering everything if it was never edited. */
function sectionsFromJob(job: Job): EditableSection[] {
  const segments = job.edit_segments;
  if (segments && segments.length > 0) {
    return segments.map((s) => ({ ...s, key: makeSectionKey() }));
  }
  return [{ ...NEUTRAL_SECTION, key: makeSectionKey() }];
}

// A split closer than this to an existing edge would leave a sliver too thin
// to be a meaningful section (and trim.py would likely refuse it outright).
const MIN_SPLIT_MARGIN_SECONDS = 0.1;

/**
 * Cut a finished clip to a chosen in/out range and grade it section by
 * section -- like a razor tool: "Split at current time" cuts the clip where
 * the preview is paused, and each resulting section gets its own brightness,
 * contrast and saturation. Previous/Next frame step the preview by exactly
 * one frame (at the clip's own fps), for lining a cut up precisely -- the
 * native scrub bar alone isn't frame-accurate. The server always keeps the
 * untouched original and re-renders from it with the full set of values
 * shown here, so range and sections are applied together and neither is
 * lost by changing the other --
 * as long as both are set the way you want them before you click Apply. The
 * fields keep whatever you last applied while this dialog stays open for the
 * same clip, so a second, smaller tweak doesn't also reset the range. Opening
 * it fresh for a clip (or reopening it later) shows exactly what's currently
 * applied -- the server records the range/sections of the last edit on the
 * job itself (`edit_*`), so this isn't guessing from the clip's current
 * length. A never-edited clip shows one neutral section covering its full
 * length. Restoring clears the fields back to that same neutral state, since
 * that's what it just did to the video.
 */
export function TrimClipDialog({ clip, videoVersion, onClose, onChanged }: TrimClipDialogProps) {
  const duration = clip ? clipSeconds(clip) : 0;
  const fps = clip?.fps && clip.fps > 0 ? clip.fps : DEFAULT_FPS;
  const [start, setStart] = React.useState('0');
  const [end, setEnd] = React.useState('');
  const [sections, setSections] = React.useState<EditableSection[]>([]);
  const [previewIndex, setPreviewIndex] = React.useState(0);
  const [busy, setBusy] = React.useState<'apply' | 'restore' | null>(null);
  const [displayTime, setDisplayTime] = React.useState(0);

  // Reset the fields only when a *different* clip opens -- onChanged() re-renders
  // this dialog with a fresh job object for the same edit, which must not reset
  // what the user is about to apply next.
  const openClipId = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (clip && clip.id !== openClipId.current) {
      openClipId.current = clip.id;
      setStart(String(clip.edit_start_seconds ?? 0));
      setEnd(clip.edit_end_seconds != null ? clip.edit_end_seconds.toFixed(2) : clipSeconds(clip).toFixed(2));
      setSections(sectionsFromJob(clip));
      setPreviewIndex(0);
    } else if (!clip) {
      openClipId.current = null;
    }
  }, [clip]);

  // This dialog is always mounted by the page (it just renders closed when
  // `clip` is null), so a `useEffect(fn, [])` here would only ever fire once,
  // the very first time -- before `clip` is ever set and before the <video>
  // below exists at all -- and never again on later opens. A callback ref
  // fires on every real mount/unmount of the DOM node itself, however many
  // times that happens across this component's one long lifetime, which is
  // what both of these actually need:
  //  - setting the *initial* muted state imperatively, so it doesn't fight the
  //    user's own unmute via the native controls (`muted` as a plain JSX prop
  //    is controlled, and React would re-apply it on every re-render instead)
  //  - keeping the frame-step readout in sync with wherever the video is,
  //    whether that's a step button, a Preview click, or the native scrub bar
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const setVideoNode = React.useCallback((node: HTMLVideoElement | null) => {
    videoRef.current = node;
    if (!node) return;
    node.muted = true;
    const onTimeChange = () => setDisplayTime(node.currentTime);
    node.addEventListener('timeupdate', onTimeChange);
    node.addEventListener('seeked', onTimeChange);
    return () => {
      node.removeEventListener('timeupdate', onTimeChange);
      node.removeEventListener('seeked', onTimeChange);
    };
  }, []);

  if (!clip) {
    return <Dialog open={false} onOpenChange={() => {}} />;
  }

  const startSeconds = Number(start);
  const endSeconds = end.trim() === '' ? undefined : Number(end);

  const rangeError =
    !Number.isFinite(startSeconds) || startSeconds < 0
      ? 'The start must be 0 or later.'
      : endSeconds !== undefined && (!Number.isFinite(endSeconds) || endSeconds <= startSeconds)
        ? 'The end must come after the start.'
        : null;
  // Color has no equivalent error: the sliders can't produce a value outside their bounds,
  // and sections are only ever built by split/merge below, which keep them in valid order.
  const invalid = !!rangeError;

  /** The [start, end) of section `i`, in the kept range's own seconds -- the same timeline the preview plays. */
  function sectionBounds(i: number): [number, number] {
    const start = i === 0 ? 0 : (sections[i - 1].end_seconds ?? duration);
    const end = sections[i].end_seconds ?? duration;
    return [start, end];
  }

  function updateSection(i: number, patch: Partial<Omit<EditableSection, 'key' | 'end_seconds'>>) {
    setSections((prev) => prev.map((s, idx) => (idx === i ? { ...s, ...patch } : s)));
  }

  /**
   * Seeks the preview to `target`, clamped to the clip's length. A seek
   * requested before the video has loaded metadata (readyState is still
   * HAVE_NOTHING) is unreliable -- browsers commonly drop or silently reset it
   * once metadata actually arrives -- so this defers to 'loadedmetadata' when
   * that's the case, instead of jumping to the wrong spot (often 0).
   */
  function seekTo(target: number) {
    const video = videoRef.current;
    if (!video) return;
    const clamped = Math.max(0, Math.min(duration, target));
    const apply = () => {
      video.currentTime = clamped;
    };
    if (video.readyState >= 1) {
      apply();
    } else {
      video.addEventListener('loadedmetadata', apply, { once: true });
    }
  }

  /** Steps the preview by exactly one frame (at the clip's own fps), pausing playback first. */
  function stepFrame(direction: 1 | -1) {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    seekTo(video.currentTime + direction / fps);
  }

  function splitAtCurrentTime() {
    const video = videoRef.current;
    if (!video) return;
    const t = video.currentTime;
    for (let i = 0; i < sections.length; i++) {
      const [sectionStart, sectionEnd] = sectionBounds(i);
      if (t <= sectionStart + MIN_SPLIT_MARGIN_SECONDS || t >= sectionEnd - MIN_SPLIT_MARGIN_SECONDS) continue;
      setSections((prev) => {
        const next = [...prev];
        next.splice(
          i,
          1,
          { ...prev[i], end_seconds: t },
          { ...NEUTRAL_SECTION, end_seconds: prev[i].end_seconds, key: makeSectionKey() },
        );
        return next;
      });
      setPreviewIndex(i + 1);
      return;
    }
    toast.error('Pause the preview somewhere inside a section (away from its edges) to split it there.');
  }

  function mergeWithNext(i: number) {
    setSections((prev) => {
      const next = [...prev];
      next[i] = { ...next[i], end_seconds: next[i + 1].end_seconds };
      next.splice(i + 1, 1);
      return next;
    });
    setPreviewIndex((p) => Math.min(p, i));
  }

  /** Seeks the preview to this section's start and previews it; BeforeAfterGrade re-captures on 'seeked'. */
  function previewSection(i: number) {
    const [sectionStart] = sectionBounds(i);
    seekTo(sectionStart);
    setPreviewIndex(i);
  }

  /** Keeps the preview panel's selected section in sync with wherever the video actually is. */
  function handleFrameCaptured(t: number) {
    for (let i = 0; i < sections.length; i++) {
      const [, sectionEnd] = sectionBounds(i);
      if (t < sectionEnd || i === sections.length - 1) {
        setPreviewIndex(i);
        return;
      }
    }
  }

  async function apply() {
    if (!clip || invalid) return;
    setBusy('apply');
    try {
      const updated = await trimJob(clip.id, {
        startSeconds,
        endSeconds,
        segments: sections.map(({ key: _key, ...segment }) => segment),
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
      // Restoring doesn't open a *different* clip, so the reset effect above
      // won't fire on its own -- but the fields have to reflect it too, since
      // that's exactly what just happened to the video.
      setStart('0');
      setEnd(clipSeconds(updated).toFixed(2));
      setSections([{ ...NEUTRAL_SECTION, key: makeSectionKey() }]);
      setPreviewIndex(0);
      toast.success('Restored the original video.');
      onChanged(updated);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't restore the original video.");
    } finally {
      setBusy(null);
    }
  }

  const preview = sections[previewIndex] ?? sections[0];

  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-lg overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit {clip.name?.trim() || clip.id.slice(0, 8)}</DialogTitle>
          <DialogDescription>
            Currently {formatClock(duration)}. Editing keeps the untouched original, so this can be adjusted or undone.
          </DialogDescription>
        </DialogHeader>

        <video
          ref={setVideoNode}
          src={getJobVideoUrl(clip.id, videoVersion)}
          aria-label={`Preview of ${clip.name ?? clip.id}`}
          className="max-h-64 w-full rounded-md border border-border bg-black"
          controls
          playsInline
          preload="metadata"
          crossOrigin="anonymous"
        />

        <div className="flex items-center justify-between gap-2">
          <div className="flex gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => stepFrame(-1)} disabled={!!busy}>
              ◂ Previous frame
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={() => stepFrame(1)} disabled={!!busy}>
              Next frame ▸
            </Button>
          </div>
          <span className="text-xs tabular-nums text-muted-foreground">
            Frame {Math.round(displayTime * fps)} · {displayTime.toFixed(2)}s
          </span>
        </div>

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

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Sections</p>
            <Button type="button" variant="outline" size="sm" onClick={splitAtCurrentTime} disabled={!!busy}>
              Split at current time
            </Button>
          </div>

          {sections.map((section, i) => {
            const [sectionStart, sectionEnd] = sectionBounds(i);
            return (
              <div
                key={section.key}
                className={cn(
                  'flex flex-col gap-2 rounded-md border p-3',
                  i === previewIndex ? 'border-accent-blue' : 'border-border',
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-muted-foreground">
                    Section {i + 1} · {formatClock(sectionStart)}–{section.end_seconds == null ? 'end' : formatClock(sectionEnd)}
                  </span>
                  <div className="flex gap-3">
                    <button
                      type="button"
                      onClick={() => previewSection(i)}
                      disabled={!!busy}
                      className="text-xs text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Preview
                    </button>
                    {i < sections.length - 1 && (
                      <button
                        type="button"
                        onClick={() => mergeWithNext(i)}
                        disabled={!!busy}
                        className="text-xs text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Merge with next
                      </button>
                    )}
                  </div>
                </div>
                <GradeSlider
                  id={`grade-brightness-${section.key}`}
                  label="Brightness"
                  ariaLabel={`Section ${i + 1} Brightness`}
                  bounds={BRIGHTNESS_BOUNDS}
                  step={5}
                  value={section.brightness}
                  onChange={(v) => updateSection(i, { brightness: v })}
                  format={(v) => String(v)}
                  disabled={!!busy}
                />
                <GradeSlider
                  id={`grade-contrast-${section.key}`}
                  label="Contrast"
                  ariaLabel={`Section ${i + 1} Contrast`}
                  bounds={CONTRAST_BOUNDS}
                  step={0.1}
                  value={section.contrast}
                  onChange={(v) => updateSection(i, { contrast: v })}
                  format={(v) => `${v.toFixed(1)}x`}
                  disabled={!!busy}
                />
                <GradeSlider
                  id={`grade-saturation-${section.key}`}
                  label="Saturation"
                  ariaLabel={`Section ${i + 1} Saturation`}
                  bounds={SATURATION_BOUNDS}
                  step={0.1}
                  value={section.saturation}
                  onChange={(v) => updateSection(i, { saturation: v })}
                  format={(v) => `${v.toFixed(1)}x`}
                  disabled={!!busy}
                />
              </div>
            );
          })}

          <BeforeAfterGrade
            videoRef={videoRef}
            brightness={preview?.brightness ?? NEUTRAL_GRADE.brightness}
            contrast={preview?.contrast ?? NEUTRAL_GRADE.contrast}
            saturation={preview?.saturation ?? NEUTRAL_GRADE.saturation}
            onFrameCaptured={handleFrameCaptured}
          />
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

function GradeSlider({
  id,
  label,
  ariaLabel,
  bounds,
  step,
  value,
  onChange,
  format,
  disabled,
}: {
  id: string;
  label: string;
  ariaLabel: string;
  bounds: readonly [number, number];
  step: number;
  value: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 text-sm">
      <div className="flex items-center justify-between">
        <Label htmlFor={id}>{label}</Label>
        <span className="text-xs tabular-nums text-muted-foreground">{format(value)}</span>
      </div>
      <Slider
        id={id}
        min={bounds[0]}
        max={bounds[1]}
        step={step}
        value={[value]}
        onValueChange={(v) => onChange(v[0])}
        disabled={disabled}
        aria-label={ariaLabel}
      />
    </div>
  );
}
