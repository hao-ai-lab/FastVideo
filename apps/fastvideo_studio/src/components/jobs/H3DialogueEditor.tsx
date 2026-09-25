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
import { Textarea } from '@/components/ui/textarea';
import { H3DialoguePreview } from '@/components/jobs/H3DialoguePreview';
import { H3TagPalette } from '@/components/jobs/H3TagPalette';
import { useUndoableText } from '@/hooks/useUndoableText';
import { applyH3Preset, applyH3Tag, type H3Preset, type H3Tag } from '@/lib/h3Tags';
import {
  SPEECH_RATES,
  dialogueTree,
  estimateDuration,
  type DialogueNode,
  type SpeechRate,
} from '@/lib/h3DialogueEstimate';
import { cn } from '@/lib/utils';

export interface H3DialogueEditorProps {
  open: boolean;
  /** Seeded from whichever prompt field the editor was opened for. */
  initialValue: string;
  onApply: (value: string) => void;
  onClose: () => void;
}

/** Every `<d>...</d>` block's inner text -- what's actually spoken, ignoring surrounding shot/camera prose. */
function extractDialogueBlocks(text: string): string[] {
  const blocks: string[] = [];
  const re = /<d>([\s\S]*?)<\/d>/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) blocks.push(m[1].replace(/^\s*\[[^\]]*\]\s*/, '')); // drop a leading [Language] marker
  return blocks;
}

/**
 * Full-screen dialogue editor for MiniMax-H3 speech tags: the same tag
 * registry as `H3TagConsole`, plus a duration estimate (scoped to `<d>`
 * blocks so scene-description prose doesn't inflate it), a styled
 * performance preview, and emotion presets. Opens with a copy of the target
 * field's text; `onApply` hands the edited text back, `onClose` discards it.
 * The shot list editor reuses the same palette, preview, and undo history.
 */
export function H3DialogueEditor({ open, initialValue, onApply, onClose }: H3DialogueEditorProps) {
  const { text, type, commit, undo, redo, reset, canUndo, canRedo } = useUndoableText(initialValue);
  const [pace, setPace] = React.useState<SpeechRate>('Natural');
  const [calib, setCalib] = React.useState(1);
  const [actualSeconds, setActualSeconds] = React.useState('');
  const taRef = React.useRef<HTMLTextAreaElement>(null);
  const pending = React.useRef<[number, number] | null>(null);

  function restoreCaretAtEnd(restored: string | null) {
    if (restored != null) pending.current = [restored.length, restored.length];
  }

  // Re-seed from the field's current value each time the editor opens, the
  // same "seed on the open transition only" guard CreateJobModal itself
  // uses -- otherwise an in-progress edit here could be clobbered by a
  // stray parent re-render while open. History resets with it: undo
  // shouldn't reach back into a previous time this editor was opened.
  const wasOpenRef = React.useRef(false);
  React.useEffect(() => {
    if (open && !wasOpenRef.current) {
      reset(initialValue);
      setActualSeconds('');
      setCalib(1);
    }
    wasOpenRef.current = open;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  React.useEffect(() => {
    if (pending.current && taRef.current) {
      const [s, e] = pending.current;
      taRef.current.focus();
      taRef.current.setSelectionRange(s, e);
      pending.current = null;
    }
  });

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'z') return;
    e.preventDefault();
    restoreCaretAtEnd(e.shiftKey ? redo() : undo());
  }

  function insertTag(tag: H3Tag) {
    const ta = taRef.current;
    const result = applyH3Tag(text, ta?.selectionStart ?? text.length, ta?.selectionEnd ?? text.length, tag);
    commit(result.value);
    pending.current = [result.start, result.end];
  }

  function applyPreset(p: H3Preset) {
    const ta = taRef.current;
    const result = applyH3Preset(text, ta?.selectionStart ?? text.length, ta?.selectionEnd ?? text.length, p);
    commit(result.value);
    pending.current = [result.start, result.end];
  }

  const dialogueBlocks = React.useMemo(() => extractDialogueBlocks(text), [text]);
  // Falls back to the whole field only when it contains no <d> blocks at
  // all yet -- e.g. a fresh field the user is about to write dialogue into.
  const segments = dialogueBlocks.length > 0 ? dialogueBlocks : [text];
  const estimates = React.useMemo(
    () => segments.map((s) => estimateDuration(s, SPEECH_RATES[pace], calib)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [segments.join('\n'), pace, calib],
  );
  const totalSeconds = estimates.reduce((sum, e) => sum + e.total, 0);

  function calibrate() {
    const actual = parseFloat(actualSeconds);
    if (!actual || actual <= 0 || segments.length === 0) return;
    const base = segments.reduce((sum, s) => sum + estimateDuration(s, SPEECH_RATES[pace], 1).total, 0);
    if (base > 0) setCalib(actual / base);
  }

  /* ---- rough browser-voice preview, for rhythm/pacing only ---- */
  const synth = typeof window !== 'undefined' ? window.speechSynthesis : null;
  const [playing, setPlaying] = React.useState(false);
  const playTokenRef = React.useRef({ cancelled: true });

  function stopPreview() {
    playTokenRef.current.cancelled = true;
    synth?.cancel();
    setPlaying(false);
  }

  function playPreview() {
    if (!synth) return;
    stopPreview();
    const token = { cancelled: false };
    playTokenRef.current = token;
    setPlaying(true);
    const VOX: Record<string, string> = {
      laughs: 'ha ha ha', chuckle: 'heh', sighs: 'haah', gasp: 'ah', uh: 'uh', mhm: 'mhm', phew: 'phew',
    };
    const atoms: { say?: string; gap?: number }[] = [];
    for (const [i, seg] of segments.entries()) {
      if (i > 0) atoms.push({ gap: 0.6 });
      const walk = (nodes: DialogueNode[]) => {
        for (const n of nodes) {
          if (n.t === 'text') {
            const t = n.v.replace(/\s+/g, ' ').trim();
            if (t) atoms.push({ say: t });
          } else if (n.t === 'point') {
            atoms.push(VOX[n.name] ? { say: VOX[n.name] } : { gap: 0.25 });
          } else {
            if (n.name === 'humming') atoms.push({ say: 'hmm hmm' });
            else walk(n.children);
          }
        }
      };
      walk(dialogueTree(seg));
    }
    let i = 0;
    const step = () => {
      if (token.cancelled) return;
      if (i >= atoms.length) {
        setPlaying(false);
        return;
      }
      const a = atoms[i++];
      if (a.gap != null) {
        setTimeout(() => { if (!token.cancelled) step(); }, a.gap * 1000);
        return;
      }
      const u = new SpeechSynthesisUtterance(a.say);
      u.rate = 0.95;
      u.onend = () => { if (!token.cancelled) step(); };
      u.onerror = () => { if (!token.cancelled) step(); };
      synth.speak(u);
    };
    step();
  }

  React.useEffect(() => () => stopPreview(), []); // eslint-disable-line react-hooks/exhaustive-deps

  function handleApply() {
    onApply(text);
    onClose();
  }

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="flex h-[92vh] max-h-[920px] w-[96vw] max-w-[1180px] flex-col gap-0 overflow-hidden p-0">
        <DialogHeader className="border-b border-border px-5 py-4">
          <DialogTitle>Dialogue editor</DialogTitle>
          <DialogDescription>
            Click a tag to insert it at the cursor, or select text first to wrap it. Applying writes the edited text
            back into the field you opened this from.
          </DialogDescription>
        </DialogHeader>

        <div className="flex min-h-0 flex-1">
          {/* palette */}
          <H3TagPalette
            onInsertTag={insertTag}
            onApplyPreset={applyPreset}
            className="w-64 shrink-0 overflow-y-auto border-r border-border p-3"
          />

          {/* editor + preview */}
          <div className="flex min-w-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
            {/* duration estimate */}
            <div className="rounded-xl border border-border bg-secondary/30 p-4">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <div className="flex items-baseline gap-3">
                  <span className="text-2xl font-semibold tracking-tight text-foreground">
                    {totalSeconds.toFixed(1)}<span className="ml-1 text-sm font-normal text-muted-foreground">s spoken</span>
                  </span>
                  <span className="font-mono text-xs text-muted-foreground">
                    {dialogueBlocks.length > 0
                      ? `across ${dialogueBlocks.length} <d> block${dialogueBlocks.length === 1 ? '' : 's'}`
                      : 'no <d> blocks yet — estimating the whole field'}
                  </span>
                </div>
                <div className="flex items-center gap-1 rounded-lg border border-border bg-card p-1">
                  {(Object.keys(SPEECH_RATES) as SpeechRate[]).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setPace(p)}
                      className={cn(
                        'rounded-md px-2.5 py-1 text-xs transition-colors',
                        pace === p ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:bg-accent',
                      )}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              {segments.length > 1 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {estimates.map((e, i) => (
                    <span key={i} className="rounded-md border border-border bg-card px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                      block {i + 1}: {e.total.toFixed(1)}s
                    </span>
                  ))}
                </div>
              )}

              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span title="Enter the real length of a clip you already generated, to tune the estimate to your setup">
                  calibrate
                </span>
                <input
                  value={actualSeconds}
                  onChange={(e) => setActualSeconds(e.target.value)}
                  placeholder="actual s"
                  inputMode="decimal"
                  className="w-16 rounded-md border border-input bg-card px-2 py-1 text-foreground"
                />
                <Button type="button" variant="outline" size="sm" onClick={calibrate}>Set</Button>
                {Math.abs(calib - 1) > 0.001 && (
                  <button
                    type="button"
                    onClick={() => { setCalib(1); setActualSeconds(''); }}
                    className="font-mono text-muted-foreground/80 hover:text-foreground"
                    title="Reset calibration"
                  >
                    ×{calib.toFixed(2)}
                  </button>
                )}
              </div>
            </div>

            {/* audio preview */}
            {synth && (
              <div className="flex items-center gap-3 rounded-xl border border-border bg-secondary/30 p-3">
                <Button type="button" size="sm" onClick={playing ? stopPreview : playPreview}>
                  {playing ? '■ Stop' : '▶ Play preview'}
                </Button>
                <span className="text-xs text-muted-foreground">
                  A stand-in browser voice for rhythm only — it won&apos;t match MiniMax&apos;s voice.
                </span>
              </div>
            )}

            {/* the field itself */}
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-foreground">Text</span>
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => restoreCaretAtEnd(undo())}
                  disabled={!canUndo}
                  title="Undo (⌘/Ctrl+Z)"
                >
                  ↶ Undo
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => restoreCaretAtEnd(redo())}
                  disabled={!canRedo}
                  title="Redo (⌘/Ctrl+Shift+Z)"
                >
                  ↷ Redo
                </Button>
              </div>
            </div>
            <Textarea
              ref={taRef}
              value={text}
              onChange={(e) => type(e.target.value)}
              onKeyDown={handleKeyDown}
              spellCheck={false}
              rows={8}
              className="flex-1 font-mono text-sm"
            />

            {/* performance preview */}
            <div>
              <div className="mb-2 text-xs font-semibold text-foreground">Performance preview</div>
              <H3DialoguePreview
                text={text}
                className="rounded-xl border border-border bg-card p-4 text-sm leading-7 text-foreground"
              />
            </div>
          </div>
        </div>

        <DialogFooter className="border-t border-border px-5 py-3">
          <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
          <Button type="button" onClick={handleApply}>Apply to field</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
