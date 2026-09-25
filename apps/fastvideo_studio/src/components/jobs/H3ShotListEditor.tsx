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
import { Input } from '@/components/ui/input';
import { NativeSelect } from '@/components/ui/native-select';
import { Textarea } from '@/components/ui/textarea';
import { H3ShotDialogueField, type H3ShotDialogueHandle } from '@/components/jobs/H3ShotDialogueField';
import { H3ScriptImportDialog } from '@/components/jobs/H3ScriptImportDialog';
import { H3ShotStoryboard } from '@/components/jobs/H3ShotStoryboard';
import { H3TagPalette } from '@/components/jobs/H3TagPalette';
import type { H3Preset, H3Tag } from '@/lib/h3Tags';
import {
  CAMERA_MOVEMENTS,
  SHOT_TYPES,
  detailedDescriptionFromShots,
  formatCutTime,
  linePart,
  newLine,
  newShot,
  newTextPart,
  parseCutTime,
  parseDetailedDescription,
  parseSubjectLabels,
  resolveShotStarts,
  retentionAnalysisFromShots,
  shotLines,
  withFreshIds,
  type CameraMovement,
  type DialogueLine,
  type LinePart,
  type ShotPart,
  type Shot,
  type ShotType,
} from '@/lib/h3Shots';
import { useUndoableState } from '@/hooks/useUndoableState';
import { cn } from '@/lib/utils';

export interface H3ShotListEditorProps {
  open: boolean;
  /** Raw subject_definitions text -- parsed for <Subject N> labels to track per shot. */
  subjectDefinitions: string;
  initialDetailedDescription: string;
  initialRetentionAnalysis: string;
  onApply: (detailedDescription: string, retentionAnalysis: string) => void;
  onClose: () => void;
  /** Where the shots get written, for the description text. */
  targetLabel?: string;
  /** False when there is no retention analysis to keep in sync (base-format and plain prompts). */
  withRetention?: boolean;
}

/**
 * Structured camera-direction editor for MiniMax-H3: an ordered shot list
 * (type, movement, which subjects appear, action, optional dialogue) that
 * writes into detailed_description's [Shot N] blocks and keeps
 * retention_analysis's "appears in [...]" lines in sync automatically,
 * instead of hand-maintaining both fields' free text in agreement.
 */
/** Cut-in time field: blank means "automatic" (shown as the placeholder); Enter or blur commits. */
function CutTimeInput({
  value,
  auto,
  disabled,
  onCommit,
}: {
  value: number | null;
  auto: number;
  disabled?: boolean;
  onCommit: (seconds: number | null) => void;
}) {
  const show = (v: number | null) => (v == null ? '' : formatCutTime(v));
  const [text, setText] = React.useState(show(value));
  React.useEffect(() => setText(show(value)), [value]);

  function commit() {
    const trimmed = text.trim();
    if (!trimmed) return onCommit(null);
    const parsed = parseCutTime(trimmed);
    if (parsed == null) return setText(show(value));
    onCommit(parsed);
  }

  return (
    <Input
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          commit();
        }
      }}
      placeholder={disabled ? 'Opening shot' : formatCutTime(auto)}
      disabled={disabled}
      inputMode="decimal"
      className="font-mono"
    />
  );
}

interface EditorState {
  shots: Shot[];
  preamble: string;
}

export function H3ShotListEditor({
  open,
  subjectDefinitions,
  initialDetailedDescription,
  initialRetentionAnalysis,
  onApply,
  onClose,
  targetLabel = 'Detailed description',
  withRetention = true,
}: H3ShotListEditorProps) {
  // One history covers everything editable here: the shots (structure, fields and
  // every line of text) and the scene notes. Typing carries a key so a burst
  // merges into one step; everything else is a step of its own.
  const history = useUndoableState<EditorState>({ shots: [], preamble: '' });
  const { shots, preamble } = history.state;
  // Returning the same list (or text) hands the history back the same state object,
  // which it treats as "nothing changed" and does not record.
  const editShots = (fn: (shots: Shot[]) => Shot[], key: string | null = null) =>
    history.set((s) => {
      const next = fn(s.shots);
      return next === s.shots ? s : { ...s, shots: next };
    }, key);
  const setPreamble = (value: string) =>
    history.set((s) => (s.preamble === value ? s : { ...s, preamble: value }), 'preamble');
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  // The shared tag palette inserts into whichever shot's dialogue was last focused.
  const [dialogueTargetId, setDialogueTargetId] = React.useState<string | null>(null);
  const dialogueFields = React.useRef(new Map<string, H3ShotDialogueHandle>());
  const [importOpen, setImportOpen] = React.useState(false);
  const subjectLabels = React.useMemo(() => parseSubjectLabels(subjectDefinitions), [subjectDefinitions]);

  // Re-seed from the field's current text each time the editor opens, the
  // same "seed on the open transition only" guard H3DialogueEditor uses --
  // otherwise a stray parent re-render while open could clobber in-progress edits.
  const wasOpenRef = React.useRef(false);
  React.useEffect(() => {
    if (open && !wasOpenRef.current) {
      const { preamble: seededPreamble, shots: seeded } = parseDetailedDescription(
        initialDetailedDescription,
        subjectLabels,
      );
      history.reset({ preamble: seededPreamble, shots: seeded });
      setSelectedId(seeded[0]?.id ?? null);
      setDialogueTargetId(null);
    }
    wasOpenRef.current = open;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  function addShot() {
    const shot = newShot();
    editShots((prev) => [...prev, shot]);
    selectShot(shot.id);
  }

  // Storyboard panel -> jump to that shot's editor card.
  function selectShot(id: string) {
    setSelectedId(id);
    requestAnimationFrame(() =>
      document.getElementById(`shot-card-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }),
    );
  }

  function removeShot(id: string) {
    editShots((prev) => prev.filter((s) => s.id !== id));
    setDialogueTargetId((cur) => (cur === id ? null : cur));
  }

  function insertTag(tag: H3Tag) {
    if (dialogueTargetId) dialogueFields.current.get(dialogueTargetId)?.insertTag(tag);
  }

  function applyPreset(preset: H3Preset) {
    if (dialogueTargetId) dialogueFields.current.get(dialogueTargetId)?.applyPreset(preset);
  }

  const partKey = (p: ShotPart) => (p.kind === 'text' ? p.id : p.line.id);

  // An edit that changes nothing returns the same list and so records no undo step
  // (a blur that re-commits the same value, for instance).
  function updateBody(shotId: string, fn: (body: ShotPart[]) => ShotPart[], editKey: string | null = null) {
    editShots((prev) => {
      let changed = false;
      const next = prev.map((s) => {
        if (s.id !== shotId) return s;
        const body = fn(s.body);
        if (body === s.body) return s;
        changed = true;
        return { ...s, body };
      });
      return changed ? next : prev;
    }, editKey);
  }

  function addLine(shotId: string) {
    updateBody(shotId, (body) => {
      const lastSpeaker = [...body].reverse().find((p): p is LinePart => p.kind === 'line')?.line.speaker;
      return [...body, linePart(newLine({ speaker: lastSpeaker ?? 'S1' }))];
    });
  }

  function addText(shotId: string) {
    updateBody(shotId, (body) => [...body, newTextPart()]);
  }

  function removePart(shotId: string, key: string) {
    updateBody(shotId, (body) => body.filter((p) => partKey(p) !== key));
    setDialogueTargetId((cur) => (cur === key ? null : cur));
  }

  function movePart(shotId: string, key: string, dir: -1 | 1) {
    updateBody(shotId, (body) => {
      const i = body.findIndex((p) => partKey(p) === key);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= body.length) return body;
      const next = [...body];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });
  }

  function updateText(shotId: string, key: string, text: string) {
    updateBody(
      shotId,
      (body) => {
        const part = body.find((p) => p.kind === 'text' && p.id === key);
        if (!part || (part.kind === 'text' && part.text === text)) return body;
        return body.map((p) => (p === part ? { ...part, text } : p));
      },
      `text:${key}`,
    );
  }

  /** `editKey` groups typing into one undo step; omit it for a discrete edit like a tag insertion. */
  function updateLine(shotId: string, lineId: string, patch: Partial<DialogueLine>, editKey?: string) {
    updateBody(
      shotId,
      (body) => {
        const part = body.find((p) => p.kind === 'line' && p.line.id === lineId);
        if (!part || part.kind !== 'line') return body;
        const unchanged = (Object.keys(patch) as (keyof DialogueLine)[]).every((k) => part.line[k] === patch[k]);
        return unchanged ? body : body.map((p) => (p === part ? { ...part, line: { ...part.line, ...patch } } : p));
      },
      editKey ?? null,
    );
  }

  const undo = () => void history.undo();
  const redo = () => void history.redo();

  // Undo/redo for the whole list, from anywhere inside it -- including from a textbox,
  // where the browser's own undo would only know about that one box. Not while the
  // script import (nested inside this dialog) is open: its box keeps native undo.
  function handleKeyDown(e: React.KeyboardEvent) {
    if (importOpen || e.altKey || !(e.metaKey || e.ctrlKey)) return;
    const k = e.key.toLowerCase();
    if (k === 'z') {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    } else if (k === 'y' && !e.shiftKey) {
      e.preventDefault();
      redo();
    }
  }

  // An imported clip replaces the list wholesale, as one undoable step.
  function loadImportedShots(imported: Shot[]) {
    const next = withFreshIds(imported);
    editShots(() => next);
    setSelectedId(next[0]?.id ?? null);
    setDialogueTargetId(null);
  }

  const dialogueTarget = (() => {
    for (let i = 0; i < shots.length; i += 1) {
      const lines = shotLines(shots[i]);
      const li = lines.findIndex((l) => l.id === dialogueTargetId);
      if (li >= 0) return { shot: i + 1, line: li + 1, multi: lines.length > 1 };
    }
    return null;
  })();

  function moveShot(id: string, dir: -1 | 1) {
    editShots((prev) => {
      const i = prev.findIndex((s) => s.id === id);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= prev.length) return prev;
      const next = [...prev];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });
  }

  function updateShot(id: string, patch: Partial<Shot>) {
    editShots((prev) => {
      const shot = prev.find((s) => s.id === id);
      const unchanged = shot && (Object.keys(patch) as (keyof Shot)[]).every((k) => shot[k] === patch[k]);
      return !shot || unchanged ? prev : prev.map((s) => (s === shot ? { ...s, ...patch } : s));
    });
  }

  function toggleSubject(id: string, label: string) {
    editShots((prev) =>
      prev.map((s) =>
        s.id === id
          ? {
              ...s,
              subjectIds: s.subjectIds.includes(label)
                ? s.subjectIds.filter((l) => l !== label)
                : [...s.subjectIds, label],
            }
          : s,
      ),
    );
  }

  const starts = React.useMemo(() => resolveShotStarts(shots), [shots]);
  const previewDetailedDescription = React.useMemo(
    () => detailedDescriptionFromShots(shots, preamble),
    [shots, preamble],
  );
  const previewRetentionAnalysis = React.useMemo(
    () => (withRetention ? retentionAnalysisFromShots(shots, subjectLabels, initialRetentionAnalysis) : ''),
    [withRetention, shots, subjectLabels, initialRetentionAnalysis],
  );

  function handleApply() {
    onApply(previewDetailedDescription, previewRetentionAnalysis);
    onClose();
  }

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent
        onKeyDown={handleKeyDown}
        className="flex h-[92vh] max-h-[920px] w-[96vw] max-w-[1100px] flex-col gap-0 overflow-hidden p-0"
      >
        <DialogHeader className="border-b border-border px-5 py-4">
          <DialogTitle>Shot list</DialogTitle>
          <DialogDescription>
            Build the scene shot by shot. Applying writes [Shot N] blocks into {targetLabel}
            {withRetention ? ' and keeps Retention analysis\u2019s \u201cappears in\u201d lines in sync' : ''}.
          </DialogDescription>
        </DialogHeader>

        <div className="flex shrink-0 flex-col gap-2 border-b border-border bg-secondary/10 px-4 py-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-foreground">Storyboard</span>
            <span className="flex items-center gap-1.5">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={undo}
                disabled={!history.canUndo}
                title="Undo (Ctrl/Cmd+Z)"
              >
                ↶ Undo
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={redo}
                disabled={!history.canRedo}
                title="Redo (Ctrl/Cmd+Shift+Z)"
              >
                ↷ Redo
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={() => setImportOpen(true)}>
                Import script…
              </Button>
            </span>
          </div>
          <H3ShotStoryboard shots={shots} selectedId={selectedId} onSelect={selectShot} onAdd={addShot} />
        </div>

        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            Scene notes (optional, written before Shot 1 -- style, lighting, anything true of the whole clip)
            <Textarea
              value={preamble}
              onChange={(e) => setPreamble(e.target.value)}
              rows={2}
              placeholder="e.g. Live-action, cinematic, shallow depth of field."
            />
          </label>

          {shots.length === 0 && (
            <p className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
              No shots yet.
            </p>
          )}

          {shots.map((shot, i) => (
            <div
              key={shot.id}
              id={`shot-card-${shot.id}`}
              onFocusCapture={() => setSelectedId(shot.id)}
              className={cn(
                'flex flex-col gap-3 rounded-xl border bg-secondary/20 p-3',
                shot.id === selectedId ? 'border-accent-blue' : 'border-border',
              )}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-foreground">
                  Shot {i + 1}
                  {i > 0 && (
                    <span className="ml-2 font-mono text-xs font-normal text-muted-foreground">
                      cuts in at {formatCutTime(starts[i])}
                    </span>
                  )}
                </span>
                <div className="flex items-center gap-1">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => moveShot(shot.id, -1)}
                    disabled={i === 0}
                    title="Move up"
                  >
                    ↑
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => moveShot(shot.id, 1)}
                    disabled={i === shots.length - 1}
                    title="Move down"
                  >
                    ↓
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={() => removeShot(shot.id)}>
                    Remove
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                  Shot type
                  <NativeSelect
                    value={shot.type}
                    onChange={(e) => updateShot(shot.id, { type: e.target.value as ShotType, freeLead: false })}
                  >
                    {SHOT_TYPES.map((t) => (
                      <option key={t.id} value={t.id}>{t.label}</option>
                    ))}
                  </NativeSelect>
                </label>
                <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                  Camera movement
                  <NativeSelect
                    value={shot.movement}
                    onChange={(e) => updateShot(shot.id, { movement: e.target.value as CameraMovement, freeLead: false })}
                  >
                    {CAMERA_MOVEMENTS.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </NativeSelect>
                </label>
                <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                  Cut time (blank = automatic)
                  <CutTimeInput
                    value={shot.startSeconds}
                    auto={starts[i]}
                    disabled={i === 0}
                    onCommit={(startSeconds) => updateShot(shot.id, { startSeconds })}
                  />
                </label>
              </div>

              {shot.freeLead && (
                <p className="text-xs text-muted-foreground">
                  This shot keeps its original wording, so type and movement aren&apos;t written. Changing either
                  adds a generated lead-in in front of the text below.
                </p>
              )}

              {subjectLabels.length > 0 && (
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Subjects in this shot</span>
                  <div className="flex flex-wrap gap-1.5">
                    {subjectLabels.map((label) => {
                      const active = shot.subjectIds.includes(label);
                      return (
                        <button
                          key={label}
                          type="button"
                          aria-pressed={active}
                          onClick={() => toggleSubject(shot.id, label)}
                          className={cn(
                            'rounded-full border px-2 py-0.5 font-mono text-[11px] transition-colors',
                            active
                              ? 'border-accent-blue bg-accent-blue/15 text-accent-blue'
                              : 'border-input bg-card text-muted-foreground hover:bg-accent',
                          )}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="flex flex-col gap-2">
                <span className="text-xs text-muted-foreground">
                  Description and dialogue, in the order they play. Dialogue is wrapped in &lt;d&gt; on apply.
                </span>
                {(() => {
                  let lineNo = 0;
                  let textNo = 0;
                  return shot.body.map((part, bi) => {
                    const key = partKey(part);
                    const noun = part.kind === 'line' ? `line ${(lineNo += 1)}` : `text ${(textNo += 1)}`;
                    const controls = (
                      <div className="mt-6 flex shrink-0 gap-1">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => movePart(shot.id, key, -1)}
                          disabled={bi === 0}
                          aria-label={`Move ${noun} up`}
                          title="Move up"
                        >
                          ↑
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => movePart(shot.id, key, 1)}
                          disabled={bi === shot.body.length - 1}
                          aria-label={`Move ${noun} down`}
                          title="Move down"
                        >
                          ↓
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => removePart(shot.id, key)}
                          aria-label={`Remove ${noun}`}
                          title="Remove"
                        >
                          ×
                        </Button>
                      </div>
                    );

                    if (part.kind === 'text') {
                      return (
                        <div key={key} className="flex items-start gap-2">
                          <label className="flex min-w-0 flex-1 flex-col gap-1 text-xs text-muted-foreground">
                            {textNo === 1 ? 'Description (composition, action, environment)' : `Text ${textNo}`}
                            <Textarea
                              value={part.text}
                              onChange={(e) => updateText(shot.id, key, e.target.value)}
                              rows={2}
                              placeholder="What's happening in this shot"
                            />
                          </label>
                          {controls}
                        </div>
                      );
                    }

                    const line = part.line;
                    return (
                      <div key={key} className="flex items-start gap-2">
                        <Input
                          value={line.speaker}
                          onChange={(e) => updateLine(shot.id, line.id, { speaker: e.target.value }, `speaker:${line.id}`)}
                          placeholder="S1"
                          aria-label={`Line ${lineNo} speaker`}
                          className="mt-7 w-16 shrink-0 font-mono"
                        />
                        <div className="min-w-0 flex-1">
                          <H3ShotDialogueField
                            ref={(handle) => {
                              if (handle) dialogueFields.current.set(line.id, handle);
                              else dialogueFields.current.delete(line.id);
                            }}
                            value={line.text}
                            onChange={(text) => updateLine(shot.id, line.id, { text }, `line:${line.id}`)}
                            onInsert={(text) => updateLine(shot.id, line.id, { text })}
                            onFocus={() => setDialogueTargetId(line.id)}
                            active={line.id === dialogueTargetId}
                            label={`Line ${lineNo}`}
                          />
                        </div>
                        {controls}
                      </div>
                    );
                  });
                })()}
                <div className="flex gap-2">
                  <Button type="button" variant="outline" size="sm" onClick={() => addText(shot.id)}>
                    + Add text
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={() => addLine(shot.id)}>
                    + Add line
                  </Button>
                </div>
              </div>
            </div>
          ))}

          <Button type="button" variant="outline" onClick={addShot} className="self-start">
            + Add shot
          </Button>

          <div className="flex flex-col gap-2 rounded-xl border border-border bg-card p-3">
            <span className="text-xs font-semibold text-foreground">Preview</span>
            <div className="flex flex-col gap-1">
              <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                {targetLabel}
              </span>
              <pre className="whitespace-pre-wrap rounded-lg bg-secondary/30 p-2 font-mono text-[11px] text-foreground">
                {previewDetailedDescription || '—'}
              </pre>
            </div>
            {withRetention && (
            <div className="flex flex-col gap-1">
              <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Retention analysis
              </span>
              <pre className="whitespace-pre-wrap rounded-lg bg-secondary/30 p-2 font-mono text-[11px] text-foreground">
                {previewRetentionAnalysis || '—'}
              </pre>
            </div>
            )}
          </div>
        </div>

        <aside className="flex max-h-56 shrink-0 flex-col border-t border-border md:max-h-none md:w-64 md:border-l md:border-t-0">
          <p className="border-b border-border px-3 py-2 text-xs text-muted-foreground">
            {dialogueTarget
              ? `Tags insert into Shot ${dialogueTarget.shot}${dialogueTarget.multi ? `, line ${dialogueTarget.line}` : ''} dialogue`
              : 'Click into a shot\'s dialogue to insert tags'}
          </p>
          <H3TagPalette
            onInsertTag={insertTag}
            onApplyPreset={applyPreset}
            disabled={!dialogueTarget}
            className="overflow-y-auto p-3"
          />
        </aside>
        </div>

        <DialogFooter className="border-t border-border px-5 py-3">
          <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
          <Button type="button" onClick={handleApply}>Apply</Button>
        </DialogFooter>

        {/* Nested so Radix treats it as part of this dialog (see CreateJobModal). */}
      <H3ScriptImportDialog
        open={importOpen}
        subjectLabels={subjectLabels}
        existingShotCount={shots.length}
        onLoad={loadImportedShots}
        onClose={() => setImportOpen(false)}
      />
      </DialogContent>
    </Dialog>
  );
}
