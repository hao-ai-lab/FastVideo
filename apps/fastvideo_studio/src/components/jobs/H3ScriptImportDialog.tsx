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
import { buildClips, parseScript, scriptSpeakers, type ImportSpeaker } from '@/lib/h3ScriptImport';
import { SHOT_TYPES, shotLines, shotProse, type Shot } from '@/lib/h3Shots';

const TYPE_LABEL = Object.fromEntries(SHOT_TYPES.map((t) => [t.id, t.label]));
const DEFAULT_CLIP_SECONDS = 11;

export interface H3ScriptImportDialogProps {
  open: boolean;
  subjectLabels: string[];
  /** shots currently in the list -- loading a clip replaces them */
  existingShotCount: number;
  onLoad: (shots: Shot[]) => void;
  onClose: () => void;
}

function preview(shot: Shot): string {
  const line = shotLines(shot).find((l) => l.text.trim());
  const text = line ? `${line.speaker ? `${line.speaker}: ` : ''}“${line.text.trim()}”` : shotProse(shot);
  return text.length > 90 ? `${text.slice(0, 89)}…` : text;
}

/**
 * Paste (or load) a "NAME: line" script and pick one clip-sized chunk of it
 * to load into the shot list. The script and speaker mapping persist between
 * openings, so working through a long scene is: load clip, apply, next clip.
 */
export function H3ScriptImportDialog({
  open,
  subjectLabels,
  existingShotCount,
  onLoad,
  onClose,
}: H3ScriptImportDialogProps) {
  const [script, setScript] = React.useState('');
  const [clipSecondsInput, setClipSecondsInput] = React.useState(String(DEFAULT_CLIP_SECONDS));
  const [overrides, setOverrides] = React.useState<Record<string, Partial<ImportSpeaker>>>({});
  const [clipIndex, setClipIndex] = React.useState(0);

  const turns = React.useMemo(() => parseScript(script), [script]);
  const names = React.useMemo(() => scriptSpeakers(turns), [turns]);

  const speakers = React.useMemo(() => {
    const map: Record<string, ImportSpeaker> = {};
    names.forEach((name, i) => {
      const o = overrides[name] ?? {};
      map[name] = {
        subject: 'subject' in o ? (o.subject ?? null) : (subjectLabels[i] ?? null),
        tag: o.tag ?? `S${i + 1}`,
      };
    });
    return map;
  }, [names, overrides, subjectLabels]);

  const parsedSeconds = Number(clipSecondsInput);
  const targetSeconds = Number.isFinite(parsedSeconds) ? Math.min(Math.max(parsedSeconds, 4), 12) : DEFAULT_CLIP_SECONDS;

  const clips = React.useMemo(
    () => (turns.length ? buildClips(turns, { speakers, targetSeconds }) : []),
    [turns, speakers, targetSeconds],
  );
  const selected = Math.min(clipIndex, Math.max(clips.length - 1, 0));
  const clip = clips[selected];

  function setOverride(name: string, patch: Partial<ImportSpeaker>) {
    setOverrides((prev) => ({ ...prev, [name]: { ...prev[name], ...patch } }));
  }

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setScript(await file.text());
    setClipIndex(0);
  }

  function handleLoad() {
    if (!clip) return;
    onLoad(clip.shots);
    setClipIndex(Math.min(selected + 1, clips.length - 1));
    onClose();
  }

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="flex max-h-[90vh] w-[94vw] max-w-[760px] flex-col gap-0 overflow-hidden p-0">
        <DialogHeader className="border-b border-border px-5 py-4">
          <DialogTitle>Import script</DialogTitle>
          <DialogDescription>
            Paste a script of NAME: line turns. It&apos;s split into clip-sized chunks; load one at a time into the
            shot list, then set the camera and descriptions.
          </DialogDescription>
        </DialogHeader>

        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-5">
          <div className="flex flex-col gap-2">
            <label htmlFor="script-import-text" className="text-xs font-semibold text-foreground">
              Script
            </label>
            <Textarea
              id="script-import-text"
              value={script}
              onChange={(e) => {
                setScript(e.target.value);
                setClipIndex(0);
              }}
              rows={8}
              spellCheck={false}
              placeholder={'MARK: First line...\n\nJORDAN: Reply...'}
              className="font-mono text-xs"
            />
            <Input
              type="file"
              accept=".md,.txt,.text"
              aria-label="Load script file"
              onChange={handleFile}
              className="h-auto py-2 file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-secondary file:px-2 file:py-1 file:text-sm file:text-secondary-foreground"
            />
          </div>

          {script.trim() && turns.length === 0 && (
            <p role="alert" className="text-sm text-destructive">
              No &quot;NAME: line&quot; turns found. Each turn should start with a speaker name and a colon.
            </p>
          )}

          {names.length > 0 && (
            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold text-foreground">Speakers</span>
              {names.map((name) => (
                <div key={name} className="grid grid-cols-[1fr_1fr_5rem] items-center gap-2">
                  <span className="truncate text-sm text-foreground">{name}</span>
                  <NativeSelect
                    aria-label={`${name} subject`}
                    value={speakers[name].subject ?? ''}
                    onChange={(e) => setOverride(name, { subject: e.target.value || null })}
                  >
                    <option value="">Not a tracked subject</option>
                    {subjectLabels.map((label) => (
                      <option key={label} value={label}>{label}</option>
                    ))}
                  </NativeSelect>
                  <Input
                    aria-label={`${name} speaker tag`}
                    value={speakers[name].tag}
                    onChange={(e) => setOverride(name, { tag: e.target.value })}
                    className="font-mono"
                  />
                </div>
              ))}
              {subjectLabels.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  No subjects are defined yet, so speakers can&apos;t be linked. Add them under Subject definitions
                  to track who appears in each shot.
                </p>
              )}
            </div>
          )}

          {turns.length > 0 && (
            <>
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                Clip length target (seconds)
                <Input
                  type="number"
                  min={4}
                  max={12}
                  step={1}
                  value={clipSecondsInput}
                  onChange={(e) => {
                    setClipSecondsInput(e.target.value);
                    setClipIndex(0);
                  }}
                  className="w-28"
                />
                <span>One generation runs about 11 s (277 frames at 24 fps), so each clip becomes its own job.</span>
              </label>

              <div className="flex flex-col gap-2">
                <label htmlFor="script-import-clip" className="text-xs font-semibold text-foreground">
                  Clip to load
                </label>
                <NativeSelect
                  id="script-import-clip"
                  value={selected}
                  onChange={(e) => setClipIndex(Number(e.target.value))}
                >
                  {clips.map((c, i) => (
                    <option key={i} value={i}>
                      Clip {i + 1} of {clips.length} · {c.shots.length} shot{c.shots.length === 1 ? '' : 's'}
                      {c.continuation ? ' · continues previous shot' : ''}
                    </option>
                  ))}
                </NativeSelect>

                {clip?.continuation && (
                  <p className="rounded-lg border border-border bg-secondary/30 px-3 py-2 text-xs text-muted-foreground">
                    This clip continues the previous clip&apos;s last shot, cut short because that speech didn&apos;t
                    fit one generation. Generate it from the previous clip&apos;s last frame (as the start Image)
                    with the same references.
                  </p>
                )}

                {clip && (
                  <ol className="flex flex-col gap-1 rounded-xl border border-border bg-secondary/20 p-3">
                    {clip.shots.map((shot, i) => (
                      <li key={shot.id} className="flex gap-2 text-xs">
                        <span className="w-6 shrink-0 font-mono text-muted-foreground">{i + 1}</span>
                        <span className="w-24 shrink-0 font-medium text-foreground">{TYPE_LABEL[shot.type]}</span>
                        <span className="min-w-0 text-muted-foreground">{preview(shot)}</span>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            </>
          )}
        </div>

        <DialogFooter className="items-center border-t border-border px-5 py-3">
          {existingShotCount > 0 && clip && (
            <span className="mr-auto text-xs text-muted-foreground">
              Replaces the {existingShotCount} shot{existingShotCount === 1 ? '' : 's'} currently in the list.
            </span>
          )}
          <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
          <Button type="button" onClick={handleLoad} disabled={!clip}>Load clip</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
