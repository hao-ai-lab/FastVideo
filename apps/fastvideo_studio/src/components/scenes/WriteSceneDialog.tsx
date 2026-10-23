'use client';

import * as React from 'react';
import { Loader2, Plus, Trash2, Upload } from 'lucide-react';
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
import { NativeSelect } from '@/components/ui/native-select';
import { Textarea } from '@/components/ui/textarea';
import {
  createJob,
  getFetchedReferenceUrl,
  getSceneDraft,
  startSceneDraft,
  uploadMedia,
  type DraftedJob,
  type MediaType,
  type SceneDraft,
  type SceneDraftReference,
} from '@/lib/api';
import { formatDuration } from '@/lib/jobTiming';

const POLL_MS = 3000;
const SCENE_NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function emptyReference(): SceneDraftReference {
  return { path: '', media_type: 'image', description: '' };
}

/** Clip length and shot count, read from the drafted job. */
function describeJob(job: DraftedJob): string {
  const seconds = (job.num_frames ?? 0) / (job.fps || 24);
  const shots = job.prompt.filter((l) => /^\[Shot \d+\]/.test(l)).length;
  return `${seconds.toFixed(1)}s · ${shots} shot${shots === 1 ? '' : 's'}`;
}

function summaryOf(job: DraftedJob): string {
  const i = job.prompt.indexOf('summary:');
  return (i >= 0 ? job.prompt[i + 1] : '')?.replace(/^\[[^\]]*\]\s*/, '') ?? '';
}

export interface WriteSceneDialogProps {
  open: boolean;
  onClose: () => void;
  /** Called after drafts are created as pending jobs. */
  onCreated: () => void;
}

/**
 * Draft a scene's jobs from a script with the scene writer agent, review the
 * drafts, then create them as pending jobs. Nothing is created or started
 * until "Create jobs".
 */
export function WriteSceneDialog({ open, onClose, onCreated }: WriteSceneDialogProps) {
  const [sceneName, setSceneName] = React.useState('');
  const [script, setScript] = React.useState('');
  const [references, setReferences] = React.useState<SceneDraftReference[]>([emptyReference()]);
  const [instructions, setInstructions] = React.useState('');
  const [findReferences, setFindReferences] = React.useState(false);
  const [draft, setDraft] = React.useState<SceneDraft | null>(null);
  const [starting, setStarting] = React.useState(false);
  const [creating, setCreating] = React.useState(false);
  const [expanded, setExpanded] = React.useState<string | null>(null);
  const [now, setNow] = React.useState(() => Date.now());

  const running = draft?.status === 'running';

  React.useEffect(() => {
    if (!running || !draft) return;
    const runId = draft.run_id;
    const poll = setInterval(async () => {
      try {
        setDraft(await getSceneDraft(runId));
      } catch {
        // Keep the last state; the next poll retries.
      }
    }, POLL_MS);
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  }, [running, draft?.run_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const usableRefs = references.filter((r) => r.path.trim());
  const nameValid = SCENE_NAME_RE.test(sceneName);
  const canStart =
    nameValid && script.trim().length > 0 && (usableRefs.length > 0 || findReferences) && !running && !starting;

  function updateReference(index: number, patch: Partial<SceneDraftReference>) {
    setReferences((refs) => refs.map((r, i) => (i === index ? { ...r, ...patch } : r)));
  }

  async function upload(index: number, file: File) {
    try {
      const { path, media_type } = await uploadMedia(file);
      updateReference(index, { path, media_type });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Upload failed');
    }
  }

  async function start() {
    setStarting(true);
    try {
      setExpanded(null);
      setDraft(
        await startSceneDraft({
          script,
          scene_name: sceneName,
          references: usableRefs.map((r) => ({ ...r, path: r.path.trim() })),
          instructions,
          find_references: findReferences,
        }),
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't start the draft.");
    } finally {
      setStarting(false);
    }
  }

  async function createAll() {
    if (!draft) return;
    setCreating(true);
    try {
      for (const job of draft.jobs) {
        await createJob({ ...job, prompt: job.prompt.join('\n') });
      }
      const n = draft.jobs.length;
      toast.success(`Created ${n} pending job${n === 1 ? '' : 's'}. Review them, then queue the scene.`);
      onCreated();
      setDraft(null);
      onClose();
    } catch (e) {
      toast.error(e instanceof Error ? `${e.message}. Jobs created before the error remain.` : "Couldn't create the jobs.");
    } finally {
      setCreating(false);
    }
  }

  const elapsed = draft ? ((draft.finished_at ?? now / 1000) - draft.started_at) : 0;

  return (
    <Dialog open={open} onOpenChange={(o) => !o && !creating && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Write a scene from a script</DialogTitle>
          <DialogDescription>
            The scene writer splits the script into clips and drafts an H3 job for each. Drafts are only created as
            pending jobs when you press Create.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4 text-sm">
          <label className="flex flex-col gap-1">
            Scene name
            <Input
              value={sceneName}
              onChange={(e) => setSceneName(e.target.value.toLowerCase())}
              placeholder="whiplash-rehearsal"
              disabled={running}
              aria-invalid={sceneName !== '' && !nameValid}
            />
            <span className="text-xs text-muted-foreground">
              Clips are named <code className="font-mono">{sceneName || 'scene'}-clip-01</code>,{' '}
              <code className="font-mono">-02</code>, … Lowercase letters, digits and dashes.
            </span>
          </label>

          <label className="flex flex-col gap-1">
            Script
            <Textarea
              value={script}
              onChange={(e) => setScript(e.target.value)}
              rows={10}
              className="font-mono text-xs"
              placeholder="Paste the scene: action lines, character names and dialogue."
              disabled={running}
            />
          </label>

          <fieldset className="flex flex-col gap-2" disabled={running}>
            <legend className="mb-1">References</legend>
            {references.map((ref, i) => (
              <div key={i} className="flex flex-wrap items-center gap-2">
                <Input
                  value={ref.path}
                  onChange={(e) => updateReference(i, { path: e.target.value })}
                  placeholder="/absolute/path/to/reference.png"
                  className="min-w-[16rem] flex-[2] font-mono text-xs"
                  aria-label={`Reference ${i + 1} path`}
                />
                <NativeSelect
                  value={ref.media_type}
                  onChange={(e) => updateReference(i, { media_type: e.target.value as MediaType })}
                  aria-label={`Reference ${i + 1} type`}
                  className="w-24"
                >
                  <option value="image">image</option>
                  <option value="audio">audio</option>
                  <option value="video">video</option>
                </NativeSelect>
                <Input
                  value={ref.description}
                  onChange={(e) => updateReference(i, { description: e.target.value })}
                  placeholder="What it is, e.g. Fletcher's face / Andrew's voice / the empty rehearsal room"
                  className="min-w-[12rem] flex-[2]"
                  aria-label={`Reference ${i + 1} description`}
                />
                <label className="inline-flex cursor-pointer items-center" title="Upload a file">
                  <Upload className="size-4 text-muted-foreground" aria-hidden />
                  <span className="sr-only">Upload reference {i + 1}</span>
                  <input
                    type="file"
                    accept="image/*,audio/*,video/*"
                    className="sr-only"
                    onChange={(e) => e.target.files?.[0] && upload(i, e.target.files[0])}
                  />
                </label>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setReferences((refs) => refs.filter((_, j) => j !== i))}
                  aria-label={`Remove reference ${i + 1}`}
                  disabled={references.length === 1}
                >
                  <Trash2 className="size-4" aria-hidden />
                </Button>
              </div>
            ))}
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="w-fit gap-1.5"
              onClick={() => setReferences((refs) => [...refs, emptyReference()])}
            >
              <Plus className="size-3.5" aria-hidden />
              Add reference
            </Button>
          </fieldset>

          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={findReferences}
              onChange={(e) => setFindReferences(e.target.checked)}
              disabled={running}
              className="mt-1 size-4"
            />
            <span>
              Find missing references online
              <span className="block text-xs text-muted-foreground">
                The writer may download photos from Wikimedia Commons for people, places or objects in the script that
                none of your references show. You can review each one, with its source and license, before creating
                jobs.
              </span>
            </span>
          </label>

          <label className="flex flex-col gap-1">
            Direction (optional)
            <Textarea
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              rows={3}
              placeholder="e.g. The first clip is just the band playing; the chair throw comes after."
              disabled={running}
            />
          </label>

          {draft && (
            <section aria-live="polite" className="flex flex-col gap-2 rounded-lg border border-border p-3">
              <div className="flex items-center gap-2">
                {running && <Loader2 className="size-4 animate-spin text-primary" aria-hidden />}
                <span className="font-medium">
                  {draft.status === 'running'
                    ? 'Drafting…'
                    : draft.status === 'completed'
                      ? 'Draft ready'
                      : 'Draft failed'}
                </span>
                <span className="text-xs text-muted-foreground">
                  {formatDuration(elapsed)} · {draft.jobs.length} clip{draft.jobs.length === 1 ? '' : 's'} written
                </span>
              </div>
              {draft.error && <p className="text-xs text-destructive">{draft.error}</p>}
              {draft.status === 'completed' && draft.jobs.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  The writer finished without writing any clips. {draft.final_response ?? ''}
                </p>
              )}
              {draft.fetched_references.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-medium">
                    Downloaded references ({draft.fetched_references.length}): check these before creating jobs
                  </span>
                  <ul className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {draft.fetched_references.map((ref) => {
                      const used = draft.jobs.some((j) => j.references?.some((r) => r.source === ref.path));
                      return (
                        <li key={ref.path} className="flex flex-col gap-1 rounded-md bg-secondary/40 p-1.5 text-[11px]">
                          {/* eslint-disable-next-line @next/next/no-img-element -- served by the studio API, not Next */}
                          <img
                            src={getFetchedReferenceUrl(draft.run_id, ref)}
                            alt={`${ref.subject}: ${ref.title}`}
                            className="aspect-square w-full rounded object-cover"
                          />
                          <span className="font-medium">
                            {ref.subject}
                            {used ? ' · used' : ''}
                          </span>
                          <a
                            href={ref.page}
                            target="_blank"
                            rel="noreferrer"
                            className="truncate text-muted-foreground underline"
                            title={ref.title}
                          >
                            {ref.title}
                          </a>
                          <span className="truncate text-muted-foreground" title={`${ref.license} · ${ref.artist}`}>
                            {ref.license} · {ref.artist}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
              <ol className="flex flex-col gap-1.5">
                {draft.jobs.map((job) => (
                  <li key={job.name} className="rounded-md bg-secondary/40 p-2">
                    <button
                      type="button"
                      className="flex w-full flex-wrap items-baseline gap-2 text-left"
                      onClick={() => setExpanded((n) => (n === job.name ? null : job.name ?? null))}
                      aria-expanded={expanded === job.name}
                    >
                      <span className="font-mono text-xs font-semibold">{job.name}</span>
                      <span className="text-xs text-muted-foreground">{describeJob(job)}</span>
                      <span className="w-full truncate text-xs text-muted-foreground">{summaryOf(job)}</span>
                    </button>
                    {expanded === job.name && (
                      <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-background p-2 text-[11px]">
                        {job.prompt.join('\n')}
                      </pre>
                    )}
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose} disabled={creating}>
            {running ? 'Hide' : 'Cancel'}
          </Button>
          {draft?.status === 'completed' && draft.jobs.length > 0 ? (
            <>
              <Button type="button" variant="outline" onClick={start} disabled={!canStart || creating}>
                Redraft
              </Button>
              <Button type="button" onClick={createAll} disabled={creating}>
                {creating ? 'Creating…' : `Create ${draft.jobs.length} pending job${draft.jobs.length === 1 ? '' : 's'}`}
              </Button>
            </>
          ) : (
            <Button type="button" onClick={start} disabled={!canStart}>
              {starting ? 'Starting…' : running ? 'Drafting…' : 'Draft clips'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
