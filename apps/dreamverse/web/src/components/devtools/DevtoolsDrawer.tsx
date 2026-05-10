'use client';

import React, { useMemo } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';

interface DevtoolsDrawerProps {
  editableMode?: boolean;
  demoMode?: boolean;
  sessionStarted?: boolean;
  selectedPreset?: Record<string, any> | null;
  editableSegments?: string[];
  editableCanJoin?: boolean;
  customPresetId?: string;
  customPresetLabel?: string;
  currentPromptWindowPrompts?: string[];
  appendingPromptWindow?: boolean;
  appendPromptWindowStatus?: string;
  appendPromptWindowError?: string;
  onCustomPresetIdInput?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onCustomPresetLabelInput?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onAddSegment?: () => void;
  onResetToPreset?: () => void;
  onExport?: () => void;
  onAppendPromptWindowToJson?: () => void;
  onRemoveSegment?: (index: number) => void;
  onUpdateSegment?: (index: number, value: string) => void;
  seedPrompts?: string[];
  playingSeedPromptIndex?: number | null;
  generatingSeedPromptIndex?: number | null;
  promptEvents?: Record<string, any>[];
  promptConfigEditorOpen?: boolean;
  promptConfigLoading?: boolean;
  promptConfigSaving?: boolean;
  promptConfigStatus?: string;
  promptConfigError?: string;
  nextSegmentPromptEditorOpen?: boolean;
  autoExtensionPromptEditorOpen?: boolean;
  rewriteWindowPromptEditorOpen?: boolean;
  nextSegmentSystemPromptDraft?: string;
  autoExtensionSystemPromptDraft?: string;
  rewriteWindowSystemPromptDraft?: string;
  onPromptConfigEditorToggle?: (e: React.SyntheticEvent<HTMLDetailsElement>) => void;
  onReloadPromptConfig?: () => void;
  onNextSegmentSystemPromptInput?: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onAutoExtensionSystemPromptInput?: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onRewriteWindowSystemPromptInput?: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSavePromptConfig?: () => void;
}

export default function DevtoolsDrawer({
  editableMode = false,
  demoMode = false,
  sessionStarted = false,
  selectedPreset = null,
  editableSegments = [],
  editableCanJoin = false,
  customPresetId = 'custom_editable',
  customPresetLabel = 'Custom Editable Preset',
  currentPromptWindowPrompts = [],
  appendingPromptWindow = false,
  appendPromptWindowStatus = '',
  appendPromptWindowError = '',
  onCustomPresetIdInput = () => {},
  onCustomPresetLabelInput = () => {},
  onAddSegment = () => {},
  onResetToPreset = () => {},
  onExport = () => {},
  onAppendPromptWindowToJson = () => {},
  onRemoveSegment = () => {},
  onUpdateSegment = () => {},
  seedPrompts = [],
  playingSeedPromptIndex = null,
  generatingSeedPromptIndex = null,
  promptEvents = [],
  promptConfigEditorOpen = false,
  promptConfigLoading = false,
  promptConfigSaving = false,
  promptConfigStatus = '',
  promptConfigError = '',
  nextSegmentPromptEditorOpen = false,
  autoExtensionPromptEditorOpen = false,
  rewriteWindowPromptEditorOpen = true,
  nextSegmentSystemPromptDraft = '',
  autoExtensionSystemPromptDraft = '',
  rewriteWindowSystemPromptDraft = '',
  onPromptConfigEditorToggle = () => {},
  onReloadPromptConfig = () => {},
  onNextSegmentSystemPromptInput = () => {},
  onAutoExtensionSystemPromptInput = () => {},
  onRewriteWindowSystemPromptInput = () => {},
  onSavePromptConfig = () => {},
}: DevtoolsDrawerProps) {
  const displayPrompts = useMemo(() => {
    if (Array.isArray(seedPrompts) && seedPrompts.length > 0) {
      return seedPrompts;
    }
    return Array.isArray(currentPromptWindowPrompts)
      ? currentPromptWindowPrompts
      : [];
  }, [seedPrompts, currentPromptWindowPrompts]);

  return (
    <section className="space-y-4" aria-label="Advanced controls">
      <div className="rounded-2xl border border-border bg-card p-5 shadow-sm backdrop-blur-sm">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-1">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-200/70">
              Devtools
            </p>
            <h2 className="text-2xl font-semibold text-foreground">
              Advanced controls
            </h2>
          </div>
          <p className="max-w-3xl text-sm leading-6 text-muted-foreground">
            Prompt memory, preset editing, and server-owned system prompts stay
            separate from the main release workflow.
          </p>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {/* Prompt window memory */}
        <details className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm" open>
          <summary className="flex cursor-pointer list-none items-start justify-between gap-4 px-5 py-4">
            <div className="space-y-1">
              <span className="block text-lg font-semibold text-foreground">
                Prompt window memory
              </span>
              <span className="block text-sm leading-6 text-muted-foreground">
                Inspect the prompts currently driving generation.
              </span>
            </div>
            <Badge variant="secondary">{displayPrompts.length} prompts</Badge>
          </summary>
          <div className="border-t border-border px-5 py-4">
            {displayPrompts.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-border bg-secondary px-4 py-6 text-sm text-muted-foreground">
                No prompts are in the active window yet.
              </div>
            ) : (
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {displayPrompts.slice(0, 6).map((prompt, index) => (
                  <div
                    key={index}
                    className={cn(
                      'rounded-2xl border bg-secondary p-4',
                      playingSeedPromptIndex === index
                        ? 'border-sky-400/35 bg-sky-500/10'
                        : generatingSeedPromptIndex === index &&
                            playingSeedPromptIndex !== index
                          ? 'border-amber-400/35 bg-amber-500/10'
                          : 'border-border',
                    )}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                        Prompt {index + 1}
                      </div>
                      {playingSeedPromptIndex === index && (
                        <Badge variant="default">
                          playing
                        </Badge>
                      )}
                      {generatingSeedPromptIndex === index &&
                        playingSeedPromptIndex !== index && (
                          <Badge variant="warning">
                            generating
                          </Badge>
                        )}
                    </div>
                    <div className="mt-3 text-sm leading-6 text-foreground">
                      {prompt}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </details>

        {/* Prompt activity */}
        <details className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
          <summary className="flex cursor-pointer list-none items-start justify-between gap-4 px-5 py-4">
            <div className="space-y-1">
              <span className="block text-lg font-semibold text-foreground">
                Prompt activity
              </span>
              <span className="block text-sm leading-6 text-muted-foreground">
                Review prompt submissions and rewrite events.
              </span>
            </div>
            <Badge variant="secondary">{promptEvents.length} events</Badge>
          </summary>
          <div className="border-t border-border px-5 py-4">
            {promptEvents.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-border bg-secondary px-4 py-6 text-sm text-muted-foreground">
                No prompt activity has been recorded yet.
              </div>
            ) : (
              <ScrollArea className="max-h-80 pr-4">
                <div className="space-y-3">
                {promptEvents.map((item, idx) => (
                  <div
                    key={idx}
                    className="grid gap-2 rounded-2xl border border-border bg-secondary px-4 py-3"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="secondary">{item.status}</Badge>
                      <Badge variant="outline">
                      {item.source}
                      </Badge>
                      {(item.model ||
                        Number.isFinite(item.latencyMs)) && (
                        <span className="text-xs text-muted-foreground">
                          {item.model && item.model}
                          {Number.isFinite(item.latencyMs) && (
                            <>
                              {item.model && ' · '}
                              {Math.round(item.latencyMs)}ms
                            </>
                          )}
                        </span>
                      )}
                    </div>
                    <span className="text-sm leading-6 text-foreground">
                      {item.text}
                    </span>
                  </div>
                ))}
                </div>
              </ScrollArea>
            )}
          </div>
        </details>

        {/* Editable preset builder */}
        {editableMode && (
          <details
            className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm xl:col-span-2"
            open
          >
            <summary className="flex cursor-pointer list-none items-start justify-between gap-4 px-5 py-4">
              <div className="space-y-1">
                <span className="block text-lg font-semibold text-foreground">
                  Editable preset builder
                </span>
                <span className="block text-sm leading-6 text-muted-foreground">
                  Shape curated segments before starting a session or export
                  them to a local overlay.
                </span>
              </div>
              <Badge variant="secondary">{editableSegments.length} segments</Badge>
            </summary>
            <div className="border-t border-border px-5 py-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="custom-preset-id">Export Preset ID</Label>
                  <Input
                  id="custom-preset-id"
                  type="text"
                  value={customPresetId}
                  onChange={onCustomPresetIdInput}
                  disabled={sessionStarted}
                />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="custom-preset-label">Export Label</Label>
                  <Input
                  id="custom-preset-label"
                  type="text"
                  value={customPresetLabel}
                  onChange={onCustomPresetLabelInput}
                  disabled={sessionStarted}
                />
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-3">
                <Button
                  variant="outline"
                  onClick={onAddSegment}
                  disabled={sessionStarted}
                >
                  Add Segment
                </Button>
                <Button
                  variant="outline"
                  onClick={onResetToPreset}
                  disabled={sessionStarted || !selectedPreset}
                >
                  Reset To Preset
                </Button>
                <Button onClick={onExport} disabled={!editableCanJoin}>
                  Export JSON
                </Button>
                <Button
                  variant="secondary"
                  onClick={onAppendPromptWindowToJson}
                  disabled={
                    currentPromptWindowPrompts.length < 2 ||
                    appendingPromptWindow
                  }
                >
                  {appendingPromptWindow
                    ? 'Appending...'
                    : 'Append Window To Presets JSON'}
                </Button>
              </div>

              {appendPromptWindowStatus && (
                <div className="mt-4 rounded-2xl border border-emerald-400/25 bg-emerald-950/40 px-4 py-3 text-sm text-emerald-100">
                  {appendPromptWindowStatus}
                </div>
              )}
              {appendPromptWindowError && (
                <div className="mt-4 rounded-2xl border border-rose-400/25 bg-rose-950/40 px-4 py-3 text-sm text-rose-100">
                  {appendPromptWindowError}
                </div>
              )}

              <div className="mt-4 grid gap-4">
                {editableSegments.map((segment, index) => (
                  <div
                    key={index}
                    className="rounded-2xl border border-border bg-secondary p-4"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <span className="text-sm font-semibold text-foreground">
                        Segment {index + 1}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onRemoveSegment(index)}
                        disabled={
                          sessionStarted || editableSegments.length <= 1
                        }
                      >
                        Remove
                      </Button>
                    </div>

                    <Textarea
                      rows={5}
                      value={segment}
                      onChange={(e) =>
                        onUpdateSegment(index, e.currentTarget.value)
                      }
                      disabled={sessionStarted}
                      className="mt-3 min-h-[148px]"
                    />
                  </div>
                ))}
              </div>
            </div>
          </details>
        )}

        {/* System prompt editor */}
        {editableMode && !demoMode && (
          <details
            className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm xl:col-span-2"
            open={promptConfigEditorOpen}
            onToggle={onPromptConfigEditorToggle}
          >
            <summary className="flex cursor-pointer list-none items-start justify-between gap-4 px-5 py-4">
              <div className="space-y-1">
                <span className="block text-lg font-semibold text-foreground">
                  System prompt editor
                </span>
                <span className="block text-sm leading-6 text-muted-foreground">
                  Edit local overlay files for the server-side system prompts.
                </span>
              </div>
              <Badge variant="secondary">
                {promptConfigSaving
                  ? 'Saving'
                  : promptConfigLoading
                    ? 'Loading'
                    : 'Ready'}
              </Badge>
            </summary>
            <div className="border-t border-border px-5 py-4">
              <div className="space-y-4">
                <details
                  className="overflow-hidden rounded-2xl border border-border bg-secondary"
                  open={nextSegmentPromptEditorOpen}
                >
                  <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-foreground">
                    Next Segment System Prompt (single-segment path)
                  </summary>
                  <div className="border-t border-border px-4 py-4">
                    <Textarea
                      id="next-segment-system-prompt"
                      rows={10}
                      value={nextSegmentSystemPromptDraft}
                      onChange={onNextSegmentSystemPromptInput}
                      disabled={promptConfigLoading || promptConfigSaving}
                      className="min-h-[220px]"
                    />
                  </div>
                </details>

                <details
                  className="overflow-hidden rounded-2xl border border-border bg-secondary"
                  open={autoExtensionPromptEditorOpen}
                >
                  <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-foreground">
                    Auto Extension System Prompt (auto path)
                  </summary>
                  <div className="border-t border-border px-4 py-4">
                    <Textarea
                      id="auto-extension-system-prompt"
                      rows={10}
                      value={autoExtensionSystemPromptDraft}
                      onChange={onAutoExtensionSystemPromptInput}
                      disabled={promptConfigLoading || promptConfigSaving}
                      className="min-h-[220px]"
                    />
                  </div>
                </details>

                <details
                  className="overflow-hidden rounded-2xl border border-border bg-secondary"
                  open={rewriteWindowPromptEditorOpen}
                >
                  <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-foreground">
                    Rewrite Window System Prompt (rewrite-all path)
                  </summary>
                  <div className="border-t border-border px-4 py-4">
                    <Textarea
                      id="rewrite-window-system-prompt"
                      rows={10}
                      value={rewriteWindowSystemPromptDraft}
                      onChange={onRewriteWindowSystemPromptInput}
                      disabled={promptConfigLoading || promptConfigSaving}
                      className="min-h-[220px]"
                    />
                  </div>
                </details>

                <div className="flex flex-wrap gap-3">
                  <Button
                    variant="outline"
                    onClick={onReloadPromptConfig}
                    disabled={promptConfigLoading || promptConfigSaving}
                  >
                    Reload From Disk
                  </Button>
                  <Button
                    onClick={onSavePromptConfig}
                    disabled={promptConfigLoading || promptConfigSaving}
                  >
                    {promptConfigSaving ? 'Saving...' : 'Save To Disk'}
                  </Button>
                </div>

                {promptConfigStatus && (
                  <div className="rounded-2xl border border-emerald-400/25 bg-emerald-950/40 px-4 py-3 text-sm text-emerald-100">
                    {promptConfigStatus}
                  </div>
                )}
                {promptConfigError && (
                  <div className="rounded-2xl border border-rose-400/25 bg-rose-950/40 px-4 py-3 text-sm text-rose-100">
                    {promptConfigError}
                  </div>
                )}
              </div>
            </div>
          </details>
        )}
      </div>
    </section>
  );
}
