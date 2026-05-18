'use client';

import React, { useMemo, useState } from 'react';

import SpeechToTextButton from '@/components/SpeechToTextButton';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';

interface DevtoolsComposerProps {
  connected?: boolean;
  gpuAssigned?: boolean;
  sessionStarted?: boolean;
  queuePosition?: number;
  connecting?: boolean;
  storyPresets?: Array<Record<string, any>>;
  selectedPresetId?: string;
  livePromptDraft?: string;
  canJoinSession?: boolean;
  canSubmitContinuation?: boolean;
  editableMode?: boolean;
  editableCanJoin?: boolean;
  demoMode?: boolean;
  enhancementEnabled?: boolean;
  autoExtensionEnabled?: boolean;
  loopGenerationEnabled?: boolean;
  curatedPromptLimit?: number;
  maxCuratedPromptCount?: number;
  rewriteWindowMode?: boolean;
  rewritingSeedPrompts?: boolean;
  autoExtensionTimeoutHint?: string;
  onPresetChange?: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onLivePromptInput?: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onLivePromptKeydown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onJoin?: () => void;
  onSubmitLivePrompt?: () => void;
  onLeave?: () => void;
  onEnhancementToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onCuratedPromptLimitChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onAutoExtensionToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onLoopToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onLivePromptModeToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSpeechTranscript?: (text: string) => void;
  onSpeechInterimChange?: (text: string) => void;
}

export default function DevtoolsComposer({
  connected = false,
  gpuAssigned = false,
  sessionStarted = false,
  queuePosition = 0,
  connecting = false,
  storyPresets = [],
  selectedPresetId = '',
  livePromptDraft = '',
  canJoinSession = false,
  canSubmitContinuation = false,
  editableMode = false,
  editableCanJoin = false,
  demoMode = false,
  enhancementEnabled = true,
  autoExtensionEnabled = false,
  loopGenerationEnabled = false,
  curatedPromptLimit = 0,
  maxCuratedPromptCount = 0,
  rewriteWindowMode = false,
  rewritingSeedPrompts = false,
  autoExtensionTimeoutHint = '',
  onPresetChange = () => {},
  onLivePromptInput = () => {},
  onLivePromptKeydown = () => {},
  onJoin = () => {},
  onSubmitLivePrompt = () => {},
  onLeave = () => {},
  onEnhancementToggle = () => {},
  onCuratedPromptLimitChange = () => {},
  onAutoExtensionToggle = () => {},
  onLoopToggle = () => {},
  onLivePromptModeToggle = () => {},
  onSpeechTranscript,
  onSpeechInterimChange,
}: DevtoolsComposerProps) {
  const [sttBusy, setSttBusy] = useState(false);
  const submitButtonLabel = useMemo(
    () =>
      rewriteWindowMode
        ? rewritingSeedPrompts
          ? 'Rewriting...'
          : 'Rewrite Window'
        : 'Extend',
    [rewriteWindowMode, rewritingSeedPrompts],
  );

  const submitDisabled = useMemo(
    () =>
      !canSubmitContinuation ||
      (rewriteWindowMode && rewritingSeedPrompts),
    [canSubmitContinuation, rewriteWindowMode, rewritingSeedPrompts],
  );

  return (
    <section>
      <Card>
        <CardContent className="space-y-5 p-5">
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
            <div className="space-y-4">
              <div className="grid gap-4 lg:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
                <div className="space-y-2">
                  <Label htmlFor="devtools-story-preset">Preset</Label>
                  <Select
                    value={selectedPresetId}
                    onValueChange={(value) =>
                      onPresetChange({
                        currentTarget: { value },
                      } as React.ChangeEvent<HTMLSelectElement>)
                    }
                    disabled={sessionStarted || storyPresets.length === 0}
                  >
                    <SelectTrigger
                      id="devtools-story-preset"
                      aria-label="Story preset"
                    >
                      <SelectValue placeholder="No presets loaded" />
                    </SelectTrigger>
                    <SelectContent>
                      {storyPresets.map((preset) => (
                        <SelectItem key={preset.id} value={preset.id}>
                          {preset.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <Label htmlFor="devtools-continuation-prompt">
                      What should happen next?
                    </Label>
                    {!demoMode && (
                      <div className="flex items-center gap-2">
                        <Checkbox
                          id="devtools-rewrite-window"
                          checked={rewriteWindowMode}
                          onCheckedChange={(checked) =>
                            onLivePromptModeToggle({
                              target: { checked: Boolean(checked) },
                              currentTarget: { checked: Boolean(checked) },
                            } as React.ChangeEvent<HTMLInputElement>)
                          }
                        />
                        <Label htmlFor="devtools-rewrite-window">
                          Rewrite prompt window
                        </Label>
                      </div>
                    )}
                  </div>

                  <div className="flex items-start gap-2">
                    <Textarea
                      id="devtools-continuation-prompt"
                      aria-label="Continuation prompt"
                      rows={3}
                      maxLength={500}
                      value={livePromptDraft}
                      onChange={onLivePromptInput}
                      onKeyDown={onLivePromptKeydown}
                      placeholder={
                        sttBusy
                          ? 'Listening\u2026'
                          : sessionStarted
                            ? rewriteWindowMode
                              ? 'Describe how to rewrite all prompt-window segments'
                              : 'Describe the next beat in the story'
                            : 'Generate from a preset first, then continue the story here'
                      }
                      disabled={!sessionStarted || sttBusy}
                      className="min-h-[132px] min-w-0 flex-1 resize-none"
                    />
                    {onSpeechTranscript && (
                      <SpeechToTextButton
                        disabled={!sessionStarted}
                        onTranscript={onSpeechTranscript}
                        onInterimChange={onSpeechInterimChange}
                        onBusyChange={setSttBusy}
                      />
                    )}
                  </div>

                  <p className="text-sm text-muted-foreground">
                    {sessionStarted
                      ? 'Ctrl/Cmd + Enter submits the current prompt.'
                      : 'Start a session to submit prompts.'}
                  </p>
                </div>
              </div>

              <div
                className="flex flex-wrap gap-2"
                aria-label="Generation settings"
              >
                <Badge variant="secondary">LTX-2 Fast</Badge>
                <Badge variant="outline">
                  {enhancementEnabled
                    ? 'Prompt enhance on'
                    : 'Prompt enhance off'}
                </Badge>
                <Badge variant="outline">
                  {autoExtensionEnabled
                    ? 'Auto extension on'
                    : 'Auto extension off'}
                </Badge>
                <Badge variant="outline">
                  {loopGenerationEnabled ? 'Loop on' : 'Loop off'}
                </Badge>
                {maxCuratedPromptCount > 0 && (
                  <Badge variant="outline">
                    Prompt count {curatedPromptLimit}/{maxCuratedPromptCount}
                  </Badge>
                )}
                {rewriteWindowMode && (
                  <Badge variant="default">Rewrite window mode</Badge>
                )}
                {sessionStarted && queuePosition > 0 ? (
                  <Badge variant="warning">Queue {queuePosition}</Badge>
                ) : sessionStarted && connecting ? (
                  <Badge variant="warning">Connecting</Badge>
                ) : connected && !gpuAssigned && queuePosition === 0 ? (
                  <Badge variant="secondary">Loading model</Badge>
                ) : null}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                {!sessionStarted ? (
                  <Button onClick={onJoin} disabled={!canJoinSession}>
                    Generate
                  </Button>
                ) : (
                  <>
                    <Button onClick={onSubmitLivePrompt} disabled={submitDisabled}>
                      {submitButtonLabel}
                    </Button>
                    <Button variant="outline" onClick={onLeave} disabled={rewritingSeedPrompts}>
                      Leave
                    </Button>
                  </>
                )}
              </div>

              {sessionStarted && autoExtensionTimeoutHint && (
                <div className="rounded-2xl border border-amber-400/25 bg-amber-950/40 px-4 py-3 text-sm text-amber-100">
                  {autoExtensionTimeoutHint}
                </div>
              )}

              {editableMode && !sessionStarted && !editableCanJoin && (
                <div className="rounded-2xl border border-rose-400/25 bg-rose-950/40 px-4 py-3 text-sm text-rose-100">
                  Add at least 2 non-empty segments to join.
                </div>
              )}
            </div>

            <div
              className="space-y-4 rounded-2xl border border-border bg-secondary p-4"
              aria-label="Session options"
            >
              <div className="space-y-1">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-200/70">
                  Session options
                </p>
                <h3 className="text-lg font-semibold text-foreground">
                  Runtime controls
                </h3>
              </div>

              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <Checkbox
                    id="devtools-enhance-prompts"
                    checked={enhancementEnabled}
                    onCheckedChange={(checked) =>
                      onEnhancementToggle({
                        target: { checked: Boolean(checked) },
                        currentTarget: { checked: Boolean(checked) },
                      } as React.ChangeEvent<HTMLInputElement>)
                    }
                    disabled={sessionStarted}
                  />
                  <div className="space-y-1">
                    <Label htmlFor="devtools-enhance-prompts">
                      Enhance prompts
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Uses the planner before the session starts.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <Checkbox
                    id="devtools-auto-extension"
                    checked={autoExtensionEnabled}
                    onCheckedChange={(checked) =>
                      onAutoExtensionToggle({
                        target: { checked: Boolean(checked) },
                        currentTarget: { checked: Boolean(checked) },
                      } as React.ChangeEvent<HTMLInputElement>)
                    }
                  />
                  <div className="space-y-1">
                    <Label htmlFor="devtools-auto-extension">
                      Auto extension
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Extends the rollout automatically between segments.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <Checkbox
                    id="devtools-loop-generation"
                    checked={loopGenerationEnabled}
                    onCheckedChange={(checked) =>
                      onLoopToggle({
                        target: { checked: Boolean(checked) },
                        currentTarget: { checked: Boolean(checked) },
                      } as React.ChangeEvent<HTMLInputElement>)
                    }
                  />
                  <div className="space-y-1">
                    <Label htmlFor="devtools-loop-generation">
                      Loop generation
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Keeps cycling through the current rollout window.
                    </p>
                  </div>
                </div>

                {maxCuratedPromptCount > 0 && (
                  <div className="space-y-2">
                    <Label htmlFor="devtools-prompt-count">Prompt count</Label>
                    <div className="flex items-center gap-3">
                      <Input
                        id="devtools-prompt-count"
                        aria-label="Prompt count"
                        type="number"
                        min={1}
                        max={maxCuratedPromptCount}
                        value={curatedPromptLimit}
                        onChange={onCuratedPromptLimitChange}
                        disabled={sessionStarted}
                        className="w-24"
                      />
                      <small className="text-sm text-muted-foreground">
                        of {maxCuratedPromptCount}
                      </small>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
