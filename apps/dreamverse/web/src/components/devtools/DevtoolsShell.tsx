'use client';

import React from 'react';
import SessionAlerts from '../shell/SessionAlerts';
import TopStatusBar from '../shell/TopStatusBar';
import WorkspaceShell from '../shell/WorkspaceShell';
import Workspace from '../Workspace';
import VideoPlayer from '../VideoPlayer';
import RewriteInspector from '../rewrite/RewriteInspector';
import DevtoolsComposer from './DevtoolsComposer';
import DevtoolsDrawer from './DevtoolsDrawer';

interface DevtoolsShellProps {
  connected?: boolean;
  gpuAssigned?: boolean;
  sessionStarted?: boolean;
  queuePosition?: number;
  connecting?: boolean;

  storyPresets?: Record<string, any>[];
  selectedPresetId?: string;
  enhancementEnabled?: boolean;
  autoExtensionEnabled?: boolean;
  loopGenerationEnabled?: boolean;
  canJoinSession?: boolean;
  canSubmitContinuation?: boolean;
  editableMode?: boolean;
  demoMode?: boolean;
  editableCanJoin?: boolean;
  curatedPromptLimit?: number;
  maxCuratedPromptCount?: number;

  onPresetChange?: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onEnhancementToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onCuratedPromptLimitChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onAutoExtensionToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onLoopToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onJoin?: () => void;
  onLeave?: () => void;

  sessionNotice?: string;
  generationCapReached?: boolean;
  generationSegmentCap?: number;
  onRestartGeneration?: () => void;

  selectedPreset?: Record<string, any> | null;
  editableSegments?: string[];
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

  livePromptDraft?: string;
  livePromptRewriteMode?: boolean;
  rewritingSeedPrompts?: boolean;
  promptEvents?: Record<string, any>[];
  onLivePromptInput?: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onLivePromptModeToggle?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onLivePromptKeydown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmitLivePrompt?: () => void;
  onSpeechTranscript?: (text: string) => void;
  onSpeechInterimChange?: (text: string) => void;

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

  autoExtensionTimeoutHint?: string;

  activeClip?: Record<string, any> | null;
  liveClipLabel?: string;
  liveClipPrompt?: string;
  galleryClips?: Record<string, any>[];
  activeClipId?: string;
  showLivePlayback?: boolean;
  onSelectClip?: (id: string) => void;

  avPlaybackStarted?: boolean;
  mediaAppendError?: string | null;
  timeLeft?: number | null;
  ttffStartAtMs?: number | null;
  ttffValueMs?: number | null;
  timeBetweenVideosMs?: number | null;
  loadingAnimation?: boolean;
  formatTime?: (seconds: number) => string;
  formatDurationMs?: (durationMs: number) => string;
  onPlaying?: () => void;
}

export default function DevtoolsShell({
  connected = false,
  gpuAssigned = false,
  sessionStarted = false,
  queuePosition = 0,
  connecting = false,

  storyPresets = [],
  selectedPresetId = '',
  enhancementEnabled = true,
  autoExtensionEnabled = false,
  loopGenerationEnabled = false,
  canJoinSession = false,
  canSubmitContinuation = false,
  editableMode = false,
  demoMode = false,
  editableCanJoin = false,
  curatedPromptLimit = 0,
  maxCuratedPromptCount = 0,

  onPresetChange = () => {},
  onEnhancementToggle = () => {},
  onCuratedPromptLimitChange = () => {},
  onAutoExtensionToggle = () => {},
  onLoopToggle = () => {},
  onJoin = () => {},
  onLeave = () => {},

  sessionNotice = '',
  selectedPreset = null,
  editableSegments = [],
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

  livePromptDraft = '',
  livePromptRewriteMode = false,
  rewritingSeedPrompts = false,
  promptEvents = [],
  onLivePromptInput = () => {},
  onLivePromptModeToggle = () => {},
  onLivePromptKeydown = () => {},
  onSubmitLivePrompt = () => {},
  onSpeechTranscript,
  onSpeechInterimChange,

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

  autoExtensionTimeoutHint = '',

  activeClip = null,
  showLivePlayback = true,

  avPlaybackStarted = false,
  mediaAppendError = null,
  timeLeft = null,
  ttffStartAtMs = null,
  ttffValueMs = null,
  timeBetweenVideosMs = null,
  loadingAnimation = false,
  formatTime = (seconds) => `${seconds}`,
  formatDurationMs = (durationMs) => `${durationMs}`,
  onPlaying = () => {},
}: DevtoolsShellProps) {
  return (
    <WorkspaceShell
      devtools={true}
      topbar={
        <TopStatusBar
          modeLabel="Devtools Mode"
          title="Preset-driven video continuation"
          connected={connected}
          connecting={connecting}
          sessionStarted={sessionStarted}
          gpuAssigned={gpuAssigned}
          queuePosition={queuePosition}
          livePromptRewriteMode={livePromptRewriteMode}
          rewritingSeedPrompts={rewritingSeedPrompts}
          timeLeft={timeLeft}
          ttffStartAtMs={ttffStartAtMs}
          ttffValueMs={ttffValueMs}
          videoGapMs={timeBetweenVideosMs}
          formatTime={formatTime}
          formatDurationMs={formatDurationMs}
        />
      }
      workspace={
        <div className="flex flex-col gap-4">
          <VideoPlayer
            activeClip={activeClip}
            sessionStarted={sessionStarted}
            avPlaybackStarted={avPlaybackStarted}
            mediaAppendError={mediaAppendError}
            timeLeft={timeLeft}
            gpuAssigned={gpuAssigned}
            connected={connected}
            queuePosition={queuePosition}
            loadingAnimation={loadingAnimation}
            showLivePlayback={showLivePlayback}
            onPlaying={onPlaying}
          />
          <Workspace
            promptEvents={promptEvents}
            sessionStarted={sessionStarted}
          />
        </div>
      }
      sidebar={
        <RewriteInspector
          currentPromptWindowPrompts={currentPromptWindowPrompts}
          promptEvents={promptEvents}
          rewritingSeedPrompts={rewritingSeedPrompts}
          rewriteWindowMode={livePromptRewriteMode}
        />
      }
      composer={
        <DevtoolsComposer
          connected={connected}
          gpuAssigned={gpuAssigned}
          sessionStarted={sessionStarted}
          queuePosition={queuePosition}
          connecting={connecting}
          storyPresets={storyPresets}
          selectedPresetId={selectedPresetId}
          livePromptDraft={livePromptDraft}
          canJoinSession={canJoinSession}
          canSubmitContinuation={canSubmitContinuation}
          editableMode={editableMode}
          editableCanJoin={editableCanJoin}
          demoMode={demoMode}
          enhancementEnabled={enhancementEnabled}
          autoExtensionEnabled={autoExtensionEnabled}
          loopGenerationEnabled={loopGenerationEnabled}
          curatedPromptLimit={curatedPromptLimit}
          maxCuratedPromptCount={maxCuratedPromptCount}
          rewriteWindowMode={livePromptRewriteMode}
          rewritingSeedPrompts={rewritingSeedPrompts}
          autoExtensionTimeoutHint={autoExtensionTimeoutHint}
          onPresetChange={onPresetChange}
          onLivePromptInput={onLivePromptInput}
          onLivePromptKeydown={onLivePromptKeydown}
          onJoin={onJoin}
          onSubmitLivePrompt={onSubmitLivePrompt}
          onLeave={onLeave}
          onEnhancementToggle={onEnhancementToggle}
          onCuratedPromptLimitChange={onCuratedPromptLimitChange}
          onAutoExtensionToggle={onAutoExtensionToggle}
          onLoopToggle={onLoopToggle}
          onLivePromptModeToggle={onLivePromptModeToggle}
          onSpeechTranscript={onSpeechTranscript}
          onSpeechInterimChange={onSpeechInterimChange}
        />
      }
      alerts={<SessionAlerts sessionNotice={sessionNotice} />}
      drawer={
        <DevtoolsDrawer
          editableMode={editableMode}
          demoMode={demoMode}
          sessionStarted={sessionStarted}
          selectedPreset={selectedPreset}
          editableSegments={editableSegments}
          editableCanJoin={editableCanJoin}
          customPresetId={customPresetId}
          customPresetLabel={customPresetLabel}
          currentPromptWindowPrompts={currentPromptWindowPrompts}
          appendingPromptWindow={appendingPromptWindow}
          appendPromptWindowStatus={appendPromptWindowStatus}
          appendPromptWindowError={appendPromptWindowError}
          onCustomPresetIdInput={onCustomPresetIdInput}
          onCustomPresetLabelInput={onCustomPresetLabelInput}
          onAddSegment={onAddSegment}
          onResetToPreset={onResetToPreset}
          onExport={onExport}
          onAppendPromptWindowToJson={onAppendPromptWindowToJson}
          onRemoveSegment={onRemoveSegment}
          onUpdateSegment={onUpdateSegment}
          seedPrompts={seedPrompts}
          playingSeedPromptIndex={playingSeedPromptIndex}
          generatingSeedPromptIndex={generatingSeedPromptIndex}
          promptEvents={promptEvents}
          promptConfigEditorOpen={promptConfigEditorOpen}
          promptConfigLoading={promptConfigLoading}
          promptConfigSaving={promptConfigSaving}
          promptConfigStatus={promptConfigStatus}
          promptConfigError={promptConfigError}
          nextSegmentPromptEditorOpen={nextSegmentPromptEditorOpen}
          autoExtensionPromptEditorOpen={autoExtensionPromptEditorOpen}
          rewriteWindowPromptEditorOpen={rewriteWindowPromptEditorOpen}
          nextSegmentSystemPromptDraft={nextSegmentSystemPromptDraft}
          autoExtensionSystemPromptDraft={autoExtensionSystemPromptDraft}
          rewriteWindowSystemPromptDraft={rewriteWindowSystemPromptDraft}
          onPromptConfigEditorToggle={onPromptConfigEditorToggle}
          onReloadPromptConfig={onReloadPromptConfig}
          onNextSegmentSystemPromptInput={onNextSegmentSystemPromptInput}
          onAutoExtensionSystemPromptInput={onAutoExtensionSystemPromptInput}
          onRewriteWindowSystemPromptInput={onRewriteWindowSystemPromptInput}
          onSavePromptConfig={onSavePromptConfig}
        />
      }
    />
  );
}
