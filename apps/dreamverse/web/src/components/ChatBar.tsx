"use client";

import React, { useRef, useState, useCallback, useEffect } from "react";
import Image from "next/image";
import { ArrowUp, X, Loader2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import LeaveSessionModal, { shouldShowLeaveWarning } from "@/components/LeaveSessionModal";
import PresetQuickLaunchRail from "@/components/creation/PresetQuickLaunchRail";
import SessionCreationConfigPills, { type SessionCreationConfig } from "@/components/creation/SessionCreationConfigPills";
import SpeechToTextButton from "@/components/SpeechToTextButton";
import type { AspectRatioId, CreationModeId, CreationModelId, ResolutionId } from "@/lib/creationConfig";
import { cn } from "@/lib/utils";

const PROMPT_MAX_LENGTH = 500;

interface Props {
	sessionStarted?: boolean;
	rewritingSeedPrompts?: boolean;
	isGenerating?: boolean;
	storyPresets?: any[];
	continuationDraft?: string;
	canJoinSession?: boolean;
	canSubmitContinuation?: boolean;
	sessionExpired?: boolean;
	sessionNotice?: string;
	projectResetPending?: boolean;
	viewingReadOnly?: boolean;
	onPresetGenerate?: (presetId: string) => void;
	onContinuationInput?: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
	onContinuationKeydown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
	onGenerate?: () => void;
	onSubmitContinuation?: () => void;
	onLeave?: () => void;
	onStartNewProject?: () => void;
	onBackFromViewing?: () => void;
	onSpeechTranscript?: (text: string) => void;
	onSpeechInterimChange?: (text: string) => void;
	sessionCreationConfig?: SessionCreationConfig | null;
	configPillsReadOnly?: boolean;
	onSessionModelChange?: (modelId: CreationModelId) => void;
	onSessionModeChange?: (modeId: CreationModeId) => void;
	onSessionAspectRatioChange?: (aspectRatio: AspectRatioId) => void;
	onSessionResolutionChange?: (resolution: ResolutionId) => void;
	onSessionDurationChange?: (durationSec: number) => void;
}

export default function ChatBar({
	sessionStarted = false,
	rewritingSeedPrompts = false,
	isGenerating = false,
	storyPresets = [],
	continuationDraft = "",
	canJoinSession = false,
	canSubmitContinuation = false,
	sessionExpired = false,
	sessionNotice = "",
	projectResetPending = false,
	viewingReadOnly = false,
	onPresetGenerate = () => {},
	onContinuationInput = () => {},
	onContinuationKeydown = () => {},
	onGenerate = () => {},
	onSubmitContinuation = () => {},
	onLeave = () => {},
	onStartNewProject = () => {},
	onBackFromViewing = () => {},
	onSpeechTranscript,
	onSpeechInterimChange,
	sessionCreationConfig = null,
	configPillsReadOnly = false,
	onSessionModelChange,
	onSessionModeChange,
	onSessionAspectRatioChange,
	onSessionResolutionChange,
	onSessionDurationChange,
}: Props) {
	const [sttBusy, setSttBusy] = useState(false);
	const [leaveModalOpen, setLeaveModalOpen] = useState(false);
	const showSpinner = isGenerating || rewritingSeedPrompts;
	const isBusy = isGenerating || rewritingSeedPrompts || projectResetPending;
	const messagePlaceholder = projectResetPending
		? "Starting new project\u2026"
		: isBusy
			? "Generating video\u2026"
			: !sessionStarted
				? "Describe your video"
				: "What do you want to edit?";
	const actionLabel = !sessionStarted ? "Generate" : "Rewrite rollout";

	const inputRef = useRef<HTMLTextAreaElement>(null);

	useEffect(() => {
		if (!isBusy && !sttBusy && !window.matchMedia("(pointer: coarse)").matches) {
			inputRef.current?.focus();
		}
	}, [isBusy, sttBusy, sessionStarted]);

	const autoResize = useCallback(() => {
		const el = inputRef.current;
		if (!el) return;
		el.style.height = "auto";
		const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 20;
		const maxHeight = lineHeight * 3;
		el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
		el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
	}, []);

	useEffect(() => {
		autoResize();
	}, [continuationDraft, autoResize]);

	const handleKeyDown = useCallback(
		(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
			if (e.key === "Enter" && !e.nativeEvent.isComposing && !e.shiftKey) {
				e.preventDefault();
				if (!sessionStarted) {
					if (canJoinSession && !isGenerating && continuationDraft.trim()) {
						onGenerate();
					}
				} else {
					onContinuationKeydown(e);
				}
				return;
			}
			onContinuationKeydown(e);
		},
		[onContinuationKeydown, sessionStarted, canJoinSession, isGenerating, continuationDraft, onGenerate],
	);

	if (viewingReadOnly) {
		return (
			<section className="mx-auto flex w-full max-w-2xl shrink-0 flex-col gap-4">
				<div className="flex flex-col items-center gap-3 rounded-2xl border border-border bg-card/80 px-6 py-4 text-center shadow-md backdrop-blur-sm">
					<div className="flex flex-col gap-1">
						<p className="text-sm font-semibold text-foreground">View-only project</p>
						<p className="max-w-md text-xs text-muted-foreground">Sessions are limited to 5 minutes. Start a new project to keep creating.</p>
					</div>
					<div className="mt-1 flex items-center gap-2">
						<Button onClick={onBackFromViewing} variant="outline" size="sm" className="gap-1.5 rounded-full px-4">
							<ArrowLeft className="size-3.5" />
							Back
						</Button>
						<Button onClick={onStartNewProject} size="sm" className="rounded-full px-5">
							New project
						</Button>
					</div>
				</div>
			</section>
		);
	}

	if (sessionExpired) {
		return (
			<section className="mx-auto flex w-full max-w-2xl shrink-0 flex-col gap-4">
				<div className="flex flex-col items-center gap-3 rounded-2xl border border-border bg-card/80 px-8 py-5 text-center shadow-md backdrop-blur-sm">
					<div className="flex flex-col gap-1">
						<p className="text-sm font-semibold text-foreground">Session ended</p>
						<p className="max-w-xs text-xs text-muted-foreground">Sessions are limited to 5 minutes. Start a new project to continue.</p>
					</div>
					<div className="mt-1 flex items-center gap-2">
						<Button onClick={onStartNewProject} size="sm" className="rounded-full px-5">
							New Project
						</Button>
						<a href="https://docs.google.com/forms/d/e/1FAIpQLSe5zpO1iD8Ds-Ih-fOLm64qd7YZVvuvAyHuJaAfw1hkRHTe_A/viewform?usp=publish-editor" target="_blank" rel="noopener noreferrer">
							<Button variant="outline" size="sm" className="rounded-full px-5">
								Join Waitlist
							</Button>
						</a>
					</div>
				</div>
			</section>
		);
	}

	return (
		<section className="mx-auto flex w-full max-w-2xl shrink-0 flex-col gap-4">
			{!sessionStarted && (
				<PresetQuickLaunchRail storyPresets={storyPresets} disabled={isGenerating} onPresetGenerate={onPresetGenerate} />
			)}

			{sessionNotice && (
				<div
					className={cn(
						"rounded-xl px-4 py-2.5 text-center text-xs",
						sessionStarted
							? "border border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-400"
							: "border border-rose-500/20 bg-rose-500/10 text-rose-700 dark:text-rose-300",
					)}
				>
					{sessionNotice}
				</div>
			)}

			{projectResetPending && sessionStarted && (
				<div className="rounded-xl border border-sky-500/20 bg-sky-500/10 px-4 py-2.5 text-center text-xs text-sky-700 dark:text-sky-300">
					Starting a new project when this shot finishes. GPU session stays open.
				</div>
			)}

			<div
				className={cn(
					"flex min-w-0 flex-col gap-2 rounded-4xl border py-2.5 pl-5 pr-2.5 shadow-md backdrop-blur-sm transition-all duration-200",
					isBusy ? "border-input/60 bg-card/40" : "border-input bg-card/65",
				)}
			>
				{sessionStarted && sessionCreationConfig && (
					<SessionCreationConfigPills
						{...sessionCreationConfig}
						disabled={isBusy}
						readOnly={configPillsReadOnly}
						onModelChange={onSessionModelChange}
						onModeChange={onSessionModeChange}
						onAspectRatioChange={onSessionAspectRatioChange}
						onResolutionChange={onSessionResolutionChange}
						onDurationChange={onSessionDurationChange}
					/>
				)}
				<div className="flex min-w-0 items-center gap-1.5">
					<textarea
					ref={inputRef}
					id="continuation-prompt"
					aria-label="Continuation prompt"
					value={continuationDraft}
					onChange={onContinuationInput}
					onKeyDown={handleKeyDown}
					placeholder={sttBusy ? "Listening\u2026" : messagePlaceholder}
					maxLength={PROMPT_MAX_LENGTH}
					disabled={isBusy || sttBusy}
					rows={1}
					className={cn(
						"min-w-0 flex-1 resize-none bg-transparent text-foreground outline-none placeholder:text-muted-foreground transition-opacity duration-200 scrollbar-thin leading-snug",
						(isBusy || sttBusy) && "cursor-not-allowed opacity-50",
					)}
					/>
					{onSpeechTranscript && <SpeechToTextButton disabled={isBusy} onTranscript={onSpeechTranscript} onInterimChange={onSpeechInterimChange} onBusyChange={setSttBusy} />}
					{!sessionStarted ? (
						<Button
							aria-label={actionLabel}
							title={actionLabel}
							onClick={onGenerate}
							disabled={!canJoinSession || isGenerating || !continuationDraft.trim()}
							size="icon-sm"
							className="shrink-0 rounded-full"
						>
							{showSpinner ? <Loader2 className="size-5 animate-spin" /> : <ArrowUp className="size-5" />}
						</Button>
					) : (
						<>
							<Button
								aria-label={actionLabel}
								title={actionLabel}
								onClick={onSubmitContinuation}
								disabled={!canSubmitContinuation || showSpinner || projectResetPending || !continuationDraft.trim()}
								size="icon-sm"
								className="shrink-0 rounded-full"
							>
								{showSpinner ? <Loader2 className="size-5 animate-spin" /> : <ArrowUp className="size-5" />}
							</Button>
							<Button variant="outline" aria-label="Leave" title="Leave" onClick={() => { if (shouldShowLeaveWarning()) setLeaveModalOpen(true); else onLeave(); }} disabled={isGenerating || projectResetPending} size="icon-sm" className="shrink-0 rounded-full">
								<X className="size-5" />
							</Button>
						</>
					)}
				</div>
			</div>
			<p className="px-2 text-center text-[11px] text-muted-foreground">
				LLM powered by{" "}
				<a
					href="https://ifm.ai/k2/"
					target="_blank"
					rel="noopener noreferrer"
					className="inline-flex items-center gap-1 font-medium text-foreground/80 transition-colors hover:text-foreground"
				>
					<span>K2-V2</span>
					<Image
						src="/k2.png"
						alt=""
						aria-hidden="true"
						width={14}
						height={14}
						className="h-3.5 w-auto opacity-80"
					/>
				</a>
			</p>
			<LeaveSessionModal
				open={leaveModalOpen}
				onClose={() => setLeaveModalOpen(false)}
				onConfirmLeave={() => { setLeaveModalOpen(false); onLeave(); }}
			/>
		</section>
	);
}
