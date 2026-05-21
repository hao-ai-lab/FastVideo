"use client";

import React, { useRef, useState, useCallback, useEffect } from "react";
import Image from "next/image";
import { Film, ArrowUp, X, Loader2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import LeaveSessionModal, { shouldShowLeaveWarning } from "@/components/LeaveSessionModal";
import SpeechToTextButton from "@/components/SpeechToTextButton";
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
				? "What video are you imagining?"
				: "What do you want to edit?";
	const actionLabel = !sessionStarted ? "Generate" : "Rewrite rollout";

	const inputRef = useRef<HTMLTextAreaElement>(null);
	const scrollRef = useRef<HTMLDivElement>(null);
	const [canScrollLeft, setCanScrollLeft] = useState(false);
	const [canScrollRight, setCanScrollRight] = useState(false);
	const [presetRailDragging, setPresetRailDragging] = useState(false);
	const presetDragStateRef = useRef({
		pointerId: null as number | null,
		startX: 0,
		startScrollLeft: 0,
		moved: false,
	});
	const suppressPresetClickRef = useRef(false);

	const updateScrollState = useCallback(() => {
		const el = scrollRef.current;
		if (!el) return;
		setCanScrollLeft(el.scrollLeft > 2);
		setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 2);
	}, []);

	const handlePresetWheel = useCallback(
		(event: React.WheelEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			if (!el) return;
			if (el.scrollWidth <= el.clientWidth + 1) return;

			const dominantDelta = Math.abs(event.deltaX) > Math.abs(event.deltaY)
				? event.deltaX
				: event.deltaY;
			if (!dominantDelta) return;

			const maxScrollLeft = Math.max(el.scrollWidth - el.clientWidth, 0);
			const nextScrollLeft = Math.min(
				Math.max(el.scrollLeft + dominantDelta, 0),
				maxScrollLeft,
			);
			if (nextScrollLeft === el.scrollLeft) return;

			event.preventDefault();
			el.scrollLeft = nextScrollLeft;
			updateScrollState();
		},
		[updateScrollState],
	);

	const finishPresetDrag = useCallback(() => {
		presetDragStateRef.current = {
			pointerId: null,
			startX: 0,
			startScrollLeft: 0,
			moved: false,
		};
		setPresetRailDragging(false);
	}, []);

	const handlePresetPointerDown = useCallback(
		(event: React.PointerEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			if (!el) return;
			if (event.pointerType !== "mouse" || event.button !== 0) return;
			if (el.scrollWidth <= el.clientWidth + 1) return;

			suppressPresetClickRef.current = false;
			presetDragStateRef.current = {
				pointerId: event.pointerId,
				startX: event.clientX,
				startScrollLeft: el.scrollLeft,
				moved: false,
			};
		},
		[],
	);

	const handlePresetPointerMove = useCallback(
		(event: React.PointerEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			const dragState = presetDragStateRef.current;
			if (!el || dragState.pointerId !== event.pointerId) return;

			const deltaX = event.clientX - dragState.startX;
			if (!dragState.moved && Math.abs(deltaX) > 4) {
				dragState.moved = true;
				suppressPresetClickRef.current = true;
				setPresetRailDragging(true);
				el.setPointerCapture?.(event.pointerId);
			}
			if (!dragState.moved) return;

			event.preventDefault();
			const maxScrollLeft = Math.max(el.scrollWidth - el.clientWidth, 0);
			el.scrollLeft = Math.min(
				Math.max(dragState.startScrollLeft - deltaX, 0),
				maxScrollLeft,
			);
			updateScrollState();
		},
		[updateScrollState],
	);

	const handlePresetPointerUp = useCallback(
		(event: React.PointerEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			if (!el || presetDragStateRef.current.pointerId !== event.pointerId) return;
			if (el.hasPointerCapture?.(event.pointerId)) {
				el.releasePointerCapture(event.pointerId);
			}
			finishPresetDrag();
		},
		[finishPresetDrag],
	);

	const handlePresetClickCapture = useCallback(
		(event: React.MouseEvent<HTMLDivElement>) => {
			if (!suppressPresetClickRef.current) return;
			suppressPresetClickRef.current = false;
			event.preventDefault();
			event.stopPropagation();
		},
		[],
	);

	useEffect(() => {
		updateScrollState();
	}, [storyPresets, updateScrollState]);

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
						<p className="max-w-md text-xs text-muted-foreground">Project sessions are currently limited to 5 minutes. Start a new project to create more videos.</p>
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
						<p className="max-w-xs text-xs text-muted-foreground">Each project currently has a 5-minute session. Start a new project to continue creating videos.</p>
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
			{storyPresets.length > 0 && !sessionStarted && (
				<div className={cn("relative transition-opacity duration-200", isGenerating && "pointer-events-none opacity-40")}>
					<div
						ref={scrollRef}
						onScroll={updateScrollState}
						onWheel={handlePresetWheel}
						onPointerDown={handlePresetPointerDown}
						onPointerMove={handlePresetPointerMove}
						onPointerUp={handlePresetPointerUp}
						onPointerCancel={handlePresetPointerUp}
						onLostPointerCapture={finishPresetDrag}
						onClickCapture={handlePresetClickCapture}
						className={cn(
							"scrollbar-hidden flex gap-3 overflow-x-auto px-1 select-none",
							presetRailDragging ? "cursor-grabbing" : "cursor-grab",
						)}
					>
						{storyPresets.map((preset) => (
							<button
								key={preset.id}
								type="button"
								disabled={isGenerating}
								onClick={() => onPresetGenerate(preset.id)}
								className="flex flex-col sm:flex-row items-start gap-1.5 shrink-0 rounded-xl border p-2.5 text-left backdrop-blur-sm transition-colors max-w-42 sm:max-w-[215px] border-input bg-card/80 text-muted-foreground hover:bg-slate-200/60 hover:border-slate-400 hover:text-slate-700 dark:bg-slate-800/80 dark:text-slate-300 dark:hover:bg-slate-700/50 dark:hover:border-slate-500 dark:hover:text-slate-200"
							>
								<Film className="mt-0.5 size-4 shrink-0 opacity-60" />
								<span className="flex flex-col gap-1 min-w-0">
									<span className="text-[14px] font-medium line-clamp-1">{preset.label}</span>
									{preset.description && <span className="text-xs leading-tight opacity-70 line-clamp-3 sm:line-clamp-2">{preset.description}</span>}
								</span>
							</button>
						))}
					</div>

					<div
						className={cn("pointer-events-none absolute inset-y-0 left-0 w-8 bg-background transition-opacity duration-150", canScrollLeft ? "opacity-100" : "opacity-0")}
						style={{ maskImage: "linear-gradient(to right, black, transparent)", WebkitMaskImage: "linear-gradient(to right, black, transparent)" }}
						aria-hidden="true"
					/>
					<div
						className={cn("pointer-events-none absolute inset-y-0 right-0 w-8 bg-background transition-opacity duration-150", canScrollRight ? "opacity-100" : "opacity-0")}
						style={{ maskImage: "linear-gradient(to left, black, transparent)", WebkitMaskImage: "linear-gradient(to left, black, transparent)" }}
						aria-hidden="true"
					/>
				</div>
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
					Starting a new project after the current shot finishes. Your GPU session stays active.
				</div>
			)}

			<div
				className={cn(
					"flex min-w-0 items-center gap-1.5 rounded-4xl border py-2.5 pl-5 pr-2.5 shadow-md backdrop-blur-sm transition-all duration-200",
					isBusy ? "border-input/60 bg-card/40" : "border-input bg-card/65",
				)}
			>
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
