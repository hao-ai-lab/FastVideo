"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { cn } from "@/lib/utils";
import { PlayFilledAlt } from "@carbon/icons-react";
import { Check, ChevronDown, Download, Loader2, Share } from "lucide-react";
import { Button } from "@/components/ui/button";
interface VideoPlayerProps {
	videoRef?: React.RefCallback<HTMLVideoElement>;
	archivedPlaybackRef?: React.RefCallback<HTMLVideoElement>;
	activeClip?: Record<string, any> | null;
	canDownload?: boolean;
	sessionStarted?: boolean;
	avPlaybackStarted?: boolean;
	mediaAppendError?: string | null;
	timeLeft?: number | null;
	gpuAssigned?: boolean;
	connected?: boolean;
	queuePosition?: number;
	loadingAnimation?: boolean;
	showLivePlayback?: boolean;
	defaultMuted?: boolean;
	rewritePending?: boolean;
	waitingForSegmentPrompt?: boolean;
	onPlaying?: () => void;
	onDownload?: () => void;
}

function LoadingSpinner({ className }: { className?: string }) {
	return (
		<div className={cn("relative size-10", className)}>
			<div className="absolute -inset-3 animate-pulse rounded-full bg-white/10 blur-lg" />
			<div className="absolute inset-0 rounded-full border-[2.5px] border-white/8" />
			<div className="absolute inset-0 animate-spin rounded-full border-[2.5px] border-transparent border-t-white/80" />
		</div>
	);
}

function generatingLabel(connected: boolean, gpuAssigned: boolean) {
	if (!connected) return "Connecting\u2026";
	if (!gpuAssigned) return "Waiting for GPU\u2026";
	return "Generating video\u2026";
}

export default function VideoPlayer({
	videoRef,
	archivedPlaybackRef,
	activeClip = null,
	canDownload = false,
	sessionStarted = false,
	avPlaybackStarted = false,
	mediaAppendError = null,
	timeLeft = null,
	gpuAssigned = false,
	connected = false,
	queuePosition = 0,
	loadingAnimation = false,
	showLivePlayback = true,
	defaultMuted = true,
	rewritePending = false,
	waitingForSegmentPrompt = false,
	onPlaying = () => {},
	onDownload,
}: VideoPlayerProps) {
	const inQueue = sessionStarted && connected && queuePosition > 0 && !gpuAssigned;
	const showArchived = !showLivePlayback && !!activeClip;

	const liveVideoEl = useRef<HTMLVideoElement | null>(null);
	const liveWasUnmuted = useRef(false);

	const combinedLiveRef = useCallback((el: HTMLVideoElement | null) => {
		liveVideoEl.current = el;
		videoRef?.(el);
	}, [videoRef]);

	useEffect(() => {
		const el = liveVideoEl.current;
		if (!el) return;
		if (showArchived) {
			liveWasUnmuted.current = !el.muted;
			el.muted = true;
		} else if (liveWasUnmuted.current) {
			el.muted = false;
			liveWasUnmuted.current = false;
		}
	}, [showArchived]);

	const [canShare, setCanShare] = useState(false);
	useEffect(() => {
		setCanShare(typeof navigator.canShare === "function" && window.matchMedia("(pointer: coarse)").matches);
	}, []);

	// Steering mode: the backend may signal "waiting for next prompt" while the current
	// segment is still PLAYING (it generates ahead). Only surface the "Segment complete"
	// overlay once the playhead actually reaches the end of the buffered segment.
	const [playbackReachedEnd, setPlaybackReachedEnd] = useState(false);
	useEffect(() => {
		if (!waitingForSegmentPrompt) {
			setPlaybackReachedEnd(false);
			return;
		}
		const el = liveVideoEl.current;
		if (!el) return;
		const check = () => {
			try {
				const buffered = el.buffered;
				if (buffered.length === 0) return;
				const end = buffered.end(buffered.length - 1);
				// Track proximity both ways: scrubbing back off the end hides the overlay,
				// playing forward to the end re-shows it.
				setPlaybackReachedEnd(el.ended || end - el.currentTime <= 0.2);
			} catch {
				/* buffered access can throw mid-append */
			}
		};
		check();
		el.addEventListener("timeupdate", check);
		el.addEventListener("ended", check);
		el.addEventListener("waiting", check);
		el.addEventListener("stalled", check);
		el.addEventListener("pause", check);
		el.addEventListener("seeking", check);
		el.addEventListener("seeked", check);
		el.addEventListener("playing", check);
		return () => {
			el.removeEventListener("timeupdate", check);
			el.removeEventListener("ended", check);
			el.removeEventListener("waiting", check);
			el.removeEventListener("stalled", check);
			el.removeEventListener("pause", check);
			el.removeEventListener("seeking", check);
			el.removeEventListener("seeked", check);
			el.removeEventListener("playing", check);
		};
	}, [waitingForSegmentPrompt]);

	// Steering mode: when the user submits the next scene, "waiting" flips off and the
	// segment is generated (a few seconds of latency) before frames stream. Show a
	// "Generating next scene…" indicator across that gap so the frozen frame isn't silent.
	const [generatingNext, setGeneratingNext] = useState(false);
	const prevWaitingRef = useRef(false);
	const freezeTimeRef = useRef(0);
	useEffect(() => {
		const wasWaiting = prevWaitingRef.current;
		prevWaitingRef.current = waitingForSegmentPrompt;
		if (waitingForSegmentPrompt) {
			// Back to waiting (next segment finished and is awaiting another prompt).
			setGeneratingNext(false);
			return;
		}
		if (wasWaiting && sessionStarted) {
			freezeTimeRef.current = liveVideoEl.current?.currentTime ?? 0;
			setGeneratingNext(true);
		}
	}, [waitingForSegmentPrompt, sessionStarted]);
	useEffect(() => {
		if (!generatingNext) return;
		if (!sessionStarted) {
			setGeneratingNext(false);
			return;
		}
		const el = liveVideoEl.current;
		if (!el) return;
		const clear = () => setGeneratingNext(false);
		const onTime = () => {
			// New segment frames are now playing past the frozen boundary.
			if (el.currentTime > freezeTimeRef.current + 0.1) setGeneratingNext(false);
		};
		el.addEventListener("playing", clear);
		el.addEventListener("timeupdate", onTime);
		return () => {
			el.removeEventListener("playing", clear);
			el.removeEventListener("timeupdate", onTime);
		};
	}, [generatingNext, sessionStarted]);

	// Drive a ~4.5s progress bar during generation so the wait has a visible ETA.
	const GEN_DURATION_MS = 4500;
	const [genProgress, setGenProgress] = useState(0);
	useEffect(() => {
		if (!generatingNext) {
			setGenProgress(0);
			return;
		}
		const start = performance.now();
		setGenProgress(0);
		const id = setInterval(() => {
			setGenProgress(Math.min((performance.now() - start) / GEN_DURATION_MS, 1));
		}, 50);
		return () => clearInterval(id);
	}, [generatingNext]);

	return (
		<div className="mx-auto w-full max-w-3xl mb-2 sm:mb-6">
			<div className="rounded-2xl border border-border bg-card/50 p-2 shadow-lg backdrop-blur-md">
				<div className="relative aspect-video w-full overflow-hidden rounded-xl border border-border bg-black shadow-lg">
					{/* Archived playback video — hidden when not viewing a clip */}
					<video ref={archivedPlaybackRef} className={cn("h-full w-full bg-slate-900/80 object-cover", !showArchived && "hidden")} onPlaying={onPlaying} playsInline muted={defaultMuted} preload="auto" controls />

					{/* Live video + overlays — always mounted to preserve MSE MediaSource attachment */}
					<div className={cn(showArchived ? "hidden" : "contents")}>
						<video ref={combinedLiveRef} className="h-full w-full bg-slate-900/80 object-cover" onPlaying={onPlaying} playsInline muted={defaultMuted} controls />

						{!sessionStarted && !avPlaybackStarted ? (
							<div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-900/50 p-4 text-center">
								<PlayFilledAlt className="size-10 text-white/25" />
								<p className="text-sm text-white/50">Your video will appear here</p>
							</div>
						) : !avPlaybackStarted && !mediaAppendError && !inQueue && !waitingForSegmentPrompt && loadingAnimation ? (
							<div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-slate-900/60 p-4 backdrop-blur-[2px]">
								<div className="pointer-events-none absolute inset-0 overflow-hidden">
									<div className="absolute inset-0 -translate-x-full animate-[shimmer_3s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-white/[0.04] to-transparent" />
								</div>
								<LoadingSpinner />
								<p className="relative text-sm font-medium text-white/90">{generatingLabel(connected, gpuAssigned)}</p>
							</div>
						) : null}

						{/* Steering mode: this segment finished — wait gracefully for the user's next scene
							instead of spinning. The last frame stays visible behind a soft bottom gradient. */}
						{sessionStarted && waitingForSegmentPrompt && playbackReachedEnd && !mediaAppendError && !inQueue && (
							<div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-end gap-2 bg-gradient-to-t from-slate-950/85 via-slate-950/15 to-transparent p-5 pb-6 text-center">
								<div className="flex size-9 items-center justify-center rounded-full border border-white/25 bg-white/10 shadow-lg backdrop-blur-md">
									<Check className="size-4 text-white/90" />
								</div>
								<div className="space-y-0.5">
									<p className="text-sm font-medium text-white/95">Segment complete</p>
									<p className="text-xs text-white/65">Describe the next scene below to keep going</p>
								</div>
								<ChevronDown className="size-4 animate-bounce text-white/45" />
							</div>
						)}

						{/* Steering mode: generating the next segment — show a ~4.5s progress bar so the wait has an ETA. */}
						{sessionStarted && generatingNext && !mediaAppendError && !inQueue && (
							<div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-end gap-3 bg-gradient-to-t from-slate-950/85 via-slate-950/15 to-transparent p-5 pb-7 text-center">
								<p className="text-sm font-medium text-white/95">Generating next scene&hellip;</p>
								<div className="h-1.5 w-48 overflow-hidden rounded-full bg-white/15 shadow-sm">
									<div
										className="h-full rounded-full bg-white/85 transition-[width] duration-100 ease-linear"
										style={{ width: `${Math.round(genProgress * 100)}%` }}
									/>
								</div>
							</div>
						)}

						{rewritePending && avPlaybackStarted && (
							<div className="absolute inset-x-0 bottom-0 z-10 flex items-center justify-center gap-2 bg-gradient-to-t from-black/60 to-transparent px-4 pb-12 pt-8 pointer-events-none">
								<Loader2 className="size-4 animate-spin text-white/90" />
								<p className="text-sm font-medium text-white/90">Applying edit&hellip;</p>
							</div>
						)}

						{mediaAppendError && (
							<div className="absolute inset-0 flex items-center justify-center bg-black/80 p-4">
								<div className="max-w-sm rounded-xl border border-rose-500/30 bg-rose-950/50 p-4 text-center text-rose-100 shadow-xl">
									<h2 className="text-base font-semibold">Playback Error</h2>
									<p className="mt-1 text-xs leading-5 text-rose-100/90">{mediaAppendError}</p>
								</div>
							</div>
						)}

						{timeLeft === 0 && gpuAssigned && (
							<div className="absolute inset-0 flex items-center justify-center bg-black/80 p-4">
								<div className="max-w-sm rounded-xl border border-amber-500/25 bg-amber-950/45 p-4 text-center text-amber-50 shadow-xl">
									<h2 className="text-base font-semibold">Session Expired</h2>
									<p className="mt-1 text-xs leading-5 text-amber-50/90">Your session has ended.</p>
								</div>
							</div>
						)}

						{inQueue && (
							<div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black/80 p-4">
								<LoadingSpinner />
								<div className="max-w-sm rounded-xl border border-border bg-card/90 p-4 text-center text-card-foreground shadow-xl backdrop-blur-sm">
									<h2 className="text-base font-semibold">In Queue</h2>
									<p className="mt-1 text-xs leading-5 text-muted-foreground">All GPUs are currently busy.</p>
									<p className="mt-2 text-xs text-foreground">
										Position: <strong>{queuePosition}</strong>
									</p>
								</div>
							</div>
						)}
					</div>

					{onDownload && canDownload && !mediaAppendError && !(timeLeft === 0 && gpuAssigned) && !inQueue && (
						<Button
							onClick={(e) => {
								e.stopPropagation();
								onDownload();
							}}
							size="icon"
							variant="outline"
							aria-label={canShare ? "Share video" : "Download video"}
							className="absolute top-3 left-3 z-10 cursor-pointer bg-slate-800/50 text-white/90 shadow-md backdrop-blur-sm transition-all border-white/30 hover:bg-slate-800/85 hover:border-white/50 hover:text-white hover:scale-105"
						>
							{canShare ? <Share className="size-5" /> : <Download className="size-5" />}
						</Button>
					)}
				</div>
			</div>
		</div>
	);
}
