"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { cn } from "@/lib/utils";
import { PlayFilledAlt } from "@carbon/icons-react";
import { Download, Loader2, Share } from "lucide-react";
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
						) : !avPlaybackStarted && !mediaAppendError && !inQueue && loadingAnimation ? (
							<div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-slate-900/60 p-4 backdrop-blur-[2px]">
								<div className="pointer-events-none absolute inset-0 overflow-hidden">
									<div className="absolute inset-0 -translate-x-full animate-[shimmer_3s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-white/[0.04] to-transparent" />
								</div>
								<LoadingSpinner />
								<p className="relative text-sm font-medium text-white/90">{generatingLabel(connected, gpuAssigned)}</p>
							</div>
						) : null}

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
