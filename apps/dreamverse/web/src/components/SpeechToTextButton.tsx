"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { startWhisperStt, getWhisperApiKey, type WhisperSttSession } from "@/lib/whisperStt";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SpeechToTextButtonProps {
	disabled?: boolean;
	onTranscript: (text: string) => void;
	onInterimChange?: (text: string) => void;
	onBusyChange?: (busy: boolean) => void;
}

export default function SpeechToTextButton({ disabled = false, onTranscript, onInterimChange, onBusyChange }: SpeechToTextButtonProps) {
	const [state, setState] = useState<"idle" | "connecting" | "recording" | "transcribing">("idle");
	const sessionRef = useRef<WhisperSttSession | null>(null);
	const errorToastId = "speech-to-text-error";

	const showErrorToast = useCallback((message: string) => {
		toast.error(message, { id: errorToastId });
	}, []);

	useEffect(() => {
		onBusyChange?.(state !== "idle");
	}, [state, onBusyChange]);

	useEffect(() => {
		return () => {
			if (sessionRef.current) {
				sessionRef.current.stop();
				sessionRef.current = null;
			}
		};
	}, []);

	async function startRecording() {
		onInterimChange?.("");

		const apiKey = getWhisperApiKey();
		if (!apiKey) {
			showErrorToast("Speech input is not configured.");
			return;
		}

		setState("connecting");

		try {
			toast.dismiss(errorToastId);
			const session = await startWhisperStt({
				apiKey,
				onError: (err) => {
					showErrorToast(err.message || "Speech recognition error");
					setState("idle");
					sessionRef.current = null;
				},
				onClose: () => {
					sessionRef.current = null;
				},
			});
			sessionRef.current = session;
			setState("recording");
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : "Failed to start recording";
			showErrorToast(msg);
			setState("idle");
			sessionRef.current = null;
		}
	}

	async function handleStop() {
		const session = sessionRef.current;
		if (!session) return;
		sessionRef.current = null;

		setState("transcribing");
		onInterimChange?.("");

		const text = await session.stop();
		if (text) {
			onTranscript(text);
		}
		setState("idle");
	}

	function toggleRecording() {
		if (state === "recording") {
			handleStop();
		} else if (state === "idle") {
			startRecording();
		}
	}

	const recording = state === "recording";
	const busy = state === "connecting" || state === "transcribing";

	return (
		<Button
			type="button"
			variant="ghost"
			size="icon-sm"
			aria-label={recording ? "Stop speech input" : "Start speech input"}
			title={recording ? "Stop speech input" : "Start speech input"}
			onClick={toggleRecording}
			disabled={(disabled && !recording) || busy}
			className={cn(
				"shrink-0 rounded-full transition-all duration-200",
				recording && "animate-stt-pulse border border-rose-500 bg-rose-600/75 text-white shadow-[0_0_0_3px_rgba(239,68,68,0.2)] hover:border-rose-400 hover:bg-rose-600/90 hover:text-white",
				!recording && !busy && "text-muted-foreground hover:text-foreground",
				busy && "text-muted-foreground",
			)}
		>
			{busy ? <Loader2 className="size-5 animate-spin" aria-hidden="true" /> : recording ? <Square className="size-4" aria-hidden="true" /> : <Mic className="size-5" aria-hidden="true" />}
		</Button>
	);
}
