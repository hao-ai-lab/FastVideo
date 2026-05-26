"use client";
import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Download, Share2 } from "lucide-react";
import DevtoolsShell from "@/components/devtools/DevtoolsShell";
import MonitorPage from "@/components/MonitorPage";
import ChatBar from "@/components/ChatBar";
import SessionTimeoutModal from "@/components/SessionTimeoutModal";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import VideoPlayer from "@/components/VideoPlayer";
import Workspace from "@/components/Workspace";
import { saveProject, saveProjectMetadata, listProjects, loadProjectClips, deleteProject, pruneOldProjects, type StoredProject, type StoredClip } from "@/lib/projectStorage";
import { isInfrastructureError } from "@/lib/ws/reducer";
import { useStore } from "@/hooks/useStore";
import { resolveDevtoolsMode } from "@/lib/devtoolsMode";
import { createAvPipeline, DEFAULT_AV_MIME } from "@/lib/media/avPipeline";
import { remuxArchivedFmp4Segments } from "@/lib/media/fmp4Remux";
import { DEFAULT_CUSTOM_PRESET_ID, parseStoryPresets, sanitizePresetId } from "@/lib/presets";
import {
	buildRewritePromptWindowSnapshot,
	buildRewritePromptWindowSnapshotFromPrompts,
	normalizePromptWindowSnapshot,
} from "@/lib/prompts/promptWindowSnapshot";
import rawPresets from "@/lib/storyPresetsData";
import { cn } from "@/lib/utils";
import { createWebSocketConnection, detachAndCloseWebSocket } from "@/lib/ws/client";
import { decodeWebSocketEvent } from "@/lib/ws/handlers";
import { normalizeSocketMessage } from "@/lib/ws/protocol";
import { applyNormalizedSocketEvent } from "@/lib/ws/reducer";
import { createPromptWindowStore } from "@/stores/promptWindow";
import { createRewriteStore } from "@/stores/rewrite";
import { createSessionStore } from "@/stores/session";
import { createStreamStore } from "@/stores/stream";
import { createUiStore } from "@/stores/ui";
import { Button } from "@/components/ui/button";

const DEFAULT_CUSTOM_PRESET_LABEL = "Custom Editable Preset";
const FIXED_REWRITE_MODEL = "gpt-oss-120b";
const DEFAULT_CURATED_PROMPT_LIMIT = 2;
const MONITOR_ROUTE_PATH = "/internal/f8a3991c/replica-monitor";
const MAX_ARCHIVED_PROJECTS = 10;
const STORAGE_RECOVERY_PREVIOUS_PROJECT_COUNTS = [6, 3, 1, 0];
const FASTVIDEO_REPO_URL = "https://haoailab.com/blogs/dreamverse/";
const FASTVIDEO_BLOG_URL = "https://haoailab.com/blogs/fastvideo_realtime_1080p_part2/";
const BACKEND_PROBE_TIMEOUT_MS = 4000;

interface PageStores {
	sessionStore: ReturnType<typeof createSessionStore>;
	promptWindowStore: ReturnType<typeof createPromptWindowStore>;
	rewriteStore: ReturnType<typeof createRewriteStore>;
	streamStore: ReturnType<typeof createStreamStore>;
	uiStore: ReturnType<typeof createUiStore>;
}

interface ArchivedSegmentLike {
	key: string;
	segmentIdx: number | null;
	streamId: string;
	mime: string;
	completed: boolean;
	chunks: ArrayBuffer[];
}

interface BackendProbeResponse {
	ok: boolean;
	status: number;
	payload: any;
	errorMessage: string;
}

interface BackendReadinessProbe {
	ok: boolean;
	notice: string;
}

function yieldToEventLoop(): Promise<void> {
	return new Promise((r) => setTimeout(r, 0));
}

const HERO_WAVE_LIGHT = ["#2A4A98", "#4878E5", "#6FA0F2", "#B0BCC8", "#E8D99E", "#D8C844", "#C2A620"];
const HERO_WAVE_DARK = ["#143468", "#1E58B8", "#3892F0", "#80B8E8", "#B8D0EA", "#E2D498", "#DABB50"];
const HERO_TEXT = "Direct scenes in seconds";

function HeroTagline() {
	const ref = useRef<HTMLHeadingElement>(null);

	useEffect(() => {
		const el = ref.current;
		if (!el) return;

		let rafId = 0;

		function play() {
			const chars = el!.querySelectorAll<HTMLSpanElement>("[data-char]");
			if (!chars.length) return;
			cancelAnimationFrame(rafId);

			const isDark = document.documentElement.classList.contains("dark");
			const colors = isDark ? HERO_WAVE_DARK : HERO_WAVE_LIGHT;
			const waveLen = 10;
			const total = chars.length + waveLen;
			const duration = 1200;
			const maxBlur = 3.5;
			const start = performance.now();

			function tick() {
				const t = Math.min((performance.now() - start) / duration, 1);
				const pos = t * total;
				chars.forEach((ch, i) => {
					const rel = pos - i;
					if (rel >= 0 && rel < waveLen) {
						const norm = rel / waveLen;
						const ci = Math.floor(norm * colors.length);
						ch.style.color = colors[Math.min(colors.length - 1, ci)];

						let blur = 0;
						if (norm < 0.25) {
							blur = maxBlur * (1 - norm / 0.25);
						} else if (norm > 0.75) {
							blur = maxBlur * ((norm - 0.75) / 0.25);
						}
						ch.style.filter = blur > 0.1 ? `blur(${blur.toFixed(1)}px)` : "";
					} else {
						ch.style.color = "";
						ch.style.filter = "";
					}
				});
				if (t < 1) {
					rafId = requestAnimationFrame(tick);
				} else {
					chars.forEach((ch) => {
						ch.style.color = "";
						ch.style.filter = "";
					});
				}
			}

			rafId = requestAnimationFrame(tick);
		}

		const initialDelay = setTimeout(play, 400);
		const interval = setInterval(play, 5000);
		return () => {
			clearTimeout(initialDelay);
			clearInterval(interval);
			cancelAnimationFrame(rafId);
		};
	}, []);

	return (
		<h1 ref={ref} className="text-center text-3xl font-medium text-[#343537] dark:text-[#FAFAFB] sm:text-4xl">
			{HERO_TEXT.split(" ").map((word, wi) => (
				<Fragment key={wi}>
					{wi > 0 && (
						<span data-char className="transition-[color,filter] duration-150">
							{" "}
						</span>
					)}
					<span className="inline-flex">
						{word.split("").map((char, ci) => (
							<span key={ci} data-char className="inline-block transition-[color,filter] duration-150">
								{char}
							</span>
						))}
					</span>
				</Fragment>
			))}
		</h1>
	);
}

export default function Page() {
	const storesRef = useRef<PageStores | null>(null);
	if (!storesRef.current) {
		const nextStoryPresets = parseStoryPresets(rawPresets);
		const nextSimpleMode = false;
		const nextDevtoolsMode = resolveDevtoolsModeFromRuntime();
		const nextDemoMode = nextDevtoolsMode ? false : resolveDemoModeFromRuntime();
		const nextEditableMode = nextDevtoolsMode;
		const nextIsMonitorRoute = typeof window !== "undefined" && window.location.pathname === MONITOR_ROUTE_PATH;

		const nextSessionStore = createSessionStore({
			enhancementEnabled: true,
			autoExtensionEnabled: false,
			loopGenerationEnabled: false,
			livePromptRewriteMode: !nextSimpleMode && !nextDevtoolsMode && !nextDemoMode,
		});
		const nextPromptWindowStore = createPromptWindowStore({
			curatedPromptLimit: DEFAULT_CURATED_PROMPT_LIMIT,
			storyPresets: nextStoryPresets as Record<string, unknown>[],
			simpleMode: nextSimpleMode,
			editableMode: nextEditableMode,
		});
		const nextRewriteStore = createRewriteStore();
		const nextStreamStore = createStreamStore();
		const nextUiStore = createUiStore({
			isMonitorRoute: nextIsMonitorRoute,
			simpleMode: nextSimpleMode,
			devtoolsMode: nextDevtoolsMode,
			demoMode: nextDemoMode,
			editableMode: nextEditableMode,
		});

		if (nextEditableMode) {
			nextPromptWindowStore.seedEditableFromPreset((nextStoryPresets[0] ?? null) as Record<string, unknown> | null, {
				defaultCustomPresetId: DEFAULT_CUSTOM_PRESET_ID,
				defaultCustomPresetLabel: DEFAULT_CUSTOM_PRESET_LABEL,
			});
		}

		storesRef.current = {
			sessionStore: nextSessionStore,
			promptWindowStore: nextPromptWindowStore,
			rewriteStore: nextRewriteStore,
			streamStore: nextStreamStore,
			uiStore: nextUiStore,
		};
	}

	const { sessionStore, promptWindowStore, rewriteStore, streamStore, uiStore } = storesRef.current;

	const sessionState = useStore(sessionStore);
	const promptWindowState = useStore(promptWindowStore);
	const rewriteState = useStore(rewriteStore);
	const streamState = useStore(streamStore);
	const uiState = useStore(uiStore);

	const {
		connected,
		connecting,
		sessionStarted,
		sessionTimeout,
		timeLeft,
		queuePosition,
		gpuAssigned,
		enhancementEnabled,
		promptExtensionError,
		autoExtensionEnabled,
		autoExtensionTimeoutHint,
		loopGenerationEnabled,
		generationPaused,
		livePromptDraft,
		sessionNotice,
		generationCapReached,
		generationSegmentCap,
		generatedSegmentCount,
		preservePlaybackOnClose,
		livePromptRewriteMode,
		sessionExpired,
		projectResetPending,
	} = sessionState;

	const {
		storyPresets,
		selectedPresetId,
		selectedPreset,
		previewSegments,
		maxCuratedPromptCount,
		curatedPromptLimit,
		outboundCuratedPrompts,
		seedPrompts,
		currentPromptWindowPrompts,
		outboundSessionPrompts,
		canJoinSession,
		editableSegments,
		sanitizedEditableSegments,
		editableCanJoin,
		editableDirty,
		customPresetId,
		customPresetLabel,
	} = promptWindowState;

	const { rewritingSeedPrompts, promptEvents, lastPromptSource, pendingSegmentSource } = rewriteState;

	const {
		playingSeedPromptIndex,
		generatingSeedPromptIndex,
		seedPromptIndexBySegment,
		currentSegmentNumber,
		promptHistory,
		selectedHistoryId,
		selectedHistoryEntry,
		completedClips,
		activeClipId,
		activeClip,
		activePlaybackStartTime,
		pendingClip,
		liveClip,
		loadingAnimation,
		avPlaybackStarted,
		mediaAppendError,
		lastVideoCompletedAtMs,
		timeBetweenVideosMs,
	} = streamState;

	const {
		isMonitorRoute,
		demoMode,
		devtoolsMode,
		editableMode,
		appendingPromptWindow,
		appendPromptWindowStatus,
		appendPromptWindowError,
		promptConfigEditorOpen,
		promptConfigLoading,
		promptConfigSaving,
		promptConfigLoaded,
		promptConfigError,
		promptConfigStatus,
		nextSegmentPromptEditorOpen,
		autoExtensionPromptEditorOpen,
		rewriteWindowPromptEditorOpen,
		nextSegmentSystemPromptDraft,
		autoExtensionSystemPromptDraft,
		rewriteWindowSystemPromptDraft,
	} = uiState;

	const wsRef = useRef<WebSocket | null>(null);
	const [runtimeReady, setRuntimeReady] = useState(false);
	const countdownIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const ttffStartAtMsRef = useRef<number | null>(null);
	const [ttffValueMs, setTtffValueMs] = useState<number | null>(null);
	const ttffIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const pendingInitialPromptRef = useRef("");
	const lastArchivedReplayKeyRef = useRef("");
	const [sidebarOpen, setSidebarOpen] = useState(false);
	const [currentThumbnail, setCurrentThumbnail] = useState<string | null>(null);
	const currentProjectIdRef = useRef("");
	const currentProjectCreatedAtRef = useRef(0);
	const [savedProjects, setSavedProjects] = useState<StoredProject[]>([]);
	const [viewingProject, setViewingProject] = useState<{
		project: StoredProject;
		clips: { id: string; label: string; prompt: string; mime: string; objectUrl: string; blob: Blob; createdAt: number }[];
	} | null>(null);
	const viewingProjectRef = useRef(viewingProject);
	viewingProjectRef.current = viewingProject;
	const saveQueueRef = useRef<Promise<void>>(Promise.resolve());
	const downloadInFlightRef = useRef(false);
	const wsMessageQueueRef = useRef<Promise<void>>(Promise.resolve());
	const [isMobileShareCapable, setIsMobileShareCapable] = useState(false);
	const [videoMuted, setVideoMuted] = useState(true);
	const [timeoutModalOpen, setTimeoutModalOpen] = useState(false);
	useEffect(() => {
		setIsMobileShareCapable(typeof navigator.canShare === "function" && window.matchMedia("(pointer: coarse)").matches);
	}, []);

	const videoElRef = useRef<HTMLVideoElement | null>(null);
	const archivedPlaybackElRef = useRef<HTMLVideoElement | null>(null);
	const viewingModePlaybackStateRef = useRef<{
		liveWasPlaying: boolean;
		archivedWasPlaying: boolean;
	}>({
		liveWasPlaying: false,
		archivedWasPlaying: false,
	});
	const previousViewingModeRef = useRef(false);

	const avPipelineRef = useRef(
		createAvPipeline({
			getVideoEl: () => videoElRef.current,
			onAppendError: (message: string, error: unknown) => {
				streamStore.patch({ mediaAppendError: message });
				console.error("media append failed:", error);
			},
			onPlaybackStarted: () => {
				streamStore.patch({ avPlaybackStarted: true, loadingAnimation: false });
			},
		}),
	);

	const archivedPlaybackPipelineRef = useRef(
		createAvPipeline({
			getVideoEl: () => archivedPlaybackElRef.current,
			onAppendError: (_message: string, error: unknown) => {
				console.error("archived playback failed:", error);
			},
		}),
	);

	const avPipeline = avPipelineRef.current;
	const archivedPlaybackPipeline = archivedPlaybackPipelineRef.current;

	const videoRefCallback = useCallback((el: HTMLVideoElement | null) => {
		videoElRef.current = el;
	}, []);

	const archivedPlaybackRefCallback = useCallback((el: HTMLVideoElement | null) => {
		archivedPlaybackElRef.current = el;
	}, []);

	// --- Derived values ---

	const canStartSession = !projectResetPending && (canJoinSession || Boolean(normalizeInitialPrompt(livePromptDraft as string)));

	const currentClipLabel = useMemo(() => {
		if ((activeClip as Record<string, any>)?.label) return (activeClip as Record<string, any>).label;
		if ((pendingClip as Record<string, any>)?.label) return (pendingClip as Record<string, any>).label;
		if ((liveClip as Record<string, any>)?.label) return (liveClip as Record<string, any>).label;
		const initLabel = getInitialPresetLabel(pendingInitialPromptRef.current || (livePromptDraft as string));
		if (initLabel) return initLabel;
		if ((selectedPreset as Record<string, any>)?.label) return (selectedPreset as Record<string, any>).label;
		return "Video player";
	}, [activeClip, pendingClip, liveClip, livePromptDraft, selectedPreset, customPresetLabel, selectedPresetId]);

	const currentClipPrompt = useMemo(() => {
		if ((activeClip as Record<string, any>)?.prompt) return (activeClip as Record<string, any>).prompt;
		if ((pendingClip as Record<string, any>)?.prompt) return (pendingClip as Record<string, any>).prompt;
		if ((liveClip as Record<string, any>)?.prompt) return (liveClip as Record<string, any>).prompt;
		const initPrompt = normalizeInitialPrompt(pendingInitialPromptRef.current || (livePromptDraft as string));
		if (initPrompt) return initPrompt;
		const presetPrompts = (selectedPreset as Record<string, any>)?.segment_prompts;
		return Array.isArray(presetPrompts) ? presetPrompts[0] || "" : "";
	}, [activeClip, pendingClip, liveClip, livePromptDraft, selectedPreset, outboundSessionPrompts]);

	const currentProjectTitle = useMemo(() => {
		const presetLabel = (selectedPreset as Record<string, any>)?.label;
		if (presetLabel) return presetLabel;
		const events = promptEvents as Record<string, any>[];
		for (let i = events.length - 1; i >= 0; i--) {
			const e = events[i];
			if (String(e?.source || "") === "user_rewrite" && typeof e?.text === "string" && e.text.trim()) {
				return e.text.trim();
			}
		}
		return "Untitled project";
	}, [selectedPreset, promptEvents]);

	const canSubmitContinuation = sessionStarted && connected && gpuAssigned && !projectResetPending && Boolean((livePromptDraft as string).trim());

	const showLivePlayback = !activeClip;
	const canDownloadVideo = useMemo(() => {
		const currentActiveClip = activeClip as Record<string, any> | null;
		if (currentActiveClip?.blob instanceof Blob) return true;
		return (completedClips as Record<string, any>[]).some((clip) => clip?.blob instanceof Blob);
	}, [activeClip, completedClips]);

	const hasEdits = useMemo(
		() => Boolean(sessionStarted) && (promptEvents as Record<string, any>[]).some((e) => typeof e?.text === "string" && e.text.trim() && String(e?.source || "").trim() === "user_rewrite"),
		[sessionStarted, promptEvents],
	);

	// Viewing mode: track which clip is selected (defaults to last clip)
	const [viewingSelectedClipId, setViewingSelectedClipId] = useState<string>("");
	const viewingSelectedClip = useMemo(() => {
		if (!viewingProject) return null;
		const lastClip = viewingProject.clips[viewingProject.clips.length - 1] ?? null;
		if (viewingSelectedClipId) {
			return viewingProject.clips.find((c) => c.id === viewingSelectedClipId) ?? lastClip;
		}
		return lastClip;
	}, [viewingProject, viewingSelectedClipId]);

	const activeReplayKey = useMemo(() => {
		if (!activeClip) return "";
		const clip = activeClip as Record<string, any>;
		return [clip.id || "", activePlaybackStartTime || 0, Array.isArray(clip.chunks) ? clip.chunks.length : 0, clip.objectUrl || ""].join(":");
	}, [activeClip, activePlaybackStartTime]);
	const isViewingMode = Boolean(viewingProject);

	// --- Reactive effect: archived playback ---

	useEffect(() => {
		if (showLivePlayback || !activeReplayKey || !archivedPlaybackElRef.current) {
			if (lastArchivedReplayKeyRef.current) {
				lastArchivedReplayKeyRef.current = "";
				archivedPlaybackPipeline.reset();
			}
		} else if (activeReplayKey !== lastArchivedReplayKeyRef.current) {
			lastArchivedReplayKeyRef.current = activeReplayKey;
			void restoreArchivedPlayback(activeClip as Record<string, any>, activePlaybackStartTime as number);
		}
	}, [showLivePlayback, activeReplayKey]);

	useLayoutEffect(() => {
		const wasViewingMode = previousViewingModeRef.current;
		const liveEl = videoElRef.current;
		const archivedEl = archivedPlaybackElRef.current;

		if (!wasViewingMode && isViewingMode) {
			viewingModePlaybackStateRef.current = {
				liveWasPlaying: Boolean(liveEl && !liveEl.paused && !liveEl.ended),
				archivedWasPlaying: Boolean(archivedEl && !archivedEl.paused && !archivedEl.ended),
			};
			if (liveEl && !liveEl.paused) {
				liveEl.pause();
			}
			if (archivedEl && !archivedEl.paused) {
				archivedEl.pause();
			}
		} else if (wasViewingMode && !isViewingMode) {
			const { liveWasPlaying, archivedWasPlaying } = viewingModePlaybackStateRef.current;
			const activeEl = showLivePlayback ? videoElRef.current : archivedPlaybackElRef.current;
			const shouldResume = showLivePlayback ? liveWasPlaying : archivedWasPlaying;
			if (shouldResume && activeEl && activeEl.paused && activeEl.readyState >= 2) {
				const playPromise = activeEl.play();
				if (playPromise?.catch) playPromise.catch(() => {});
			}
			viewingModePlaybackStateRef.current = {
				liveWasPlaying: false,
				archivedWasPlaying: false,
			};
		}

		previousViewingModeRef.current = isViewingMode;
	}, [isViewingMode, showLivePlayback]);

	useEffect(() => {
		if (sessionExpired && Number(timeLeft) === 0) {
			setTimeoutModalOpen(true);
		}
	}, [sessionExpired, timeLeft]);

	useEffect(() => {
		if (!sessionExpired) {
			setTimeoutModalOpen(false);
		}
	}, [sessionExpired]);

	// --- Initialization ---

	const initializedRef = useRef(false);
	useEffect(() => {
		setRuntimeReady(true);
	}, []);

	useEffect(() => {
		if (!runtimeReady || initializedRef.current) return;
		initializedRef.current = true;

		if (uiStore.get().devtoolsMode) {
			Promise.resolve().then(() => loadStoryPresetsFromServer());
		}
		if (!uiStore.get().devtoolsMode) {
			void refreshSavedProjects().then(() => {
				try {
					const lastId = localStorage.getItem("fastvideo-active-project");
					if (lastId) {
						localStorage.removeItem("fastvideo-active-project");
						void viewSavedProject(lastId);
					}
				} catch (_) {
					/* private browsing */
				}
			});
		}
	}, [runtimeReady, uiStore]);

	// --- Save project on page unload ---

	useEffect(() => {
		function triggerSaveIfActive() {
			if (currentProjectIdRef.current && sessionStore.get().sessionStarted) {
				void saveCurrentProject();
			}
		}
		function handleBeforeUnload() {
			triggerSaveIfActive();
		}
		function handleVisibilityChange() {
			if (document.visibilityState === "hidden") {
				triggerSaveIfActive();
			} else if (document.visibilityState === "visible") {
				if (viewingProjectRef.current) {
					return;
				}
				// Mobile browsers pause <video> when the page is backgrounded.
				// Resume playback on the active video element when returning.
				const hasActiveClip = Boolean(streamStore.get().activeClipId);
				const activeEl = hasActiveClip ? archivedPlaybackElRef.current : videoElRef.current;
				if (activeEl && activeEl.paused && activeEl.readyState >= 2) {
					const p = activeEl.play();
					if (p?.catch) p.catch(() => {});
				}
			}
		}
		window.addEventListener("beforeunload", handleBeforeUnload);
		document.addEventListener("visibilitychange", handleVisibilityChange);
		return () => {
			window.removeEventListener("beforeunload", handleBeforeUnload);
			document.removeEventListener("visibilitychange", handleVisibilityChange);
		};
	}, []);

	// --- Periodic auto-save as safety net ---

	useEffect(() => {
		const AUTO_SAVE_INTERVAL_MS = 30_000;
		const interval = setInterval(() => {
			if (currentProjectIdRef.current && sessionStore.get().sessionStarted) {
				void saveCurrentProject();
			}
		}, AUTO_SAVE_INTERVAL_MS);
		return () => clearInterval(interval);
	}, []);

	// --- Cleanup on unmount ---

	useEffect(() => {
		return () => {

			clearCountdownInterval();
			clearTtffInterval();
			if (wsRef.current) {
				detachAndCloseWebSocket(wsRef.current);
				wsRef.current = null;
			}
			resetPlaybackState();
			revokeCompletedClipUrls();
			if (viewingProjectRef.current) {
				viewingProjectRef.current.clips.forEach((clip) => URL.revokeObjectURL(clip.objectUrl));
			}
		};
	}, []);

	// --- Helper functions ---

	function resolveDevtoolsModeFromRuntime() {
		return resolveDevtoolsMode({
			buildEnabled: Boolean(process.env.NEXT_PUBLIC_INCLUDE_DEVTOOLS),
			search: typeof window === "undefined" ? "" : window.location.search,
		});
	}

	function resolveDemoModeFromRuntime() {
		const search = typeof window === "undefined" ? "" : window.location.search;
		const params = new URLSearchParams(search || "");
		if (!params.has("demo")) return false;
		const value = String(params.get("demo") || "")
			.trim()
			.toLowerCase();
		return value === "" || value === "1" || value === "true" || value === "yes" || value === "on";
	}

	async function fetchBackendProbe(path: string): Promise<BackendProbeResponse> {
		const controller = typeof AbortController !== "undefined"
			? new AbortController()
			: null;
		const timeoutId = controller
			? window.setTimeout(() => controller.abort(), BACKEND_PROBE_TIMEOUT_MS)
			: null;

		try {
			const response = await fetch(path, {
				headers: { Accept: "application/json" },
				cache: "no-store",
				signal: controller?.signal,
			});
			let payload: any = null;
			try {
				payload = await response.json();
			} catch (_) {
				payload = null;
			}
			const fallbackMessage = typeof payload?.detail === "string"
				? payload.detail
				: `Request failed with status ${response.status}.`;
			return {
				ok: response.ok,
				status: response.status,
				payload,
				errorMessage: fallbackMessage,
			};
		} catch (error: any) {
			const message = error?.name === "AbortError"
				? "Backend probe timed out."
				: error?.message || String(error);
			return {
				ok: false,
				status: 0,
				payload: null,
				errorMessage: message,
			};
		} finally {
			if (timeoutId !== null) {
				window.clearTimeout(timeoutId);
			}
		}
	}

	function resolveBackendUnavailableNotice(): string {
		return "Dreamverse backend is not reachable. Start uv run dreamverse-server and wait for /readyz to return 200 before retrying.";
	}

	function resolveBackendNotReadyNotice(detail: string, statusPayload: any): string {
		const totalGpus = Number(statusPayload?.total_gpus);
		const warmupFailures = Number(statusPayload?.warmup_failed_gpus);
		if (Number.isFinite(totalGpus) && totalGpus <= 0) {
			return "Dreamverse backend is running, but no GPUs were detected. Confirm that your local GPU is visible to FastVideo, then retry.";
		}
		if (Number.isFinite(warmupFailures) && warmupFailures > 0) {
			return `Dreamverse backend is running, but GPU warmup failed on ${warmupFailures} worker${warmupFailures === 1 ? "" : "s"}. Check the backend logs and FastVideo/model runtime setup, then retry.`;
		}
		if (detail === "Prompt enhancer not initialized.") {
			return "Dreamverse backend is still initializing prompt services. Wait for /readyz to return 200 and retry.";
		}
		if (detail === "No ready GPU worker processes.") {
			return "Dreamverse backend is running, but GPU workers are not ready yet. Wait for startup warmup to finish and retry.";
		}
		return `Dreamverse backend is not ready yet: ${detail}`;
	}

	async function probeBackendReadiness(): Promise<BackendReadinessProbe> {
		const health = await fetchBackendProbe("/healthz");
		if (!health.ok) {
			return {
				ok: false,
				notice: resolveBackendUnavailableNotice(),
			};
		}

		const ready = await fetchBackendProbe("/readyz");
		if (ready.ok) {
			return {
				ok: true,
				notice: "",
			};
		}

		const status = await fetchBackendProbe("/status");
		const detail = typeof ready.payload?.detail === "string" && ready.payload.detail.trim()
			? ready.payload.detail.trim()
			: ready.errorMessage;
		return {
			ok: false,
			notice: resolveBackendNotReadyNotice(detail, status.payload),
		};
	}

	function clearPendingProjectPointers() {
		clearSessionArchivedClips();
		currentProjectIdRef.current = "";
		currentProjectCreatedAtRef.current = 0;
		markActiveProjectId(null);
		setCurrentThumbnail(null);
	}

	function showPreSessionNotice(notice: string) {
		sessionStore.patch({
			connected: false,
			connecting: false,
			gpuAssigned: false,
			sessionNotice: notice,
			sessionExpired: false,
		});
	}

	function recoverFailedSessionStart(notice: string) {
		const restoredDraft = normalizeInitialPrompt(pendingInitialPromptRef.current);
		resetToLobbyState();
		clearPendingProjectPointers();
		pendingInitialPromptRef.current = "";
		sessionStore.patch({
			connected: false,
			connecting: false,
			gpuAssigned: false,
			livePromptDraft: restoredDraft,
			sessionNotice: notice,
			sessionExpired: false,
		});
	}

	async function loadStoryPresetsFromServer() {
		if (!uiStore.get().devtoolsMode) return;
		try {
			const response = await fetch("/curated-presets");
			const payload = await response.json();
			if (!response.ok) {
				throw new Error(payload.detail || "Failed to load curated presets.");
			}
			const nextStoryPresets = parseStoryPresets(payload.presets);
			if (nextStoryPresets.length === 0) return;
			promptWindowStore.setStoryPresets(nextStoryPresets);
			const nextSelectedPreset = nextStoryPresets.find((preset: any) => preset.id === promptWindowStore.get().selectedPresetId) || nextStoryPresets[0];
			promptWindowStore.setSelectedPresetId(nextSelectedPreset.id);
			const pwState = promptWindowStore.get();
			if (uiStore.get().editableMode && !pwState.editableDirty) {
				seedEditableFromPreset(nextSelectedPreset);
			}
		} catch (error) {
			console.error("Failed to load curated presets:", error);
		}
	}

	function seedEditableFromPreset(preset: any) {
		promptWindowStore.seedEditableFromPreset(preset, {
			defaultCustomPresetId: DEFAULT_CUSTOM_PRESET_ID,
			defaultCustomPresetLabel: DEFAULT_CUSTOM_PRESET_LABEL,
		});
	}

	// --- Preset & prompt handlers ---

	function handlePresetSelectionChange(event: any) {
		const nextPresetId = event.currentTarget.value;
		if (nextPresetId === promptWindowStore.get().selectedPresetId) return;
		const pwState = promptWindowStore.get();
		if (uiStore.get().editableMode && pwState.editableDirty) {
			const shouldOverwrite = window.confirm("Switching presets will replace your edited segments. Continue?");
			if (!shouldOverwrite) {
				event.currentTarget.value = promptWindowStore.get().selectedPresetId;
				return;
			}
		}
		promptWindowStore.setSelectedPresetId(nextPresetId);
		if (uiStore.get().editableMode) {
			const nextPreset = (promptWindowStore.get().storyPresets as any[]).find((preset: any) => preset.id === nextPresetId) || null;
			seedEditableFromPreset(nextPreset);
		}
	}

	function handlePresetGenerate(presetId: string) {
		if (sessionStore.get().sessionStarted || sessionStore.get().projectResetPending) return;
		promptWindowStore.setSelectedPresetId(presetId);
		joinSession({ force: true });
	}

	function normalizeInitialPrompt(value: unknown): string {
		return typeof value === "string" ? value.trim() : "";
	}

	function shouldStartFromCustomPrompt(prompt?: string): boolean {
		const p = prompt ?? (pendingInitialPromptRef.current || (sessionStore.get().livePromptDraft as string));
		const normalizedPrompt = normalizeInitialPrompt(p);
		return Boolean(normalizeInitialPrompt(pendingInitialPromptRef.current) || (!sessionStore.get().sessionStarted && normalizedPrompt));
	}

	function getInitialPresetId(prompt?: string): string {
		if (shouldStartFromCustomPrompt(prompt)) {
			return sanitizePresetId(promptWindowStore.get().customPresetId as string) || DEFAULT_CUSTOM_PRESET_ID;
		}
		return (promptWindowStore.get().selectedPresetId as string) || sanitizePresetId(promptWindowStore.get().customPresetId as string);
	}

	function getInitialPresetLabel(prompt?: string): string {
		if (shouldStartFromCustomPrompt(prompt)) {
			return String(promptWindowStore.get().customPresetLabel || "").trim() || "Custom rollout";
		}
		const sp = promptWindowStore.get().selectedPreset as Record<string, any> | null;
		return sp?.label || String(promptWindowStore.get().customPresetLabel || "").trim() || "Current rollout";
	}

	function getSessionInitPrompts(): string[] {
		if (shouldStartFromCustomPrompt()) return [];
		return promptWindowStore.get().outboundSessionPrompts as string[];
	}

	// --- Playback & clip management ---

	function makePromptId(): string {
		if (typeof crypto !== "undefined" && crypto.randomUUID) {
			return crypto.randomUUID();
		}
		return `prompt_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
	}

	function resetPlaybackState() {
		avPipeline.reset();
		archivedPlaybackPipeline.reset();
		lastArchivedReplayKeyRef.current = "";
		streamStore.patch({
			loadingAnimation: false,
			mediaAppendError: null,
			avPlaybackStarted: false,
		});
	}

	function clearCountdownInterval() {
		if (countdownIntervalRef.current) {
			clearInterval(countdownIntervalRef.current);
			countdownIntervalRef.current = null;
		}
	}

	function clearTtffInterval() {
		if (ttffIntervalRef.current) {
			clearInterval(ttffIntervalRef.current);
			ttffIntervalRef.current = null;
		}
	}

	function resetTtffTimer() {
		clearTtffInterval();
		ttffStartAtMsRef.current = null;
		setTtffValueMs(null);
	}

	function startTtffTimer() {
		clearTtffInterval();
		ttffStartAtMsRef.current = performance.now();
		setTtffValueMs(0);
		ttffIntervalRef.current = setInterval(() => {
			if (ttffStartAtMsRef.current !== null) {
				setTtffValueMs(performance.now() - ttffStartAtMsRef.current);
			}
		}, 16);
	}

	function markFirstFrameRendered() {
		const now = performance.now();
		if (ttffStartAtMsRef.current !== null && ttffIntervalRef.current) {
			setTtffValueMs(now - ttffStartAtMsRef.current);
			clearTtffInterval();
		}
		if (streamStore.get().lastVideoCompletedAtMs !== null) {
			streamStore.patch({
				timeBetweenVideosMs: now - (streamStore.get().lastVideoCompletedAtMs as number),
				lastVideoCompletedAtMs: null,
			});
		}
		setTimeout(() => {
			if (!showLivePlayback) return;
			const thumb = captureVideoThumbnail();
			if (thumb) {
				setCurrentThumbnail(thumb);
				const events = rewriteStore.get().promptEvents as Record<string, any>[];
				const latest = events.find((e) => String(e?.source || "") === "user_rewrite");
				if (latest?.promptId) {
					rewriteStore.trackPromptEvent(latest.promptId, { resultThumbnail: thumb });
				}
				void saveCurrentProject();
			}
		}, 500);
	}

	function startSessionCountdown() {
		clearCountdownInterval();
		countdownIntervalRef.current = setInterval(() => {
			const currentTimeLeft = sessionStore.get().timeLeft as number;
			if (typeof currentTimeLeft === "number" && currentTimeLeft > 0) {
				sessionStore.patch({ timeLeft: currentTimeLeft - 1 });
			} else {
				clearCountdownInterval();
			}
		}, 1000);
	}

	function shouldUseArchivedPlaybackFallback(): boolean {
		return typeof avPipeline.usesNativePlaybackFallback === "function" && avPipeline.usesNativePlaybackFallback();
	}

	function shouldPreferArchivedBlobPlayback(clip: any): boolean {
		return Boolean(clip?.objectUrl && clip?.blob instanceof Blob);
	}

	function cloneArchivedChunk(chunk: any): any {
		if (chunk instanceof ArrayBuffer) return chunk.slice(0);
		if (ArrayBuffer.isView(chunk)) {
			return chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength);
		}
		return chunk;
	}

	function normalizeArchivedSegments(rawSegments: any): ArchivedSegmentLike[] {
		if (!Array.isArray(rawSegments)) return [];
		const normalized: ArchivedSegmentLike[] = [];
		rawSegments.forEach((segment: any, index: number) => {
			const rawChunks = Array.isArray(segment?.chunks) ? segment.chunks : [];
			const chunks = rawChunks
				.map((chunk: any) => {
					const cloned = cloneArchivedChunk(chunk);
					return cloned instanceof ArrayBuffer ? cloned : null;
				})
				.filter((chunk: ArrayBuffer | null): chunk is ArrayBuffer => chunk instanceof ArrayBuffer);
			if (chunks.length === 0) return;
			const streamId = typeof segment?.streamId === "string" && segment.streamId.trim() ? segment.streamId.trim() : `segment-${index + 1}`;
			const segmentIdx = Number.isInteger(segment?.segmentIdx) ? Number(segment.segmentIdx) : null;
			normalized.push({
				key: typeof segment?.key === "string" && segment.key.trim() ? segment.key.trim() : `${segmentIdx ?? "na"}:${streamId}`,
				segmentIdx,
				streamId,
				mime: typeof segment?.mime === "string" && segment.mime.trim() ? segment.mime.trim() : DEFAULT_AV_MIME,
				completed: Boolean(segment?.completed),
				chunks,
			});
		});
		return normalized;
	}

	async function remuxArchivedSegmentsBestEffort(segments: ArchivedSegmentLike[], label: string): Promise<Blob | null> {
		const normalizedSegments = normalizeArchivedSegments(segments);
		if (normalizedSegments.length === 0) return null;

		try {
			return await remuxArchivedFmp4Segments(normalizedSegments, {
				includeInProgress: true,
				mimeType: "video/mp4",
			});
		} catch (error) {
			try {
				return await remuxArchivedFmp4Segments(normalizedSegments, {
					includeInProgress: false,
					mimeType: "video/mp4",
				});
			} catch (fallbackError) {
				console.warn("Unable to remux archived segments:", {
					error,
					fallbackError,
					label,
				});
				return null;
			}
		}
	}

	async function archiveCompletedClip() {
		const lrc = streamStore.get().liveClip as Record<string, any> | null;
		if (!lrc || !avPipeline.hasArchivedChunks()) return null;
		const previewBlob = avPipeline.buildArchivedStreamBlob();
		const archivedSegments = normalizeArchivedSegments(typeof avPipeline.takeArchivedSegmentSnapshots === "function" ? avPipeline.takeArchivedSegmentSnapshots({ includeInProgress: true }) : []);
		const chunks = avPipeline.takeArchivedStreamChunks();
		const rawBlob =
			previewBlob instanceof Blob && previewBlob.size > 0
				? new Blob(chunks, {
						type: previewBlob.type || DEFAULT_AV_MIME,
					})
				: null;
		if (!(rawBlob instanceof Blob) || rawBlob.size === 0) return null;

		let blob = rawBlob;
		let remuxed = false;
		if (archivedSegments.length > 0) {
			const remuxedBlob = await remuxArchivedSegmentsBestEffort(archivedSegments, lrc.label || "Generated clip");
			if (remuxedBlob instanceof Blob && remuxedBlob.size > 0) {
				blob = remuxedBlob;
				remuxed = true;
			}
		}
		const objectUrl = URL.createObjectURL(blob);
		const archivedClip = {
			id: makePromptId(),
			label: lrc.label || "Generated clip",
			prompt: lrc.prompt || "",
			promptWindowPrompts: clonePromptWindowPrompts(lrc.promptWindowPrompts),
			mime: blob.type || DEFAULT_AV_MIME,
			blob,
			objectUrl,
			chunks: chunks.map((chunk: any) => cloneArchivedChunk(chunk)),
			archivedSegments,
			remuxed,
			createdAt: Date.now(),
		};
		streamStore.addCompletedClip(archivedClip);

		// Link this archived clip to the most recent unlinked user_rewrite prompt event
		const events = rewriteStore.get().promptEvents as Record<string, any>[];
		const unlinkedEvent = events.find(
			(e) => String(e?.source || "") === "user_rewrite" && !e.clipId
		);
		if (unlinkedEvent?.promptId) {
			rewriteStore.trackPromptEvent(unlinkedEvent.promptId, { clipId: archivedClip.id });
		}

		return archivedClip;
	}

	function revokeCompletedClipUrls() {
		(streamStore.get().completedClips as any[]).forEach((clip: any) => {
			if (clip?.objectUrl) URL.revokeObjectURL(clip.objectUrl);
		});
	}

	function clearSessionArchivedClips() {
		revokeCompletedClipUrls();
		streamStore.patch({
			completedSimpleClips: [],
			completedClips: [],
		});
	}

	function selectClip(clipId: string, playbackStartTime = 0) {
		const clip = (streamStore.get().completedClips as any[]).find((item: any) => item.id === clipId) || null;
		if (!clip) return;
		streamStore.selectClip(clip.id, playbackStartTime);
	}

	async function playArchivedObjectUrl(clip: any, playbackStartTime = 0) {
		const el = archivedPlaybackElRef.current;
		if (!el || !clip?.objectUrl) return;
		el.src = clip.objectUrl;
		el.load();
		try {
			if (Number.isFinite(playbackStartTime) && playbackStartTime > 0) {
				el.currentTime = playbackStartTime;
			}
		} catch (_) {
			/* noop */
		}
		const playPromise = el.play();
		if (playPromise?.catch) playPromise.catch(() => {});
	}

	async function restoreArchivedPlayback(clip: any, playbackStartTime = 0) {
		if (!clip || !archivedPlaybackElRef.current) return;
		if (shouldUseArchivedPlaybackFallback() || shouldPreferArchivedBlobPlayback(clip)) {
			archivedPlaybackPipeline.reset();
			await yieldToEventLoop();
			await playArchivedObjectUrl(clip, playbackStartTime);
			return;
		}
		const clipChunks = Array.isArray(clip.chunks) ? clip.chunks : [];
		archivedPlaybackPipeline.reset();
		await yieldToEventLoop();
		if (!archivedPlaybackElRef.current) return;
		if (clipChunks.length === 0) {
			await playArchivedObjectUrl(clip, playbackStartTime);
			return;
		}
		try {
			await archivedPlaybackPipeline.ensurePipeline(clip.mime || DEFAULT_AV_MIME, true);
			clipChunks.forEach((chunk: any) => {
				archivedPlaybackPipeline.enqueueChunk(cloneArchivedChunk(chunk));
			});
			archivedPlaybackPipeline.setStreamCompleted(true);
			archivedPlaybackPipeline.tryEndStream();
			try {
				const el = archivedPlaybackElRef.current;
				if (el && Number.isFinite(playbackStartTime) && playbackStartTime > 0) {
					el.currentTime = playbackStartTime;
				}
			} catch (_) {
				/* noop */
			}
			archivedPlaybackPipeline.maybeStartPlayback();
		} catch (error) {
			console.error("Failed to restore archived playback:", error);
			archivedPlaybackPipeline.reset();
			await playArchivedObjectUrl(clip, playbackStartTime);
		}
	}

	async function finalizeStreamCompletion({ isFallback = false } = {}) {
		avPipeline.setStreamCompleted(true);
		avPipeline.maybeStartPlayback();
		avPipeline.tryEndStream();
		streamStore.patch({
			loadingAnimation: false,
			avPlaybackStarted: false,
			generatingSeedPromptIndex: null,
			lastVideoCompletedAtMs: performance.now(),
		});
		if (shouldUseArchivedPlaybackFallback()) {
			// Native fallback path: must await archive because the blob
			// is the only way to display the video.
			const archivedClip = await archiveCompletedClip();
			if (archivedClip) {
				streamStore.selectClip(archivedClip.id, 0);
				streamStore.patch({
					pendingClip: null,
					avPlaybackStarted: true,
					loadingAnimation: false,
				});
			} else {
				streamStore.patch({
					activeClipId: "",
					activePlaybackStartTime: 0,
					pendingClip: null,
				});
			}
		} else {
			// MSE path (desktop, iOS Safari 17+): archive in background
			// so the message queue isn't blocked by slow remuxing.
			// archiveCompletedClip takes data from the pipeline
			// synchronously before its first await, so it's safe even
			// if markStreamStarting() runs before the remux finishes.
			streamStore.patch({
				activeClipId: "",
				activePlaybackStartTime: 0,
				pendingClip: null,
			});
			void archiveCompletedClip().then(() => {
				void saveCurrentProject({ refreshList: true });
			});
		}
		if (isFallback) {
			console.log("All segments complete (fallback)");
		} else {
			console.log("All segments complete");
		}
		void saveCurrentProject({ refreshList: true });
	}

	// --- Session controls ---

	function handleEnhancementToggle(event: any) {
		sessionStore.patch({
			enhancementEnabled: Boolean(event.currentTarget.checked),
			promptExtensionError: "",
		});
	}

	function handleCuratedPromptLimitChange(event: any) {
		const nextValue = Number.parseInt(event.currentTarget.value, 10);
		if (Number.isNaN(nextValue)) return;
		promptWindowStore.setCuratedPromptLimit(nextValue);
	}

	function setSeedPrompts(nextPrompts: string[]) {
		promptWindowStore.setSeedPrompts(nextPrompts);
	}

	function addEditableSegment() {
		promptWindowStore.addEditableSegment();
	}

	function removeEditableSegment(index: number) {
		promptWindowStore.removeEditableSegment(index);
	}

	function updateEditableSegment(index: number, value: string) {
		promptWindowStore.updateEditableSegment(index, value);
	}

	function handleCustomPresetIdInput(event: any) {
		promptWindowStore.setCustomPresetId(event.currentTarget.value);
	}

	function handleCustomPresetLabelInput(event: any) {
		promptWindowStore.setCustomPresetLabel(event.currentTarget.value);
	}

	function resetEditableToPreset() {
		seedEditableFromPreset(promptWindowStore.get().selectedPreset);
	}

	function exportEditablePreset() {
		const segs = promptWindowStore.get().sanitizedEditableSegments as string[];
		if (segs.length < 2) {
			window.alert("Please provide at least 2 non-empty segments before exporting.");
			return;
		}
		const pw = promptWindowStore.get();
		const label = String(pw.customPresetLabel || "").trim() || DEFAULT_CUSTOM_PRESET_LABEL;
		const id = sanitizePresetId(pw.customPresetId as string);
		const payload = { id, label, segment_prompts: segs };
		const blob = new Blob([JSON.stringify(payload, null, 2)], {
			type: "application/json",
		});
		const objectUrl = URL.createObjectURL(blob);
		const anchor = document.createElement("a");
		anchor.href = objectUrl;
		anchor.download = `${id}.json`;
		document.body.appendChild(anchor);
		anchor.click();
		anchor.remove();
		URL.revokeObjectURL(objectUrl);
	}

	async function appendPromptWindowToPresetsJson() {
		const prompts = (promptWindowStore.get().currentPromptWindowPrompts as string[]).map((p) => (typeof p === "string" ? p.trim() : "")).filter((p) => p.length > 0);
		if (prompts.length < 2) {
			uiStore.patch({
				appendPromptWindowError: "Need at least 2 prompts in the prompt window to append.",
				appendPromptWindowStatus: "",
			});
			return;
		}
		const pw = promptWindowStore.get();
		const id = sanitizePresetId(pw.customPresetId as string);
		const label = String(pw.customPresetLabel || "").trim() || DEFAULT_CUSTOM_PRESET_LABEL;
		uiStore.patch({
			appendingPromptWindow: true,
			appendPromptWindowStatus: "",
			appendPromptWindowError: "",
		});
		try {
			const response = await fetch("/curated-presets/append", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ id, label, segment_prompts: prompts }),
			});
			const payload = await response.json();
			if (!response.ok) {
				throw new Error(payload.detail || "Failed to append prompt window to presets JSON.");
			}
			const nextPreset = payload.preset;
			if (nextPreset?.id && nextPreset?.label && Array.isArray(nextPreset?.segment_prompts)) {
				promptWindowStore.appendStoryPreset(nextPreset);
			}
			uiStore.patch({
				appendPromptWindowStatus: `Appended preset "${label}" to presets JSON.`,
			});
		} catch (error: any) {
			uiStore.patch({
				appendPromptWindowError: error?.message || String(error),
			});
		} finally {
			uiStore.patch({ appendingPromptWindow: false });
		}
	}

	// --- Live prompt ---

	function formatPromptWindowEventText(prompts: any): string {
		if (!Array.isArray(prompts)) return "";
		const normalized = prompts.map((p: any) => (typeof p === "string" ? p.trim() : "")).filter((p: string) => p.length > 0);
		return normalized.map((p: string, i: number) => `[${i + 1}] ${p}`).join("\n");
	}

	function parseLatencyMs(value: any): number | null {
		const numericValue = Number(value);
		return Number.isFinite(numericValue) ? numericValue : null;
	}

	function trackPromptEvent(promptId: string, update: any) {
		rewriteStore.trackPromptEvent(promptId, update);
	}

	function addPromptEvent(event: any) {
		rewriteStore.addPromptEvent(event);
	}

	function captureVideoThumbnail(): string | null {
		const video = showLivePlayback ? videoElRef.current : (archivedPlaybackElRef.current || videoElRef.current);
		if (!video || video.videoWidth === 0 || video.videoHeight === 0) return null;
		try {
			const canvas = document.createElement("canvas");
			canvas.width = 160;
			canvas.height = Math.round(160 * (video.videoHeight / video.videoWidth));
			const ctx = canvas.getContext("2d");
			if (!ctx) return null;
			ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
			return canvas.toDataURL("image/jpeg", 0.7);
		} catch {
			return null;
		}
	}

	function summarizePresetPrompt(prompts: any): string {
		if (!Array.isArray(prompts)) return "";
		return prompts
			.map((p: any) => (typeof p === "string" ? p.trim() : ""))
			.filter((p: string) => p.length > 0)
			.slice(0, 2)
			.join("\n\n");
	}

	function clonePromptWindowPrompts(prompts: any): string[] {
		return normalizePromptWindowSnapshot(prompts);
	}

	function getSelectedEventPromptWindowPrompts(selectedClipId: string): string[] {
		if (!selectedClipId) return [];
		const selectedEvent = (rewriteStore.get().promptEvents as any[]).find(
			(event: any) => event?.clipId === selectedClipId,
		);
		return clonePromptWindowPrompts(selectedEvent?.sourcePromptWindowPrompts);
	}

	function getSelectedClipPromptWindowPrompts(): string[] {
		const selectedClipId = String(streamStore.get().activeClipId || "").trim();
		if (!selectedClipId) return [];
		const selectedClip = (streamStore.get().completedClips as any[]).find((clip: any) => clip?.id === selectedClipId) || null;
		const clipPromptWindowPrompts = clonePromptWindowPrompts(selectedClip?.promptWindowPrompts);
		if (clipPromptWindowPrompts.length > 0) {
			return clipPromptWindowPrompts;
		}
		return getSelectedEventPromptWindowPrompts(selectedClipId);
	}

	function getActivePromptWindowPrompts(): string[] {
		const selectedClipPrompts = getSelectedClipPromptWindowPrompts();
		if (selectedClipPrompts.length > 0) {
			return selectedClipPrompts;
		}
		return clonePromptWindowPrompts(promptWindowStore.get().currentPromptWindowPrompts);
	}

	function getRewriteSourcePromptWindowSnapshot(): string[] {
		const selectedClipPrompts = getSelectedClipPromptWindowPrompts();
		if (selectedClipPrompts.length > 0) {
			return buildRewritePromptWindowSnapshotFromPrompts(selectedClipPrompts);
		}
		return buildRewritePromptWindowSnapshot(promptWindowStore.get());
	}

	function buildPendingStartClip(initialPrompt?: string) {
		const p = initialPrompt ?? pendingInitialPromptRef.current;
		const normalizedInitialPrompt = normalizeInitialPrompt(p);
		return {
			label: getInitialPresetLabel(p) || "Preset story",
			prompt: normalizedInitialPrompt || summarizePresetPrompt(promptWindowStore.get().outboundSessionPrompts),
			source: normalizedInitialPrompt ? "initial_rewrite" : "preset",
			promptWindowPrompts: clonePromptWindowPrompts(getSessionInitPrompts()),
		};
	}

	function buildPendingContinuationClip(prompt: string) {
		const nextIndex = (streamStore.get().completedClips as any[]).length + 1;
		return {
			label: `Cuts ${nextIndex}`,
			prompt: prompt.trim(),
			source: "append_prompt",
			promptWindowPrompts: getActivePromptWindowPrompts(),
		};
	}

	function buildClipLabel({ segmentIdx = null as number | null, source = "" } = {}) {
		const normalizedSource = String(source || "")
			.trim()
			.toLowerCase();
		const sp = promptWindowStore.get().selectedPreset as Record<string, any> | null;
		if (normalizedSource === "curated") {
			if (segmentIdx === 1) return sp?.label || "Preset story";
			if (Number.isInteger(segmentIdx) && (segmentIdx as number) > 1) return `${sp?.label || "Preset story"} ${segmentIdx}`;
			return sp?.label || "Preset story";
		}
		if (Number.isInteger(segmentIdx) && (segmentIdx as number) > 0) {
			return `Cuts ${segmentIdx}`;
		}
		return `Cuts ${(streamStore.get().completedClips as any[]).length + 1}`;
	}

	function submitLivePrompt() {
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		if (sessionStore.get().projectResetPending) return;
		const now = Date.now();
		if (now - lastSubmitTimeRef.current < SUBMIT_COOLDOWN_MS) return;
		const prompt = (sessionStore.get().livePromptDraft as string).trim();
		if (!prompt) return;
		lastSubmitTimeRef.current = now;
		const rCAR = !uiStore.get().devtoolsMode && !uiStore.get().demoMode;
		if (rCAR || (sessionStore.get().livePromptRewriteMode && !uiStore.get().demoMode)) {
			if (rewriteStore.get().rewritingSeedPrompts) return;
			const rewriteSourcePromptWindowPrompts = getActivePromptWindowPrompts();
			const nextPendingClip = {
				...buildPendingContinuationClip(prompt),
				promptWindowPrompts: rewriteSourcePromptWindowPrompts,
			};
			rewriteStore.patch({ rewritingSeedPrompts: true });
			setCurrentThumbnail(null);
			const promptId = makePromptId();
			addPromptEvent({
				promptId,
				status: "rewrite_requested",
				source: "user_rewrite",
				text: prompt,
				thumbnail: captureVideoThumbnail(),
				sourcePromptWindowPrompts: rewriteSourcePromptWindowPrompts,
			});
			ws.send(
				JSON.stringify({
					type: "rewrite_seed_prompts",
					rewrite_instruction: prompt,
					prompt_window_prompts: buildRewritePromptWindowSnapshotFromPrompts(
						rewriteSourcePromptWindowPrompts,
					),
				}),
			);
			streamStore.patch({
				pendingClip: nextPendingClip,
				activeClipId: shouldUseArchivedPlaybackFallback() ? streamStore.get().activeClipId : "",
				activePlaybackStartTime: shouldUseArchivedPlaybackFallback() ? streamStore.get().activePlaybackStartTime : 0,
			});
			sessionStore.patch({ livePromptDraft: "" });
			return;
		}
		const promptId = makePromptId();
		addPromptEvent({
			promptId,
			status: "submitted",
			source: "user_raw",
			text: prompt,
		});
		ws.send(
			JSON.stringify({
				type: "append_prompt",
				prompt_id: promptId,
				prompt,
			}),
		);
		streamStore.patch({
			pendingClip: buildPendingContinuationClip(prompt),
			activeClipId: shouldUseArchivedPlaybackFallback() ? streamStore.get().activeClipId : "",
			activePlaybackStartTime: shouldUseArchivedPlaybackFallback() ? streamStore.get().activePlaybackStartTime : 0,
		});
		sessionStore.patch({ livePromptDraft: "" });
	}

	function setLivePromptRewriteMode(enabled: boolean) {
		const dm = uiStore.get().demoMode;
		if (dm) {
			sessionStore.patch({ livePromptRewriteMode: false });
			return;
		}
		const rCAR = !uiStore.get().devtoolsMode && !dm;
		if (rCAR) {
			sessionStore.patch({ livePromptRewriteMode: true });
			return;
		}
		sessionStore.patch({ livePromptRewriteMode: Boolean(enabled) });
	}

	function handleLivePromptModeToggle(event: any) {
		setLivePromptRewriteMode(Boolean(event.currentTarget.checked));
	}

	const PROMPT_MAX_LENGTH = 500;

	function handleLivePromptInput(event: any) {
		const value = (event.currentTarget.value as string).slice(0, PROMPT_MAX_LENGTH);
		sessionStore.patch({ livePromptDraft: value });
	}

	const lastSubmitTimeRef = useRef(0);
	const SUBMIT_COOLDOWN_MS = 1000;
	const liveInterimLenRef = useRef(0);

	function handleLivePromptSpeechTranscript(text: string) {
		const current = (sessionStore.get().livePromptDraft as string) || "";
		const base = liveInterimLenRef.current > 0 ? current.slice(0, -liveInterimLenRef.current) : current;
		liveInterimLenRef.current = 0;
		const separator = base && !base.endsWith(" ") ? " " : "";
		sessionStore.patch({ livePromptDraft: base + separator + text });
	}

	function handleLivePromptSpeechInterim(text: string) {
		const current = (sessionStore.get().livePromptDraft as string) || "";
		const base = liveInterimLenRef.current > 0 ? current.slice(0, -liveInterimLenRef.current) : current;
		if (!text) {
			liveInterimLenRef.current = 0;
			sessionStore.patch({ livePromptDraft: base });
			return;
		}
		const separator = base && !base.endsWith(" ") ? " " : "";
		const appended = separator + text;
		liveInterimLenRef.current = appended.length;
		sessionStore.patch({ livePromptDraft: base + appended });
	}

	function handleLivePromptKeydown(event: any) {
		if (event.key !== "Enter" || event.isComposing) return;
		if (event.shiftKey) return;
		event.preventDefault();
		submitLivePrompt();
	}

	function handleAutoExtensionToggle(event: any) {
		sessionStore.patch({
			autoExtensionEnabled: Boolean(event.currentTarget.checked),
		});
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		ws.send(
			JSON.stringify({
				type: "set_auto_extension",
				enabled: Boolean(event.currentTarget.checked),
			}),
		);
	}

	function handleLoopGenerationToggle(event: any) {
		sessionStore.patch({
			loopGenerationEnabled: Boolean(event.currentTarget.checked),
		});
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		ws.send(
			JSON.stringify({
				type: "set_loop_generation",
				enabled: Boolean(event.currentTarget.checked),
			}),
		);
	}

	function handlePauseToggle() {
		const nextPaused = !sessionStore.get().generationPaused;
		sessionStore.patch({ generationPaused: nextPaused });
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		ws.send(
			JSON.stringify({
				type: "set_generation_paused",
				paused: nextPaused,
			}),
		);
	}

	function resetToSeedPrompts() {
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		ws.send(JSON.stringify({ type: "reset_to_seed_prompts" }));
	}

	function rewriteSeedPrompts() {
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		if (rewriteStore.get().rewritingSeedPrompts) return;
		rewriteStore.patch({ rewritingSeedPrompts: true });
		ws.send(
			JSON.stringify({
				type: "rewrite_seed_prompts",
				prompt_window_prompts: getRewriteSourcePromptWindowSnapshot(),
			}),
		);
	}

	function restartGenerationAfterCap() {
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		ws.send(JSON.stringify({ type: "restart_generation" }));
	}

	// --- Prompt config editor ---

	async function loadPromptConfig(force = false) {
		const ui = uiStore.get();
		if (!ui.editableMode) return;
		if (ui.promptConfigLoading) return;
		if (ui.promptConfigLoaded && !force) return;
		uiStore.patch({
			promptConfigLoading: true,
			promptConfigError: "",
			promptConfigStatus: "",
		});
		try {
			const response = await fetch("/prompt-system-config");
			const payload = await response.json();
			if (!response.ok) {
				throw new Error(payload.detail || "Failed to load prompt system config.");
			}
			uiStore.patch({
				nextSegmentSystemPromptDraft: payload.next_segment_system_prompt || "",
				autoExtensionSystemPromptDraft: payload.auto_extension_system_prompt || "",
				rewriteWindowSystemPromptDraft: payload.rewrite_window_system_prompt || "",
				promptConfigLoaded: true,
				promptConfigStatus: "Loaded from disk.",
			});
		} catch (error: any) {
			uiStore.patch({
				promptConfigError: error?.message || String(error),
			});
		} finally {
			uiStore.patch({ promptConfigLoading: false });
		}
	}

	function handlePromptConfigEditorToggle(event: any) {
		const isOpen = Boolean(event.currentTarget.open);
		uiStore.patch({ promptConfigEditorOpen: isOpen });
		if (isOpen) loadPromptConfig();
	}

	function handleNextSegmentSystemPromptInput(event: any) {
		uiStore.patch({
			nextSegmentSystemPromptDraft: event.currentTarget.value,
		});
	}

	function handleRewriteWindowSystemPromptInput(event: any) {
		uiStore.patch({
			rewriteWindowSystemPromptDraft: event.currentTarget.value,
		});
	}

	function handleAutoExtensionSystemPromptInput(event: any) {
		uiStore.patch({
			autoExtensionSystemPromptDraft: event.currentTarget.value,
		});
	}

	async function savePromptConfig() {
		const ui = uiStore.get();
		if (!ui.editableMode || ui.promptConfigSaving) return;
		uiStore.patch({
			promptConfigSaving: true,
			promptConfigError: "",
			promptConfigStatus: "",
		});
		try {
			const response = await fetch("/prompt-system-config", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					next_segment_system_prompt: ui.nextSegmentSystemPromptDraft,
					auto_extension_system_prompt: ui.autoExtensionSystemPromptDraft,
					rewrite_window_system_prompt: ui.rewriteWindowSystemPromptDraft,
				}),
			});
			const payload = await response.json();
			if (!response.ok) {
				throw new Error(payload.detail || "Failed to save prompt system config.");
			}
			uiStore.patch({
				nextSegmentSystemPromptDraft: payload.next_segment_system_prompt || "",
				autoExtensionSystemPromptDraft: payload.auto_extension_system_prompt || "",
				rewriteWindowSystemPromptDraft: payload.rewrite_window_system_prompt || "",
				promptConfigLoaded: true,
				promptConfigStatus: "Saved to disk.",
			});
		} catch (error: any) {
			uiStore.patch({
				promptConfigError: error?.message || String(error),
			});
		} finally {
			uiStore.patch({ promptConfigSaving: false });
		}
	}

	// --- Session lifecycle ---

	function resetToLobbyState({ preserveSessionNotice = false, preservePlayback = false } = {}) {
		setVideoMuted(true);
		clearCountdownInterval();
		pendingInitialPromptRef.current = "";
		sessionStore.patch({
			sessionStarted: false,
			connected: false,
			connecting: false,
			gpuAssigned: false,
			timeLeft: null,
			queuePosition: 0,
			sessionTimeout: null,
			livePromptDraft: "",
			autoExtensionTimeoutHint: "",
			generationPaused: false,
			loopGenerationEnabled: false,
			generationCapReached: false,
			generationSegmentCap: 0,
			generatedSegmentCount: 0,
			enhancementEnabled: sessionStore.get().enhancementEnabled,
			promptExtensionError: "",
			preservePlaybackOnClose: false,
			sessionNotice: preserveSessionNotice ? sessionStore.get().sessionNotice : "",
			sessionExpired: preserveSessionNotice ? sessionStore.get().sessionExpired : false,
			projectResetPending: false,
		});
		rewriteStore.resetSessionState();
		streamStore.resetSessionState();
		promptWindowStore.resetSessionState();
		uiStore.resetSessionState();
		resetTtffTimer();
		if (!preservePlayback) resetPlaybackState();
	}

	function resetToProjectLobbyState() {
		setVideoMuted(true);
		pendingInitialPromptRef.current = "";
		sessionStore.patch({
			sessionStarted: false,
			livePromptDraft: "",
			autoExtensionTimeoutHint: "",
			generationPaused: false,
			generationCapReached: false,
			generationSegmentCap: 0,
			generatedSegmentCount: 0,
			promptExtensionError: "",
			preservePlaybackOnClose: false,
			sessionNotice: "",
			sessionExpired: false,
			projectResetPending: false,
		});
		rewriteStore.resetSessionState();
		streamStore.resetSessionState();
		promptWindowStore.resetSessionState();
		uiStore.resetSessionState();
		resetTtffTimer();
		resetPlaybackState();
	}

	function buildProjectInitPayload(type: "session_init_v2" | "project_init_v1") {
		const segmentPrompts = getSessionInitPrompts();
		setSeedPrompts(segmentPrompts);
		return {
			type,
			preset_id: getInitialPresetId(),
			preset_label: getInitialPresetLabel(),
			curated_prompts: segmentPrompts,
			initial_rollout_prompt: normalizeInitialPrompt(pendingInitialPromptRef.current),
			initial_image: null,
			single_clip_mode: false,
			enhancement_enabled: sessionStore.get().enhancementEnabled,
			auto_extension_enabled: sessionStore.get().autoExtensionEnabled,
			loop_generation_enabled: sessionStore.get().loopGenerationEnabled,
		};
	}

	function sendSessionInitMessage() {
		const ws = wsRef.current;
		if (!ws) return;
		ws.send(JSON.stringify(buildProjectInitPayload("session_init_v2")));
	}

	function sendProjectInitMessage() {
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return;
		ws.send(JSON.stringify(buildProjectInitPayload("project_init_v1")));
	}

	function sendEndProjectKeepSession() {
		const ws = wsRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) return false;
		ws.send(JSON.stringify({ type: "end_project_keep_session" }));
		return true;
	}

	async function handleProjectIdle() {
		await saveCurrentProject({ refreshList: true });
		clearSessionArchivedClips();
		currentProjectIdRef.current = "";
		currentProjectCreatedAtRef.current = 0;
		markActiveProjectId(null);
		setCurrentThumbnail(null);
		resetToProjectLobbyState();
	}

	async function handleSocketMessage(event: MessageEvent) {
		const decoded = await decodeWebSocketEvent(event);
		if (decoded.kind === "binary") {
			avPipeline.enqueueChunk(decoded.data);
			return;
		}
		if (decoded.kind !== "json") return;
		if (decoded.data?.type === "error" && isInfrastructureError(decoded.data)) {
			const message = typeof decoded.data?.message === "string" && decoded.data.message.trim()
				? decoded.data.message.trim()
				: "Dreamverse backend is temporarily unavailable.";
			console.warn("[InfrastructureError]", message);
			rewriteStore.patch({ rewritingSeedPrompts: false });
			recoverFailedSessionStart(message);
			return;
		}
		const normalizedEvent = normalizeSocketMessage(decoded.data);
		await applyNormalizedSocketEvent(normalizedEvent, {
			sessionStore,
			promptWindowStore,
			rewriteStore,
			streamStore,
			uiStore,
			avPipeline,
			tick: yieldToEventLoop,
			defaultAvMime: DEFAULT_AV_MIME,
			fixedRewriteModel: FIXED_REWRITE_MODEL,
			parseLatencyMs,
			formatPromptWindowEventText,
			makePromptId,
			buildClipLabel,
			startSessionCountdown,
			clearCountdownInterval,
			resetTtffTimer,
			startTtffTimer,
			preserveArchivedPlaybackSelection: shouldUseArchivedPlaybackFallback(),
			finalizeStreamCompletion,
		});
		if (normalizedEvent.type === "session/project_idle") {
			await handleProjectIdle();
		}
	}

	function connectWebSocket() {
		if (wsRef.current) {
			detachAndCloseWebSocket(wsRef.current);
			wsRef.current = null;
		}
		wsMessageQueueRef.current = Promise.resolve();
		sessionStore.patch({ connecting: true, connected: false });
		try {
			const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
			let opened = false;
			wsRef.current = createWebSocketConnection({
				url: `${wsProtocol}//${window.location.host}/ws`,
				binaryType: "arraybuffer",
				onOpen: () => {
					opened = true;
					sessionStore.patch({ connected: true, connecting: false });
					sendSessionInitMessage();
				},
				onMessage: (event: MessageEvent) => {
					wsMessageQueueRef.current = wsMessageQueueRef.current
						.then(() => handleSocketMessage(event))
						.catch((error) => {
							console.error("Failed to handle websocket message:", error);
						});
				},
				onError: () => {},
				onClose: () => {
					wsRef.current = null;
					const ss = sessionStore.get();
					const hadActiveStream = ss.sessionStarted && avPipeline.hasArchivedChunks();
					rewriteStore.patch({ rewritingSeedPrompts: false });
					if (!opened && ss.sessionStarted && !ss.sessionExpired && !hadActiveStream) {
						if (ss.sessionNotice) {
							recoverFailedSessionStart(ss.sessionNotice);
							return;
						}
						void (async () => {
							const probe = await probeBackendReadiness();
							const fallbackNotice = probe.ok
								? "Dreamverse backend closed the websocket before the session started. Retry after the backend finishes starting."
								: probe.notice;
							recoverFailedSessionStart(fallbackNotice);
						})();
						return;
					}
					if (ss.sessionExpired || hadActiveStream) {
						void (async () => {
							await wsMessageQueueRef.current;
							await finalizeStreamCompletion();
							const clips = streamStore.get().completedClips as any[];
							if (clips.length > 0) {
								streamStore.selectClip(clips[clips.length - 1].id, 0);
							}
							// Reuse the sessionExpired UI path: it keeps
							// the VideoPlayer mounted and shows "Session
							// ended" with a "New Project" button, so the
							// user's generated video stays visible.

							sessionStore.patch({
								connected: false,
								connecting: false,
								gpuAssigned: false,
								sessionExpired: true,
								sessionNotice: "",
							});
							await saveCurrentProject({ refreshList: true });
						})();
					} else {
						resetToLobbyState({
							preserveSessionNotice: Boolean(ss.sessionNotice),
							preservePlayback: Boolean(ss.preservePlaybackOnClose),
						});
					}
				},
			});
		} catch (_) {
			wsRef.current = null;
			const ss = sessionStore.get();
			if (ss.sessionExpired) {
				sessionStore.patch({ connected: false, connecting: false, gpuAssigned: false });
			} else {
				resetToLobbyState({
					preserveSessionNotice: Boolean(ss.sessionNotice),
					preservePlayback: Boolean(ss.preservePlaybackOnClose),
				});
			}
		}
	}

	function beginProjectLocally({ force = false } = {}) {
		if (!force && !canStartSession) return;
		if (sessionStore.get().sessionStarted || sessionStore.get().projectResetPending) return false;
		setTimeoutModalOpen(false);
		// Unmute during the user gesture so iOS Safari permits audio playback.
		setVideoMuted(false);
		if (viewingProject) closeViewingProject();
		clearSessionArchivedClips();
		currentProjectIdRef.current = makePromptId();
		currentProjectCreatedAtRef.current = Date.now();
		markActiveProjectId(currentProjectIdRef.current);
		setCurrentThumbnail(null);
		const initialPrompt = normalizeInitialPrompt(sessionStore.get().livePromptDraft as string);
		pendingInitialPromptRef.current = initialPrompt;
		const rCAR = !uiStore.get().devtoolsMode && !uiStore.get().demoMode;
		sessionStore.patch({
			sessionNotice: "",
			sessionExpired: false,
			promptExtensionError: "",
			generationCapReached: false,
			generationSegmentCap: 0,
			generatedSegmentCount: 0,
			preservePlaybackOnClose: false,
			sessionStarted: true,
			livePromptDraft: "",
			livePromptRewriteMode: rCAR || Boolean(initialPrompt),
			autoExtensionTimeoutHint: "",
			generationPaused: false,
			projectResetPending: false,
		});
		resetPlaybackState();
		streamStore.patch({
			loadingAnimation: true,
			playingSeedPromptIndex: null,
			generatingSeedPromptIndex: null,
			seedPromptIndexBySegment: {},
			currentSegmentNumber: 0,
			promptHistory: [],
			promptHistoryCounter: 0,
			selectedHistoryId: "",
		});
		rewriteStore.resetSessionState();
		if (initialPrompt) {
			addPromptEvent({
				promptId: makePromptId(),
				status: "rewrite_requested",
				source: "user_rewrite",
				text: initialPrompt,
			});
		}
		setSeedPrompts(getSessionInitPrompts());
		const nextPendingClip = buildPendingStartClip(initialPrompt);
		streamStore.patch({
			pendingClip: nextPendingClip,
			liveClip: nextPendingClip,
			activeClipId: "",
			activePlaybackStartTime: 0,
		});
		return true;
	}

	async function joinSession({ force = false } = {}) {
		if (
			wsRef.current
			&& wsRef.current.readyState === WebSocket.OPEN
			&& sessionStore.get().connected
		) {
			if (!beginProjectLocally({ force })) return;
			sendProjectInitMessage();
			return;
		}
		showPreSessionNotice("");
		streamStore.patch({ loadingAnimation: true });
		sessionStore.patch({ connecting: true });
		const probe = await probeBackendReadiness();
		if (!probe.ok) {
			streamStore.patch({ loadingAnimation: false });
			showPreSessionNotice(probe.notice);
			return;
		}
		if (!beginProjectLocally({ force })) {
			streamStore.patch({ loadingAnimation: false });
			sessionStore.patch({ connecting: false });
			return;
		}
		if (
			wsRef.current
			&& wsRef.current.readyState === WebSocket.OPEN
			&& sessionStore.get().connected
		) {
			sendProjectInitMessage();
			return;
		}
		connectWebSocket();
	}

	async function saveCurrentProject({ refreshList = false } = {}) {
		if (!currentProjectIdRef.current) return;
		const pw = promptWindowStore.get();
		const project: StoredProject = {
			id: currentProjectIdRef.current,
			label: currentClipLabel,
			presetId: (pw.selectedPresetId as string) || "",
			originalLabel: pendingInitialPromptRef.current || ((pw.selectedPreset as Record<string, any>)?.label as string) || "",
			createdAt: currentProjectCreatedAtRef.current || Date.now(),
			lastThumbnail: currentThumbnail,
			promptEvents: [...(rewriteStore.get().promptEvents as Record<string, unknown>[])],
		};
		const clips: StoredClip[] = (streamStore.get().completedClips as any[])
			.filter((clip: any) => clip?.blob instanceof Blob)
			.map((clip: any) => ({
				id: clip.id,
				projectId: project.id,
				label: clip.label || "",
				prompt: clip.prompt || "",
				mime: clip.mime || "",
				blob: clip.blob,
				createdAt: clip.createdAt || Date.now(),
			}));
		const persistSnapshot = async () => {
			try {
				await saveProjectWithRecovery(project, clips, refreshList);
			} catch (error) {
				console.error("Failed to save project:", error);
			}
		};
		const queuedSave = saveQueueRef.current.then(persistSnapshot, persistSnapshot);
		saveQueueRef.current = queuedSave.catch(() => {});
		await queuedSave;
	}

	function markActiveProjectId(id: string | null) {
		try {
			if (id) {
				localStorage.setItem("fastvideo-active-project", id);
			} else {
				localStorage.removeItem("fastvideo-active-project");
			}
		} catch (_) {
			/* private browsing */
		}
	}

	function isLikelyProjectStoragePressure(error: unknown): boolean {
		if (error instanceof DOMException) {
			return [
				"QuotaExceededError",
				"AbortError",
				"UnknownError",
				"InvalidStateError",
			].includes(error.name);
		}

		const message = String(
			(error as { message?: unknown })?.message || error || "",
		).toLowerCase();
		return (
			message.includes("quota")
			|| message.includes("storage")
			|| message.includes("space")
			|| message.includes("indexeddb")
			|| message.includes("backing store")
			|| message.includes("transaction failed")
			|| message.includes("transaction was aborted")
		);
	}

	async function pruneProjectsForStorageRecovery(currentProjectId: string, retainPreviousCount: number): Promise<number> {
		const projects = await listProjects();
		let keptPrevious = 0;
		let deletedCount = 0;

		for (const project of projects) {
			if (project.id === currentProjectId) {
				continue;
			}
			if (keptPrevious < retainPreviousCount) {
				keptPrevious += 1;
				continue;
			}
			await deleteProject(project.id);
			deletedCount += 1;
		}

		return deletedCount;
	}

	async function saveProjectWithRecovery(
		project: StoredProject,
		clips: StoredClip[],
		refreshList: boolean,
	): Promise<void> {
		try {
			await saveProject(project, clips);
		} catch (error) {
			if (!isLikelyProjectStoragePressure(error)) {
				throw error;
			}

			let hasExistingSnapshot = false;
			try {
				const existingProjects = await listProjects();
				hasExistingSnapshot = existingProjects.some(
					(existingProject) => existingProject.id === project.id,
				);
			} catch (snapshotError) {
				if (!isLikelyProjectStoragePressure(snapshotError)) {
					throw snapshotError;
				}
				console.warn(
					"Unable to inspect existing project snapshots before retrying save.",
					{
						projectId: project.id,
						error: snapshotError,
					},
				);
			}
			let lastError: unknown = error;
			for (const retainPreviousCount of STORAGE_RECOVERY_PREVIOUS_PROJECT_COUNTS) {
				let deletedCount = 0;
				try {
					deletedCount = await pruneProjectsForStorageRecovery(
						project.id,
						retainPreviousCount,
					);
				} catch (pruneError) {
					lastError = pruneError;
					if (!isLikelyProjectStoragePressure(pruneError)) {
						throw pruneError;
					}
					continue;
				}
				if (deletedCount === 0 && retainPreviousCount !== 0) {
					continue;
				}

				try {
					await saveProject(project, clips);
					console.warn(
						"Recovered project save after pruning older archives.",
						{
							projectId: project.id,
							deletedCount,
							retainPreviousCount,
						},
					);
					lastError = null;
					break;
				} catch (retryError) {
					lastError = retryError;
					if (!isLikelyProjectStoragePressure(retryError)) {
						throw retryError;
					}
				}
			}

			if (lastError && !hasExistingSnapshot && clips.length > 1) {
				try {
					await saveProject(project, clips.slice(-1));
					console.warn(
						"Recovered project save with only the latest archived clip.",
						{
							projectId: project.id,
							savedClipCount: 1,
							originalClipCount: clips.length,
						},
					);
					lastError = null;
				} catch (retryError) {
					lastError = retryError;
					if (!isLikelyProjectStoragePressure(retryError)) {
						throw retryError;
					}
				}
			}

			if (lastError) {
				try {
					await saveProjectMetadata(project);
					console.warn(
						hasExistingSnapshot
							? "Recovered project save by preserving the existing archive and updating metadata only."
							: "Recovered project save by storing project metadata without archived clips.",
						{
							projectId: project.id,
							originalClipCount: clips.length,
							hadExistingSnapshot: hasExistingSnapshot,
						},
					);
					lastError = null;
				} catch (metadataError) {
					lastError = metadataError;
					if (!isLikelyProjectStoragePressure(metadataError)) {
						throw metadataError;
					}
				}
			}

			if (lastError) {
				throw lastError;
			}
		}

		try {
			await pruneOldProjects(MAX_ARCHIVED_PROJECTS);
		} catch (pruneError) {
			console.warn("Failed to prune older archived projects after save.", {
				projectId: project.id,
				error: pruneError,
			});
		}
		if (refreshList) {
			await refreshSavedProjects();
		}
	}

	async function refreshSavedProjects() {
		try {
			const projects = await listProjects();
			setSavedProjects(projects);
		} catch (error) {
			console.error("Failed to load saved projects:", error);
		}
	}

	async function handleDeleteProject(projectId: string) {
		try {
			await deleteProject(projectId);
			await refreshSavedProjects();
			if (viewingProject?.project.id === projectId) {
				closeViewingProject();
			}
		} catch (error) {
			console.error("Failed to delete project:", error);
		}
	}

	async function viewSavedProject(projectId: string) {
		try {
			const clips = await loadProjectClips(projectId);
			const project = savedProjects.find((p) => p.id === projectId);
			if (!project) return;
			if (viewingProject) {
				closeViewingProject();
			}
			const hydratedClips = clips.map((clip) => ({
				id: clip.id,
				label: clip.label,
				prompt: clip.prompt,
				mime: clip.mime,
				objectUrl: URL.createObjectURL(clip.blob),
				blob: clip.blob,
				createdAt: clip.createdAt,
			}));
			setViewingProject({ project, clips: hydratedClips });
			setViewingSelectedClipId("");
			setSidebarOpen(false);
		} catch (error) {
			console.error("Failed to load project:", error);
		}
	}

	function closeViewingProject() {
		if (viewingProject) {
			viewingProject.clips.forEach((clip) => URL.revokeObjectURL(clip.objectUrl));
		}
		setViewingProject(null);
	}

	async function leaveSession() {
		setTimeoutModalOpen(false);
		if (wsRef.current) {
			detachAndCloseWebSocket(wsRef.current);
			wsRef.current = null;
		}
		await finalizeStreamCompletion();
		await saveCurrentProject({ refreshList: true });
		clearSessionArchivedClips();
		currentProjectIdRef.current = "";
		currentProjectCreatedAtRef.current = 0;
		markActiveProjectId(null);
		resetToLobbyState();
	}

	async function handleStartNewProject() {
		setTimeoutModalOpen(false);
		setSidebarOpen(false);
		if (viewingProject) {
			closeViewingProject();
		}
		const ss = sessionStore.get();
		if (ss.projectResetPending) {
			return;
		}

		if (
			wsRef.current
			&& wsRef.current.readyState === WebSocket.OPEN
			&& ss.connected
			&& ss.sessionStarted
		) {
			sessionStore.patch({
				projectResetPending: true,
				sessionNotice: "",
			});
			sendEndProjectKeepSession();
			return;
		}

		clearSessionArchivedClips();
		currentProjectIdRef.current = "";
		currentProjectCreatedAtRef.current = 0;
		markActiveProjectId(null);
		setCurrentThumbnail(null);
		sessionStore.patch({
			sessionNotice: "",
			sessionExpired: false,
			projectResetPending: false,
		});

		if (
			wsRef.current
			&& wsRef.current.readyState === WebSocket.OPEN
			&& ss.connected
		) {
			resetToProjectLobbyState();
			return;
		}

		resetToLobbyState();
	}

	// --- Formatting helpers ---

	function formatTime(seconds: number): string {
		const mins = Math.floor(seconds / 60);
		const secs = seconds % 60;
		return `${mins}:${secs.toString().padStart(2, "0")}`;
	}

	function formatDurationMs(durationMs: number): string {
		return `${(durationMs / 1000).toFixed(3)}s`;
	}

	async function triggerBlobDownload(blob: Blob, label: string) {
		const ext = blob.type.includes("webm") ? "webm" : "mp4";
		const firstPrompt = (currentClipPrompt as string || "")
			.replace(/[^a-zA-Z0-9 _-]/g, "")
			.trim()
			.replace(/\s+/g, "_")
			.substring(0, 60);
		const safeName = firstPrompt || "video";
		const fileName = `${safeName}.${ext}`;
		const mimeType = blob.type || `video/${ext}`;

		// Use Web Share API on touch-first devices (phones/tablets) for camera-roll access.
		// "pointer: coarse" excludes desktops where canShare exists but a share dialog is unwanted.
		if (typeof navigator.canShare === "function" && window.matchMedia("(pointer: coarse)").matches) {
			try {
				const file = new File([blob], fileName, { type: mimeType });
				if (navigator.canShare({ files: [file] })) {
					await navigator.share({ files: [file] });
					return;
				}
			} catch (err: any) {
				if (err?.name === "AbortError") return;
			}
		}

		const url = URL.createObjectURL(blob);
		const anchor = document.createElement("a");
		anchor.href = url;
		anchor.download = fileName;
		document.body.appendChild(anchor);
		anchor.click();
		anchor.remove();
		URL.revokeObjectURL(url);
	}

	function getRemuxableSegmentsFromClip(clip: Record<string, any> | null): ArchivedSegmentLike[] {
		if (!clip) return [];
		const archivedSegments = normalizeArchivedSegments(clip.archivedSegments);
		if (archivedSegments.length > 0) return archivedSegments;
		return [];
	}

	async function handleDownloadVideo() {
		if (downloadInFlightRef.current) return;
		downloadInFlightRef.current = true;
		try {
			const liveSegments = normalizeArchivedSegments(typeof avPipeline.buildArchivedSegmentSnapshots === "function" ? avPipeline.buildArchivedSegmentSnapshots({ includeInProgress: true }) : []);
			if (liveSegments.length > 0) {
				const remuxedBlob = await remuxArchivedSegmentsBestEffort(liveSegments, currentClipLabel);
				if (remuxedBlob instanceof Blob && remuxedBlob.size > 0) {
					await triggerBlobDownload(remuxedBlob, currentClipLabel);
				}
				return;
			}

			const currentActiveClip = streamStore.get().activeClip as Record<string, any> | null;
			const activeClipSegments = getRemuxableSegmentsFromClip(currentActiveClip);
			if (activeClipSegments.length > 0) {
				const remuxedBlob = await remuxArchivedSegmentsBestEffort(activeClipSegments, currentActiveClip?.label || currentClipLabel);
				if (remuxedBlob instanceof Blob && remuxedBlob.size > 0) {
					await triggerBlobDownload(remuxedBlob, currentActiveClip?.label || currentClipLabel);
					return;
				}
				return;
			}
			if (currentActiveClip?.blob instanceof Blob) {
				await triggerBlobDownload(currentActiveClip.blob, currentActiveClip.label || currentClipLabel);
				return;
			}

			const clips = streamStore.get().completedClips as Record<string, any>[];
			const latestCompletedClip = clips[clips.length - 1] || null;
			const latestClipSegments = getRemuxableSegmentsFromClip(latestCompletedClip);
			if (latestClipSegments.length > 0) {
				const remuxedBlob = await remuxArchivedSegmentsBestEffort(latestClipSegments, latestCompletedClip?.label || currentClipLabel);
				if (remuxedBlob instanceof Blob && remuxedBlob.size > 0) {
					await triggerBlobDownload(remuxedBlob, latestCompletedClip?.label || currentClipLabel);
					return;
				}
				return;
			}
			if (latestCompletedClip?.blob instanceof Blob) {
				await triggerBlobDownload(latestCompletedClip.blob, latestCompletedClip.label || currentClipLabel);
			}
		} finally {
			downloadInFlightRef.current = false;
		}
	}

	async function handleDownloadViewingVideo() {
		const lastClip = viewingProject?.clips[viewingProject.clips.length - 1];
		if (!lastClip?.blob) return;
		await triggerBlobDownload(lastClip.blob, viewingProject?.project.label || "video");
	}

	// --- Render ---

	if (!runtimeReady) {
		return null;
	}

	if (isMonitorRoute) {
		return <MonitorPage />;
	}

	if (devtoolsMode) {
		return (
			<DevtoolsShell
				connected={connected as boolean}
				gpuAssigned={gpuAssigned as boolean}
				sessionStarted={sessionStarted as boolean}
				queuePosition={queuePosition as number}
				connecting={connecting as boolean}
				storyPresets={storyPresets as any[]}
				selectedPresetId={selectedPresetId as string}
				enhancementEnabled={enhancementEnabled as boolean}
				autoExtensionEnabled={autoExtensionEnabled as boolean}
				loopGenerationEnabled={loopGenerationEnabled as boolean}
				canJoinSession={canJoinSession as boolean}
				canSubmitContinuation={canSubmitContinuation}
				editableMode={editableMode as boolean}
				demoMode={demoMode as boolean}
				editableCanJoin={editableCanJoin as boolean}
				curatedPromptLimit={curatedPromptLimit as number}
				maxCuratedPromptCount={maxCuratedPromptCount as number}
				onPresetChange={handlePresetSelectionChange}
				onEnhancementToggle={handleEnhancementToggle}
				onCuratedPromptLimitChange={handleCuratedPromptLimitChange}
				onAutoExtensionToggle={handleAutoExtensionToggle}
				onLoopToggle={handleLoopGenerationToggle}
				onJoin={joinSession}
				onLeave={leaveSession}
				sessionNotice={sessionNotice as string}
				generationCapReached={generationCapReached as boolean}
				generationSegmentCap={generationSegmentCap as number}
				onRestartGeneration={restartGenerationAfterCap}
				selectedPreset={selectedPreset}
				editableSegments={editableSegments as string[]}
				customPresetId={customPresetId as string}
				customPresetLabel={customPresetLabel as string}
				currentPromptWindowPrompts={currentPromptWindowPrompts as string[]}
				appendingPromptWindow={appendingPromptWindow as boolean}
				appendPromptWindowStatus={appendPromptWindowStatus as string}
				appendPromptWindowError={appendPromptWindowError as string}
				onCustomPresetIdInput={handleCustomPresetIdInput}
				onCustomPresetLabelInput={handleCustomPresetLabelInput}
				onAddSegment={addEditableSegment}
				onResetToPreset={resetEditableToPreset}
				onExport={exportEditablePreset}
				onAppendPromptWindowToJson={appendPromptWindowToPresetsJson}
				onRemoveSegment={removeEditableSegment}
				onUpdateSegment={updateEditableSegment}
				seedPrompts={seedPrompts as string[]}
				playingSeedPromptIndex={playingSeedPromptIndex}
				generatingSeedPromptIndex={generatingSeedPromptIndex}
				livePromptDraft={livePromptDraft as string}
				livePromptRewriteMode={livePromptRewriteMode as boolean}
				rewritingSeedPrompts={rewritingSeedPrompts as boolean}
				promptEvents={promptEvents as any[]}
				onLivePromptInput={handleLivePromptInput}
				onLivePromptModeToggle={handleLivePromptModeToggle}
				onLivePromptKeydown={handleLivePromptKeydown}
				onSubmitLivePrompt={submitLivePrompt}
				onSpeechTranscript={handleLivePromptSpeechTranscript}
				onSpeechInterimChange={handleLivePromptSpeechInterim}
				promptConfigEditorOpen={promptConfigEditorOpen as boolean}
				promptConfigLoading={promptConfigLoading as boolean}
				promptConfigSaving={promptConfigSaving as boolean}
				promptConfigStatus={promptConfigStatus as string}
				promptConfigError={promptConfigError as string}
				nextSegmentPromptEditorOpen={nextSegmentPromptEditorOpen as boolean}
				autoExtensionPromptEditorOpen={autoExtensionPromptEditorOpen as boolean}
				rewriteWindowPromptEditorOpen={rewriteWindowPromptEditorOpen as boolean}
				nextSegmentSystemPromptDraft={nextSegmentSystemPromptDraft as string}
				autoExtensionSystemPromptDraft={autoExtensionSystemPromptDraft as string}
				rewriteWindowSystemPromptDraft={rewriteWindowSystemPromptDraft as string}
				onPromptConfigEditorToggle={handlePromptConfigEditorToggle}
				onReloadPromptConfig={() => loadPromptConfig(true)}
				onNextSegmentSystemPromptInput={handleNextSegmentSystemPromptInput}
				onAutoExtensionSystemPromptInput={handleAutoExtensionSystemPromptInput}
				onRewriteWindowSystemPromptInput={handleRewriteWindowSystemPromptInput}
				onSavePromptConfig={savePromptConfig}
				autoExtensionTimeoutHint={autoExtensionTimeoutHint as string}
				activeClip={
					activeClip
						? {
								...(activeClip as Record<string, any>),
								playbackStartTime: activePlaybackStartTime,
							}
						: null
				}
				liveClipLabel={currentClipLabel}
				liveClipPrompt={currentClipPrompt}
				galleryClips={completedClips as any[]}
				activeClipId={activeClipId as string}
				showLivePlayback={showLivePlayback}
				onSelectClip={selectClip}
				avPlaybackStarted={avPlaybackStarted as boolean}
				mediaAppendError={mediaAppendError}
				timeLeft={timeLeft}
				ttffStartAtMs={ttffStartAtMsRef.current}
				ttffValueMs={ttffValueMs}
				timeBetweenVideosMs={timeBetweenVideosMs}
				loadingAnimation={loadingAnimation as boolean}
				formatTime={formatTime}
				formatDurationMs={formatDurationMs}
				onPlaying={markFirstFrameRendered}
			/>
		);
	}

	const viewingEvents = viewingProject?.project.promptEvents ?? [];
	const viewingHasEdits = viewingEvents.some((e: any) => typeof e?.text === "string" && e.text.trim() && String(e?.source || "").trim() === "user_rewrite");
	const viewingLastClip = viewingProject?.clips[viewingProject.clips.length - 1] ?? null;
	const viewingFirstClip = viewingProject?.clips[0] ?? null;
	const showActiveProject = (sessionStarted as boolean) || (sessionExpired as boolean);
	const headerTimeLeft = (
		(connected as boolean)
		|| (connecting as boolean)
		|| (sessionStarted as boolean)
		|| (sessionExpired as boolean)
	)
		? timeLeft
		: null;

	return (
		<main className="flex h-dvh w-full flex-col overflow-hidden bg-background text-foreground">
			<SessionTimeoutModal
				open={timeoutModalOpen && !isViewingMode}
				onClose={() => setTimeoutModalOpen(false)}
				onStartNewProject={handleStartNewProject}
				repoUrl={FASTVIDEO_REPO_URL}
				blogUrl={FASTVIDEO_BLOG_URL}
			/>
			<Sidebar
				open={sidebarOpen}
				currentProjectId={currentProjectIdRef.current}
				currentProjectLabel={currentProjectTitle}
				sessionActive={Boolean(sessionStarted) && !sessionExpired}
				sessionExpired={sessionExpired as boolean}
				projectResetPending={projectResetPending as boolean}
				savedProjects={savedProjects}
				viewingProjectId={viewingProject?.project.id ?? null}
				isViewingPastProject={isViewingMode}
				onClose={() => setSidebarOpen(false)}
				onSelectProject={viewSavedProject}
				onSelectCurrentProject={() => {
					closeViewingProject();
					setSidebarOpen(false);
				}}
				onDeleteProject={handleDeleteProject}
				onNewProject={() => {
					setSidebarOpen(false);
					handleStartNewProject();
				}}
			/>
			<Header timeLeft={headerTimeLeft} formatTime={formatTime} onToggleSidebar={() => setSidebarOpen((prev) => !prev)} />

			<div className="relative flex flex-1 min-h-0 flex-col justify-center px-4 pb-2 sm:px-6 sm:pb-12">
				{isViewingMode && (
					<>
						{viewingSelectedClip && (
							<div className="relative z-30 w-full shrink-0">
								<div className="relative mx-auto aspect-video w-full max-w-3xl overflow-hidden rounded-2xl border border-border bg-black">
									<video key={viewingSelectedClip.id} src={viewingSelectedClip.objectUrl} className="h-full w-full object-contain" autoPlay loop playsInline controls />
									<Button
										onClick={handleDownloadViewingVideo}
										size="icon"
										variant="outline"
										className="absolute top-3 right-3 z-10 bg-slate-900/60 backdrop-blur-md text-white hover:bg-slate-800/85 border-white/25"
									>
										{isMobileShareCapable ? <Share2 className="size-5" /> : <Download className="size-5" />}
									</Button>
								</div>
							</div>
						)}
						<section className={cn("mx-auto w-full max-w-2xl", viewingHasEdits && "flex-1 min-h-0 overflow-y-auto")}>
								<Workspace
									promptEvents={viewingEvents as any[]}
									currentThumbnail={viewingProject?.project.lastThumbnail ?? null}
									originalLabel={viewingProject?.project.originalLabel || ""}
									sessionStarted={true}
									originalClipId={viewingFirstClip?.id || ""}
									selectedClipId={viewingSelectedClip?.id || ""}
									onSelectOriginal={() => {
										if (viewingFirstClip) setViewingSelectedClipId(viewingFirstClip.id);
								}}
								onSelectEvent={(event) => {
									const clip = viewingProject?.clips.find((c) => c.id === event.clipId);
									if (clip) setViewingSelectedClipId(clip.id);
								}}
								onSelectCurrent={() => {
									if (viewingLastClip) setViewingSelectedClipId(viewingLastClip.id);
								}}
							/>
						</section>
						<motion.div layout="position" className="mx-auto w-full max-w-2xl shrink-0" transition={{ type: "spring", stiffness: 200, damping: 25 }}>
							<ChatBar sessionStarted={false} viewingReadOnly={true} onStartNewProject={handleStartNewProject} onBackFromViewing={closeViewingProject} />
						</motion.div>
					</>
				)}

				{/* Keep active session mounted (hidden when viewing past project) so the video pipeline stays alive */}
				<div className={isViewingMode ? "hidden" : "contents"}>
					<AnimatePresence>
						{showActiveProject && (
							<motion.div
								key="video-player"
								layout="position"
								initial={{ opacity: 0, scale: 0.98 }}
								animate={{ opacity: 1, scale: 1 }}
								exit={{ opacity: 0, scale: 0.98 }}
								transition={{
									layout: { type: "spring", stiffness: 200, damping: 25 },
									opacity: { duration: 0.4, ease: "easeOut" },
									scale: { duration: 0.4, ease: "easeOut" },
								}}
								className="relative z-30 w-full shrink-0"
							>
								<VideoPlayer
									videoRef={videoRefCallback}
									archivedPlaybackRef={archivedPlaybackRefCallback}
									activeClip={
										activeClip
											? {
													...(activeClip as Record<string, any>),
													playbackStartTime: activePlaybackStartTime,
												}
											: null
									}
									sessionStarted={sessionStarted as boolean}
									avPlaybackStarted={avPlaybackStarted as boolean}
									mediaAppendError={mediaAppendError}
									timeLeft={timeLeft}
									gpuAssigned={gpuAssigned as boolean}
									connected={connected as boolean}
									queuePosition={queuePosition as number}
									loadingAnimation={loadingAnimation as boolean}
									showLivePlayback={showLivePlayback}
									defaultMuted={videoMuted}
									canDownload={canDownloadVideo}
									onPlaying={markFirstFrameRendered}
									onDownload={handleDownloadVideo}
								/>
							</motion.div>
						)}
					</AnimatePresence>

					<section className={cn("mx-auto w-full max-w-2xl", hasEdits && "flex-1 min-h-0 overflow-y-auto")}>
						<Workspace
							promptEvents={promptEvents as any[]}
							currentThumbnail={currentThumbnail}
							originalLabel={pendingInitialPromptRef.current || (selectedPreset as Record<string, any>)?.label || ""}
							sessionStarted={sessionStarted as boolean}
							originalClipId={((completedClips as any[])[0])?.id || ""}
							selectedClipId={(activeClipId as string) || ""}
							onSelectOriginal={() => {
								const firstClip = (completedClips as any[])[0];
								if (firstClip) selectClip(firstClip.id, 0);
							}}
							onSelectEvent={(event) => {
								if (event.clipId) selectClip(event.clipId, 0);
							}}
							onSelectCurrent={() => {
								const clips = completedClips as any[];
								if (sessionStarted && !sessionExpired) {
									// Return to live playback
									streamStore.patch({ activeClipId: "", activePlaybackStartTime: 0 });
								} else if (clips.length > 0) {
									selectClip(clips[clips.length - 1].id, 0);
								}
							}}
						/>
					</section>

					<AnimatePresence>
						{!showActiveProject && (
							<motion.div
								key="hero-tagline"
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								exit={{ opacity: 0, transition: { duration: 0.2, ease: "easeIn" } }}
								transition={{ duration: 0.5, ease: "easeOut" }}
								className="pointer-events-none absolute inset-x-0 top-0 bottom-1/2 z-10 flex items-center justify-center px-4"
							>
								<HeroTagline />
							</motion.div>
						)}
					</AnimatePresence>

					<motion.div layout="position" className="mx-auto w-full max-w-2xl shrink-0" transition={{ type: "spring", stiffness: 200, damping: 25 }}>
						<ChatBar
							sessionStarted={sessionStarted as boolean}
							rewritingSeedPrompts={rewritingSeedPrompts as boolean}
							isGenerating={loadingAnimation as boolean}
							storyPresets={storyPresets as any[]}
							continuationDraft={livePromptDraft as string}
							canJoinSession={canStartSession}
							canSubmitContinuation={canSubmitContinuation}
							sessionExpired={sessionExpired as boolean}
							sessionNotice={sessionNotice as string}
							projectResetPending={projectResetPending as boolean}
							onPresetGenerate={handlePresetGenerate}
							onContinuationInput={handleLivePromptInput}
							onContinuationKeydown={handleLivePromptKeydown}
							onGenerate={joinSession}
							onSubmitContinuation={submitLivePrompt}
							onLeave={leaveSession}
							onStartNewProject={handleStartNewProject}
							onSpeechTranscript={handleLivePromptSpeechTranscript}
							onSpeechInterimChange={handleLivePromptSpeechInterim}
						/>
					</motion.div>
				</div>
			</div>
		</main>
	);
}
