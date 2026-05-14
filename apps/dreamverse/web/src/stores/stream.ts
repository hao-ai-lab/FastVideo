import { createManagedStore, type ManagedStore } from "./createManagedStore";

export interface StreamState {
	playingSeedPromptIndex: number | null;
	generatingSeedPromptIndex: number | null;
	seedPromptIndexBySegment: Record<string, unknown>;
	currentSegmentNumber: number;
	promptHistory: Record<string, unknown>[];
	promptHistoryCounter: number;
	selectedHistoryId: string;
	selectedHistoryEntry: Record<string, unknown> | null;
	completedSimpleClips: Record<string, unknown>[];
	activeSimpleClip: Record<string, unknown> | null;
	simpleLiveCardVisible: boolean;
	simpleCompletionFinalized: boolean;
	activeSimpleArchivedClipId: string;
	activeSimplePlaybackObjectUrl: string;
	activeSimplePlaybackStartTime: number;
	completedClips: Record<string, unknown>[];
	activeClipId: string;
	activeClip: Record<string, unknown> | null;
	activePlaybackStartTime: number;
	pendingClip: Record<string, unknown> | null;
	liveClip: Record<string, unknown> | null;
	loadingAnimation: boolean;
	avPlaybackStarted: boolean;
	mediaAppendError: string | null;
	lastVideoCompletedAtMs: number | null;
	timeBetweenVideosMs: number | null;
	lastGenerationLatencyMs: number | null;
	lastE2eLatencyMs: number | null;
}

function deriveStreamState(state: StreamState): StreamState {
	const promptHistory = Array.isArray(state.promptHistory) ? state.promptHistory : [];
	const completedSimpleClips = Array.isArray(state.completedSimpleClips) ? state.completedSimpleClips : [];
	const completedClips = Array.isArray(state.completedClips) ? state.completedClips : [];

	return {
		...state,
		promptHistory,
		completedSimpleClips,
		completedClips,
		selectedHistoryEntry: promptHistory.find((item) => item.id === state.selectedHistoryId) || null,
		activeClip: completedClips.find((clip) => clip.id === state.activeClipId) || null,
	} as unknown as StreamState;
}

const DEFAULT_STREAM_STATE: StreamState = {
	playingSeedPromptIndex: null,
	generatingSeedPromptIndex: null,
	seedPromptIndexBySegment: {},
	currentSegmentNumber: 0,
	promptHistory: [],
	promptHistoryCounter: 0,
	selectedHistoryId: "",
	selectedHistoryEntry: null,
	completedSimpleClips: [],
	activeSimpleClip: null,
	simpleLiveCardVisible: false,
	simpleCompletionFinalized: false,
	activeSimpleArchivedClipId: "",
	activeSimplePlaybackObjectUrl: "",
	activeSimplePlaybackStartTime: 0,
	completedClips: [],
	activeClipId: "",
	activeClip: null,
	activePlaybackStartTime: 0,
	pendingClip: null,
	liveClip: null,
	loadingAnimation: false,
	avPlaybackStarted: false,
	mediaAppendError: null,
	lastVideoCompletedAtMs: null,
	timeBetweenVideosMs: null,
	lastGenerationLatencyMs: null,
	lastE2eLatencyMs: null,
};

export interface PushPromptHistoryArgs {
	segmentIdx?: number;
	source?: string;
	prompt?: string;
	seedPromptIndex?: number | null;
	loopIteration?: number | null;
}

export type StreamStore = ManagedStore<StreamState> & {
	reset: (overrides?: Partial<StreamState>) => StreamState;
	resetSessionState: () => StreamState;
	pushPromptHistory: (args: PushPromptHistoryArgs) => StreamState;
	selectPromptHistory: (selectedHistoryId: string) => StreamState;
	addCompletedSimpleClip: (clip: Record<string, unknown>) => StreamState;
	addCompletedClip: (clip: Record<string, unknown>) => StreamState;
	selectClip: (clipId: string, playbackStartTime?: number) => StreamState;
};

export function createStreamStore(initialState: Partial<StreamState> = {}): StreamStore {
	const store = createManagedStore<StreamState>(
		{
			...DEFAULT_STREAM_STATE,
			...initialState,
		} as StreamState,
		deriveStreamState,
	);

	return {
		...store,
		reset(overrides: Partial<StreamState> = {}) {
			return store.set({
				...DEFAULT_STREAM_STATE,
				...initialState,
				...overrides,
			} as StreamState);
		},
		resetSessionState() {
			return store.patch({
				playingSeedPromptIndex: null,
				generatingSeedPromptIndex: null,
				seedPromptIndexBySegment: {},
				currentSegmentNumber: 0,
				promptHistory: [],
				promptHistoryCounter: 0,
				selectedHistoryId: "",
				activeSimpleClip: null,
				simpleLiveCardVisible: false,
				simpleCompletionFinalized: false,
				activeSimpleArchivedClipId: "",
				activeSimplePlaybackObjectUrl: "",
				activeSimplePlaybackStartTime: 0,
				activeClipId: "",
				activePlaybackStartTime: 0,
				pendingClip: null,
				liveClip: null,
				loadingAnimation: false,
				avPlaybackStarted: false,
				mediaAppendError: null,
				lastVideoCompletedAtMs: null,
				timeBetweenVideosMs: null,
				lastGenerationLatencyMs: null,
				lastE2eLatencyMs: null,
			});
		},
		pushPromptHistory({ segmentIdx, source, prompt, seedPromptIndex = null, loopIteration = null }: PushPromptHistoryArgs) {
			const text = typeof prompt === "string" ? prompt.trim() : "";
			if (!text) {
				return store.get();
			}

			return store.update((state) => {
				const nextCounter = state.promptHistoryCounter + 1;
				const entry: Record<string, unknown> = {
					id: `h_${nextCounter}`,
					order: nextCounter,
					segmentIdx: typeof segmentIdx === "number" ? segmentIdx : null,
					source: source || "unknown",
					prompt: text,
					seedPromptIndex: typeof seedPromptIndex === "number" ? seedPromptIndex : null,
					loopIteration: typeof loopIteration === "number" ? loopIteration : null,
				};

				return {
					...state,
					promptHistoryCounter: nextCounter,
					promptHistory: [entry, ...state.promptHistory].slice(0, 120),
					selectedHistoryId: state.selectedHistoryId || (entry.id as string),
				};
			});
		},
		selectPromptHistory(selectedHistoryId: string) {
			return store.patch({ selectedHistoryId });
		},
		addCompletedSimpleClip(clip: Record<string, unknown>) {
			return store.update((state) => ({
				...state,
				completedSimpleClips: [clip, ...state.completedSimpleClips],
			}));
		},
		addCompletedClip(clip: Record<string, unknown>) {
			return store.update((state) => ({
				...state,
				completedClips: [...state.completedClips, clip],
			}));
		},
		selectClip(clipId: string, playbackStartTime: number = 0) {
			return store.patch({
				activeClipId: clipId,
				activePlaybackStartTime: Number.isFinite(playbackStartTime) ? Math.max(playbackStartTime, 0) : 0,
			});
		},
	};
}

export { DEFAULT_STREAM_STATE };
