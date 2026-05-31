function buildPromptExtensionFailureMessage(): string {
	return "Prompt extension failed for this request. " + "Turn off Prompt extension and try again to use your raw prompt directly.";
}

function resolveSessionErrorMessage(payload: any): string {
	if (payload?.error_code === "ip_session_limit") {
		return "Only one active websocket session is allowed per IP. " + "Close the other session and click Run to retry.";
	}

	if (typeof payload?.message === "string" && payload.message.trim()) {
		return payload.message.trim();
	}

	return "An error occurred. Please start a new project.";
}

function isInfrastructureError(payload: any): boolean {
	const msg = typeof payload?.message === "string" ? payload.message.toLowerCase() : "";
	return msg.includes("replica unavailable");
}

export async function applyNormalizedSocketEvent(event: any, context: any): Promise<void> {
	const {
		sessionStore,
		promptWindowStore,
		rewriteStore,
		streamStore,
		uiStore,
		avPipeline,
		tick,
		defaultAvMime,
		fixedRewriteModel,
		parseLatencyMs,
		formatPromptWindowEventText,
		makePromptId,
		buildClipLabel,
		startSessionCountdown,
		clearCountdownInterval,
		resetTtffTimer,
		startTtffTimer,
		preserveArchivedPlaybackSelection,
		finalizeStreamCompletion,
	} = context;

	const payload = event?.payload;
	sessionStore.applyServerUiMessage(payload);

	switch (event?.type) {
		case "session/queue_status":
			console.log(`Queue position: ${sessionStore.get().queuePosition}`);
			return;

		case "prompt/received":
			rewriteStore.trackPromptEvent(payload.prompt_id, {
				status: "queued",
			});
			return;

		case "prompt/enhancing":
			rewriteStore.trackPromptEvent(payload.prompt_id, {
				status: "enhancing",
			});
			return;

		case "prompt/ready":
			if (uiStore.get().simpleMode) {
				sessionStore.patch({
					promptExtensionError: "",
				});
			}
			rewriteStore.trackPromptEvent(payload.prompt_id, {
				status: "ready",
				source: payload.source || "user_raw",
				text: payload.prompt,
			});
			return;

		case "prompt/fallback_used":
			if (uiStore.get().simpleMode) {
				const fallbackClip = streamStore.get().completedSimpleClips[0] || null;
				const promptExtensionError = buildPromptExtensionFailureMessage();
				avPipeline.reset();
				sessionStore.patch({
					promptExtensionError,
				});
				streamStore.patch({
					loadingAnimation: false,
					avPlaybackStarted: false,
					mediaAppendError: null,
					generatingSeedPromptIndex: null,
					simpleCompletionFinalized: true,
					simpleLiveCardVisible: true,
					activeSimpleArchivedClipId: fallbackClip?.id || "",
					activeSimplePlaybackObjectUrl: fallbackClip?.objectUrl || "",
					activeSimplePlaybackStartTime: 0,
				});
			}
			rewriteStore.trackPromptEvent(payload.prompt_id, {
				status: "ready_fallback",
				source: payload.source || "user_raw",
				text: payload.prompt,
			});
			console.warn("[PromptEnhanceFallback] Prompt extension failed for this request.");
			return;

		case "prompt/source_selected":
			rewriteStore.applySegmentPromptSource(payload);
			if (payload.prompt_id) {
				rewriteStore.trackPromptEvent(payload.prompt_id, {
					status: "consumed",
					source: payload.source || "user_raw",
				});
			}
			return;

		case "prompt_window/updated": {
			const nextPrompts = Array.isArray(payload.prompts) ? payload.prompts : [];
			streamStore.patch((state: any) => ({
				pendingClip: state.pendingClip
					? {
							...state.pendingClip,
							promptWindowPrompts: [...nextPrompts],
						}
					: state.pendingClip,
				liveClip:
					!state.pendingClip && state.liveClip
						? {
								...state.liveClip,
								promptWindowPrompts: [...nextPrompts],
							}
						: state.liveClip,
			}));
			if (payload.reason === "rewrite") {
				promptWindowStore.replacePromptWindow(nextPrompts, { syncEditable: true });
				const latencyMs = parseLatencyMs(payload.latency_ms);
				const rewriteModel = typeof payload.model === "string" && payload.model.trim() ? payload.model.trim() : fixedRewriteModel;
				const rawLlmOutput = typeof payload.raw_llm_output === "string" ? payload.raw_llm_output.trim() : "";
				const rewriteOutput = formatPromptWindowEventText(nextPrompts);

				if (rewriteOutput) {
					rewriteStore.addPromptEvent({
						promptId: makePromptId(),
						status: payload.fallback_used ? "rewrite_fallback" : "rewrite_ready",
						source: "llm_rewrite",
						model: rewriteModel,
						latencyMs,
						text: rewriteOutput,
					});
					if (rawLlmOutput) {
						rewriteStore.addPromptEvent({
							promptId: makePromptId(),
							status: "rewrite_raw_output",
							source: "llm_rewrite",
							model: rewriteModel,
							latencyMs,
							text: rawLlmOutput,
						});
					}
				} else if (payload.error) {
					rewriteStore.addPromptEvent({
						promptId: makePromptId(),
						status: "rewrite_error",
						source: "llm_rewrite",
						model: rewriteModel,
						latencyMs,
						text: payload.error,
					});
					if (rawLlmOutput) {
						rewriteStore.addPromptEvent({
							promptId: makePromptId(),
							status: "rewrite_raw_output",
							source: "llm_rewrite",
							model: rewriteModel,
							latencyMs,
							text: rawLlmOutput,
						});
					}
				}
				rewriteStore.patch({
					rewritingSeedPrompts: false,
				});
				return;
			}

			promptWindowStore.setSeedPrompts(nextPrompts);
			return;
		}

		case "rewrite/started":
			rewriteStore.patch({ rewritingSeedPrompts: true });
			return;

		case "rewrite/completed":
			rewriteStore.patch({ rewritingSeedPrompts: false });
			if (payload?.error) {
				const latencyMs = parseLatencyMs(payload.latency_ms);
				rewriteStore.addPromptEvent({
					promptId: makePromptId(),
					status: "rewrite_error",
					source: "llm_rewrite",
					model: typeof payload.model === "string" && payload.model.trim() ? payload.model.trim() : fixedRewriteModel,
					latencyMs,
					text: payload.error,
				});
			}
			return;

		case "session/loop_generation_updated":
			sessionStore.patch({
				loopGenerationEnabled: Boolean(payload.enabled),
			});
			return;

		case "session/generation_paused_updated":
			sessionStore.patch({
				generationPaused: Boolean(payload.paused),
			});
			return;

		case "session/loop_restarted":
		case "prompt_window/reset_applied":
			streamStore.patch({
				currentSegmentNumber: 0,
				playingSeedPromptIndex: null,
				generatingSeedPromptIndex: null,
				seedPromptIndexBySegment: {},
			});
			return;

		case "prompt/auto_failed": {
			if (uiStore.get().simpleMode) {
				sessionStore.patch({ autoExtensionTimeoutHint: "" });
				return;
			}

			console.error("[AutoPromptFailed]", payload);
			const segmentLabel = Number.isInteger(payload?.segment_idx) ? `segment ${payload.segment_idx}` : "next segment";
			const errorText = typeof payload?.error === "string" && payload.error.trim() ? payload.error.trim() : "Auto prompt timed out.";
			sessionStore.patch({
				autoExtensionTimeoutHint: `${errorText} Retry ${segmentLabel} by toggling Auto extension off, then on.`,
			});
			return;
		}

		case "prompt/sources_blocked":
			sessionStore.patch({
				autoExtensionTimeoutHint: uiStore.get().simpleMode ? "" : "blocked on user input, increase prompt count for smoother experience",
			});
			return;

		case "prompt/sources_resumed":
		case "session/auto_extension_updated":
			sessionStore.patch({ autoExtensionTimeoutHint: "" });
			if (event.type === "session/auto_extension_updated") {
				console.log("[AutoExtensionUpdated]", {
					enabled: sessionStore.get().autoExtensionEnabled,
				});
			}
			return;

		case "segment/step_complete":
			streamStore.patch({
				lastGenerationLatencyMs: parseLatencyMs(payload?.latency_ms?.worker_e2e),
				lastE2eLatencyMs: parseLatencyMs(payload?.latency_ms?.total),
			});
			return;

		case "session/gpu_assigned":
			console.log(`GPU ${payload.gpu_id} assigned, session timeout: ${sessionStore.get().sessionTimeout}s`);
			startSessionCountdown();
			return;

		case "session/timeout":
			sessionStore.patch({
				preservePlaybackOnClose: false,
				generationCapReached: false,
				projectResetPending: false,
				sessionExpired: true,
				sessionNotice: "",
			});
			console.log("Session timed out");
			clearCountdownInterval();
			resetTtffTimer();
			return;

		case "session/project_idle":
			return;

		case "session/generation_cap_reached": {
			if (uiStore.get().simpleMode) {
				return;
			}
			sessionStore.patch({
				preservePlaybackOnClose: false,
				generationCapReached: false,
				sessionNotice: "",
			});
			await finalizeStreamCompletion();
			resetTtffTimer();
			return;
		}

		case "session/generation_restarted": {
			const capSegments = Number.parseInt(payload?.segment_cap, 10);
			sessionStore.patch({
				generatedSegmentCount: 0,
				generationSegmentCap: uiStore.get().simpleMode ? 0 : Number.isFinite(capSegments) && capSegments > 0 ? capSegments : sessionStore.get().generationSegmentCap,
				generationCapReached: false,
				sessionNotice: "",
			});
			return;
		}

		case "stream/started":
			sessionStore.patch({
				generatedSegmentCount: 0,
				generationSegmentCap: uiStore.get().simpleMode
					? 0
					: Number.parseInt(payload?.generation_segment_cap, 10) > 0
						? Number.parseInt(payload.generation_segment_cap, 10)
						: sessionStore.get().generationSegmentCap,
				generationCapReached: false,
				sessionNotice: "",
				promptExtensionError: "",
				autoExtensionTimeoutHint: "",
			});
			streamStore.patch({
				mediaAppendError: null,
				loadingAnimation: true,
				currentSegmentNumber: 0,
				playingSeedPromptIndex: null,
				generatingSeedPromptIndex: null,
				lastGenerationLatencyMs: null,
				lastE2eLatencyMs: null,
				seedPromptIndexBySegment: {},
				simpleCompletionFinalized: false,
				activeSimplePlaybackObjectUrl: uiStore.get().simpleMode ? "" : streamStore.get().activeSimplePlaybackObjectUrl,
				activeSimplePlaybackStartTime: uiStore.get().simpleMode ? 0 : streamStore.get().activeSimplePlaybackStartTime,
				liveClip: uiStore.get().simpleMode ? streamStore.get().liveClip : streamStore.get().pendingClip || streamStore.get().liveClip,
				pendingClip: uiStore.get().simpleMode ? streamStore.get().pendingClip : null,
				activeClipId: uiStore.get().simpleMode || preserveArchivedPlaybackSelection ? streamStore.get().activeClipId : "",
				activePlaybackStartTime: uiStore.get().simpleMode || preserveArchivedPlaybackSelection ? streamStore.get().activePlaybackStartTime : 0,
				simpleLiveCardVisible: uiStore.get().simpleMode ? true : streamStore.get().simpleLiveCardVisible,
			});
			avPipeline.reset();
			avPipeline.setStreamCompleted(false);
			if (typeof payload.loop_generation_enabled === "boolean") {
				sessionStore.patch({
					loopGenerationEnabled: payload.loop_generation_enabled,
				});
			}
			startTtffTimer();
			console.log(`Stream starting: ${payload.total_segments} segments`);
			return;

		case "stream/media_init":
			streamStore.patch({
				mediaAppendError: null,
				loadingAnimation: streamStore.get().avPlaybackStarted ? streamStore.get().loadingAnimation : true,
				activeClipId: uiStore.get().simpleMode || preserveArchivedPlaybackSelection ? streamStore.get().activeClipId : "",
				activePlaybackStartTime: uiStore.get().simpleMode || preserveArchivedPlaybackSelection ? streamStore.get().activePlaybackStartTime : 0,
			});
			avPipeline.noteSegmentInit({
				segmentIdx: Number.isInteger(payload?.segment_idx) ? payload.segment_idx : null,
				streamId: typeof payload?.stream_id === "string" ? payload.stream_id : "",
				mime: typeof payload?.mime === "string" ? payload.mime : "",
			});

			try {
				await tick();
				await avPipeline.ensurePipeline(payload.mime || defaultAvMime);
			} catch (error) {
				streamStore.patch({
					mediaAppendError: "Unable to initialize AV streaming.",
					avPlaybackStarted: false,
				});
				console.error("media_init failed:", error);
				avPipeline.reset();
			}
			return;

		case "stream/media_segment_complete": {
			avPipeline.noteSegmentComplete({
				segmentIdx: Number.isInteger(payload?.segment_idx) ? payload.segment_idx : null,
				streamId: typeof payload?.stream_id === "string" ? payload.stream_id : "",
			});
			avPipeline.maybeStartPlayback();

			const currentSession = sessionStore.get();
			if (!currentSession.generationCapReached) {
				const nextGeneratedSegmentCount = currentSession.generatedSegmentCount + 1;
				const reachedCap = !uiStore.get().simpleMode && currentSession.generationSegmentCap > 0 && nextGeneratedSegmentCount >= currentSession.generationSegmentCap;
				sessionStore.patch({
					generatedSegmentCount: nextGeneratedSegmentCount,
					generationCapReached: reachedCap || currentSession.generationCapReached,
				});
				if (reachedCap) {
					resetTtffTimer();
				}
			}

			const completedSegment = Number.isInteger(payload?.segment_idx) ? payload.segment_idx : null;
			if (completedSegment !== null) {
				const completedSeedIndex = streamStore.get().seedPromptIndexBySegment[completedSegment];
				if (Number.isInteger(completedSeedIndex)) {
					streamStore.patch({
						playingSeedPromptIndex: completedSeedIndex,
						generatingSeedPromptIndex: streamStore.get().generatingSeedPromptIndex === completedSeedIndex ? null : streamStore.get().generatingSeedPromptIndex,
					});
				}
			}
			return;
		}

		case "segment/started": {
			sessionStore.patch({ autoExtensionTimeoutHint: "" });
			const pendingSegmentSource = rewriteStore.get().pendingSegmentSource;
			const seedIndex = Number.isInteger(payload.seed_prompt_index) ? payload.seed_prompt_index : (pendingSegmentSource?.seedPromptIndex ?? null);

			const nextSeedPromptIndexBySegment = Number.isInteger(seedIndex)
				? {
						...streamStore.get().seedPromptIndexBySegment,
						[payload.segment_idx]: seedIndex,
					}
				: streamStore.get().seedPromptIndexBySegment;

			streamStore.patch({
				currentSegmentNumber: payload.segment_idx,
				generatingSeedPromptIndex: Number.isInteger(seedIndex) ? seedIndex : null,
				playingSeedPromptIndex: streamStore.get().playingSeedPromptIndex === null && Number.isInteger(seedIndex) ? seedIndex : streamStore.get().playingSeedPromptIndex,
				seedPromptIndexBySegment: nextSeedPromptIndexBySegment,
			});

			if (!uiStore.get().simpleMode) {
				const fallbackClip = {
					label: buildClipLabel({
						segmentIdx: Number.isInteger(payload.segment_idx) ? payload.segment_idx : null,
						source: payload.source || pendingSegmentSource?.source || rewriteStore.get().lastPromptSource || "",
					}),
					prompt: typeof payload.prompt === "string" ? payload.prompt.trim() : streamStore.get().liveClip?.prompt || "",
					source: payload.source || pendingSegmentSource?.source || rewriteStore.get().lastPromptSource || "",
					segmentIdx: Number.isInteger(payload.segment_idx) ? payload.segment_idx : null,
				};
				streamStore.patch({
					liveClip: streamStore.get().liveClip || fallbackClip,
				});
			}

			streamStore.pushPromptHistory({
				segmentIdx: payload.segment_idx,
				source: payload.source || pendingSegmentSource?.source || rewriteStore.get().lastPromptSource || "unknown",
				prompt: payload.prompt || "",
				seedPromptIndex: seedIndex,
				loopIteration: Number.isInteger(payload.loop_iteration) ? payload.loop_iteration : (pendingSegmentSource?.loopIteration ?? null),
			});
			rewriteStore.patch({
				pendingSegmentSource: null,
			});
			console.log(`Segment ${payload.segment_idx}/${payload.total_segments}: ${payload.prompt?.substring(0, 60)}...`);
			return;
		}

		case "segment/completed":
			console.log(`Segment ${payload.segment_idx}/${payload.total_segments} complete`);
			if (uiStore.get().simpleMode && Number.isInteger(payload?.segment_idx) && Number.isInteger(payload?.total_segments) && payload.segment_idx >= payload.total_segments) {
				await finalizeStreamCompletion({ isFallback: true });
			}
			return;

		case "stream/completed":
			sessionStore.patch({
				generationCapReached: false,
				sessionNotice: "",
			});
			await finalizeStreamCompletion();
			return;

		case "session/error": {
			const errorMessage = resolveSessionErrorMessage(payload);
			sessionStore.patch({
				generationCapReached: false,
				preservePlaybackOnClose: false,
				promptExtensionError: "",
				sessionNotice: errorMessage,
			});
			rewriteStore.patch({
				rewritingSeedPrompts: false,
			});
			console.error("[StreamingError]", payload.message || payload);
			return;
		}

		case "server/unhandled":
		default:
			return;
	}
}

export { resolveSessionErrorMessage, isInfrastructureError };
