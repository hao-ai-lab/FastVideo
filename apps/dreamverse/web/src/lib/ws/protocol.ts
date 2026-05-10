const EVENT_TYPE_MAP: Record<string, string> = {
  queue_status: 'session/queue_status',
  prompt_received: 'prompt/received',
  prompt_enhancing: 'prompt/enhancing',
  prompt_ready: 'prompt/ready',
  prompt_fallback_used: 'prompt/fallback_used',
  segment_prompt_source: 'prompt/source_selected',
  seed_prompts_updated: 'prompt_window/updated',
  rewrite_seed_prompts_started: 'rewrite/started',
  rewrite_seed_prompts_complete: 'rewrite/completed',
  loop_generation_updated: 'session/loop_generation_updated',
  generation_paused_updated: 'session/generation_paused_updated',
  loop_restarted: 'session/loop_restarted',
  seed_prompts_reset_applied: 'prompt_window/reset_applied',
  auto_prompt_failed: 'prompt/auto_failed',
  prompt_sources_blocked: 'prompt/sources_blocked',
  prompt_sources_resumed: 'prompt/sources_resumed',
  auto_extension_updated: 'session/auto_extension_updated',
  step_complete: 'segment/step_complete',
  gpu_assigned: 'session/gpu_assigned',
  session_timeout: 'session/timeout',
  project_idle: 'session/project_idle',
  generation_cap_reached: 'session/generation_cap_reached',
  generation_restarted: 'session/generation_restarted',
  ltx2_stream_start: 'stream/started',
  media_init: 'stream/media_init',
  media_segment_complete: 'stream/media_segment_complete',
  ltx2_segment_start: 'segment/started',
  ltx2_segment_complete: 'segment/completed',
  ltx2_stream_complete: 'stream/completed',
  error: 'session/error',
};

export interface NormalizedSocketMessage {
  type: string;
  rawType: string;
  payload: any;
}

export function normalizeSocketMessage(
  data: any,
): NormalizedSocketMessage {
  const rawType =
    typeof data?.type === 'string' ? data.type : 'unknown';

  return {
    type: EVENT_TYPE_MAP[rawType] || 'server/unhandled',
    rawType,
    payload: data,
  };
}
