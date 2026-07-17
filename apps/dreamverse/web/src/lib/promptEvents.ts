export const MAX_PROMPT_EVENTS = 24;

export function updatePromptEvent(
  events: Record<string, any>[],
  promptId: string,
  update: Record<string, any>,
): Record<string, any>[] {
  return events.map((event) => (
    event.promptId === promptId
      ? { ...event, ...update }
      : event
  ));
}

export function prependPromptEvent(
  events: Record<string, any>[],
  event: Record<string, any>,
): Record<string, any>[] {
  const next = [event, ...events];
  if (next.length <= MAX_PROMPT_EVENTS) {
    return next;
  }
  // Steering scene events (steeringUserPrompt) are exempt from the cap: the
  // scene list is derived from them and must stay complete and stably numbered
  // for long sessions. Only the oldest non-scene events are dropped.
  let nonSceneKept = 0;
  return next.filter((e) => {
    if (e?.steeringUserPrompt) {
      return true;
    }
    nonSceneKept += 1;
    return nonSceneKept <= MAX_PROMPT_EVENTS;
  });
}
