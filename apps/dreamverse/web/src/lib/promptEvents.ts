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
  return [event, ...events].slice(0, MAX_PROMPT_EVENTS);
}
