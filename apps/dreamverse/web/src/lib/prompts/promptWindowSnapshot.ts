const REWRITE_SEGMENT_COUNT = 6;
const CONTINUATION_FILL =
  'The scene continues naturally, maintaining the current mood and pacing.';

export function normalizePromptWindowSnapshot(
  prompts: any,
): string[] {
  if (!Array.isArray(prompts)) {
    return [];
  }

  return prompts
    .map((prompt: any) => (typeof prompt === 'string' ? prompt.trim() : ''))
    .filter((prompt: string) => prompt.length > 0);
}

export function buildPromptWindowSnapshot(
  promptWindowState: any,
): string[] {
  return normalizePromptWindowSnapshot(promptWindowState?.currentPromptWindowPrompts);
}

export function buildRewritePromptWindowSnapshotFromPrompts(
  prompts: any,
): string[] {
  const snapshot = normalizePromptWindowSnapshot(prompts);
  while (snapshot.length > 0 && snapshot.length < REWRITE_SEGMENT_COUNT) {
    snapshot.push(CONTINUATION_FILL);
  }
  return snapshot;
}

/**
 * Pad snapshot to the segment count the server-side rewrite system
 * prompt expects, so the LLM response length matches validation.
 */
export function buildRewritePromptWindowSnapshot(
  promptWindowState: any,
): string[] {
  return buildRewritePromptWindowSnapshotFromPrompts(
    buildPromptWindowSnapshot(promptWindowState),
  );
}
