/**
 * Work out what "queue this scene" would do, so the user can confirm it first.
 *
 * Clips are queued in scene order. The server runs them one after another and
 * holds a clip that starts from another clip's last frame until that clip has
 * finished -- so a whole scene can be queued at once and each clip still opens
 * on the frame the one before it ended on. This decides which clips are queued,
 * and (optionally) which get that link attached first.
 */
import { attachedLastFrames, whyCannotAttach } from "@/lib/continuation";
import type { SceneClip } from "@/lib/scenes";

/** Statuses a job can be queued from (the same ones it can be started or edited in). */
export const QUEUEABLE_STATUSES = ["pending", "failed", "stopped"];

export interface SceneQueuePlan {
	/** clips to queue, in order */
	queue: SceneClip[];
	/** clips that will get the previous clip's last frame attached first */
	chain: { clip: SceneClip; previous: SceneClip }[];
	/** clips that would need the link but can't have it attached automatically */
	cannotChain: { clip: SceneClip; reason: string }[];
	/** clips left alone, and why */
	skipped: { clip: SceneClip; reason: string }[];
}

function skipReason(status: string): string {
	if (status === "completed") return "already finished";
	if (status === "running") return "already running";
	if (status === "queued") return "already queued";
	return `is ${status}`;
}

export function planSceneQueue(clips: SceneClip[], { chain: wantChain }: { chain: boolean }): SceneQueuePlan {
	const plan: SceneQueuePlan = { queue: [], chain: [], cannotChain: [], skipped: [] };
	clips.forEach((clip, i) => {
		if (!QUEUEABLE_STATUSES.includes(clip.job.status)) {
			plan.skipped.push({ clip, reason: skipReason(clip.job.status) });
			return;
		}
		plan.queue.push(clip);

		const previous = clips[i - 1];
		if (!wantChain || !previous) return;
		// A clip that already opens on a frame (of this or another take) keeps it.
		if (attachedLastFrames(clip.job.references).length > 0) return;
		const blocked = whyCannotAttach(clip.job.prompt, clip.job.references);
		if (blocked) plan.cannotChain.push({ clip, reason: blocked });
		else plan.chain.push({ clip, previous });
	});
	return plan;
}
