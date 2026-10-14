import type { Job } from "@/lib/types";

export function formatDuration(seconds: number): string {
	const roundedSeconds = Math.round(seconds);
	if (roundedSeconds < 60) return `${roundedSeconds}s`;
	if (roundedSeconds < 3600) {
		const mins = Math.floor(roundedSeconds / 60);
		const secs = roundedSeconds % 60;
		return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
	}
	const hours = Math.floor(roundedSeconds / 3600);
	const mins = Math.floor((roundedSeconds % 3600) / 60);
	return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
}

function toMs(time: number): number {
	return time < 1e12 ? time * 1000 : time;
}

/**
 * How long a job has run, formatted: live while running (the server sets
 * `started_at` when it leaves the queue, so queue time isn't counted), then
 * frozen at `finished_at`. Null for a job that hasn't started.
 */
export function computeElapsed(job: Job, currentTime: number): string | null {
	if (!job.started_at) return null;
	const endTime = job.status === "running" ? currentTime : (job.finished_at ?? 0);
	if (!endTime) return null;
	const elapsedSeconds = (toMs(endTime) - toMs(job.started_at)) / 1000;
	if (elapsedSeconds <= 0) return null;
	return formatDuration(elapsedSeconds);
}
