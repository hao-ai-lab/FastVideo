/**
 * Group jobs into scenes and lay their shots out on one timeline.
 *
 * A long scene is generated as many short clips, one job each. Nothing on the
 * server links them, so the link is the job name: `wolf-lunch-clip-01`,
 * `wolf-lunch-clip-02`, ... share the stem `wolf-lunch` and are ordered by the
 * trailing number. Everything else -- shots, dialogue, subjects -- is read back
 * out of each job's own prompt, so no backend support is needed.
 */
import { readField, shotTarget } from "@/lib/h3PromptFields";
import {
	parseDetailedDescription,
	parseSubjectLabels,
	resolveShotStarts,
	shotProse,
	type Shot,
} from "@/lib/h3Shots";
import type { Job } from "@/lib/types";

export const UNNAMED_SCENE = "__unnamed__";
const DEFAULT_FPS = 24;

// "<stem>[-_. ][clip|part|pt|shot|take][-_. ]<number>". "scene" is deliberately not
// a clip word: it is far more likely part of the scene's own name ("rooftop-scene-2").
const CLIP_SUFFIX_RE = /^(.*?)[\s._-]*(?:clip|part|pt|shot|take)?[\s._-]*(\d+)$/i;

export interface SceneName {
	key: string;
	title: string;
	/** the clip number from the end of the name, or null when there isn't one */
	order: number | null;
}

export function parseSceneName(name: string | undefined | null): SceneName {
	const trimmed = (name ?? "").trim();
	if (!trimmed) return { key: UNNAMED_SCENE, title: "Unnamed jobs", order: null };
	const m = trimmed.match(CLIP_SUFFIX_RE);
	const stem = m ? m[1].replace(/[\s._-]+$/, "") : "";
	if (m && stem) return { key: stem.toLowerCase(), title: stem, order: Number(m[2]) };
	return { key: trimmed.toLowerCase(), title: trimmed, order: null };
}

/** The shots a job's prompt describes, wherever they live in it (base, reference, or plain prompt). */
export function shotsOfJob(job: Pick<Job, "prompt">): Shot[] {
	const prompt = job.prompt ?? "";
	const labels = parseSubjectLabels(readField(prompt, "subject_definitions") ?? "");
	return parseDetailedDescription(shotTarget(prompt).body, labels).shots;
}

export interface SceneClip {
	/** the take shown for this clip */
	job: Job;
	/**
	 * Every job that claims this clip number, oldest first. Re-running a clip
	 * gives the same name again, so these are takes of one clip, not extra clips.
	 */
	takes: Job[];
	/** identifies this clip for choosing a take; null for a clip with no number */
	takeKey: string | null;
	/** 1-based place in the scene */
	position: number;
	shots: Shot[];
	/** seconds from the start of the scene to the start of this clip */
	startSeconds: number;
	durationSeconds: number;
	/** each shot's cut-in time measured from the start of the scene */
	shotStarts: number[];
	/** this clip picks up in the middle of the previous clip's last shot */
	continuation: boolean;
}

export interface Scene {
	key: string;
	title: string;
	clips: SceneClip[];
	totalSeconds: number;
	shotCount: number;
	statusCounts: Record<string, number>;
	/** most recent created_at among the clips, for ordering scenes */
	latest: number;
}

export function clipSeconds(job: Pick<Job, "num_frames" | "fps">): number {
	const fps = job.fps && job.fps > 0 ? job.fps : DEFAULT_FPS;
	return Math.max(0, (job.num_frames ?? 0) / fps);
}

function isContinuation(job: Job, shots: Shot[]): boolean {
	if (/continues the previous shot/i.test(job.prompt ?? "")) return true;
	return shots.length > 0 && /^Continuing the same shot/i.test(shotProse(shots[0]));
}

function createdAt(job: Job): number {
	return job.created_at ?? 0;
}

/** The take to show: the one the user chose, else the most recent completed one, else the newest. */
function pickTake(takes: Job[], choice?: string): Job {
	const chosen = choice ? takes.find((j) => j.id === choice) : undefined;
	if (chosen) return chosen;
	const recent = (a: Job, b: Job) => (b.finished_at ?? createdAt(b)) - (a.finished_at ?? createdAt(a));
	const completed = takes.filter((j) => j.status === "completed").sort(recent);
	return completed[0] ?? [...takes].sort((a, b) => createdAt(b) - createdAt(a))[0];
}

/**
 * Group inference jobs into scenes, each with its clips in order and a
 * scene-wide timeline. `choices` maps a clip's `takeKey` to the job id of the
 * take to show in place of the default.
 */
export function buildScenes(jobs: Job[], choices: Record<string, string> = {}): Scene[] {
	const groups = new Map<string, { title: string; entries: { job: Job; order: number | null }[] }>();
	for (const job of jobs) {
		const { key, title, order } = parseSceneName(job.name);
		const group = groups.get(key) ?? { title, entries: [] };
		group.entries.push({ job, order });
		groups.set(key, group);
	}

	const scenes: Scene[] = [];
	for (const [key, { title, entries }] of groups) {
		// One clip per clip number (its takes together); each unnumbered job stands alone, after the numbered ones.
		const byNumber = new Map<number, Job[]>();
		const unnumbered: Job[][] = [];
		for (const { job, order } of entries) {
			if (order === null) unnumbered.push([job]);
			else byNumber.set(order, [...(byNumber.get(order) ?? []), job]);
		}
		const slots: { takes: Job[]; takeKey: string | null }[] = [
			...[...byNumber.entries()].sort(([a], [b]) => a - b).map(([order, takes]) => ({ takes, takeKey: `${key}#${order}` })),
			...unnumbered.sort((a, b) => createdAt(a[0]) - createdAt(b[0])).map((takes) => ({ takes, takeKey: null })),
		];

		let elapsed = 0;
		const statusCounts: Record<string, number> = {};
		const clips: SceneClip[] = slots.map(({ takes, takeKey }, i) => {
			const ordered = [...takes].sort((a, b) => createdAt(a) - createdAt(b));
			const job = pickTake(ordered, takeKey ? choices[takeKey] : undefined);
			const shots = shotsOfJob(job);
			const durationSeconds = clipSeconds(job);
			const clip: SceneClip = {
				job,
				takes: ordered,
				takeKey,
				position: i + 1,
				shots,
				startSeconds: elapsed,
				durationSeconds,
				shotStarts: resolveShotStarts(shots).map((t) => elapsed + t),
				continuation: isContinuation(job, shots),
			};
			elapsed += durationSeconds;
			statusCounts[job.status] = (statusCounts[job.status] ?? 0) + 1;
			return clip;
		});

		scenes.push({
			key,
			title,
			clips,
			totalSeconds: elapsed,
			shotCount: clips.reduce((n, c) => n + c.shots.length, 0),
			statusCounts,
			latest: Math.max(0, ...entries.map((e) => createdAt(e.job))),
		});
	}

	// Newest scene first; the catch-all group of unnamed jobs goes last.
	return scenes.sort((a, b) => {
		if ((a.key === UNNAMED_SCENE) !== (b.key === UNNAMED_SCENE)) return a.key === UNNAMED_SCENE ? 1 : -1;
		return b.latest - a.latest;
	});
}

/** m:ss, or h:mm:ss past an hour -- for clip ranges and runtimes. */
export function formatClock(seconds: number): string {
	const total = Math.max(0, Math.round(seconds));
	const h = Math.floor(total / 3600);
	const m = Math.floor((total % 3600) / 60);
	const sec = String(total % 60).padStart(2, "0");
	return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}
