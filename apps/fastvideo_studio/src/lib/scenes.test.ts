import { describe, expect, it } from "vitest";
import { UNNAMED_SCENE, buildScenes, clipSeconds, formatClock, parseSceneName, shotsOfJob } from "@/lib/scenes";
import { shotLines, shotProse } from "@/lib/h3Shots";
import type { Job } from "@/lib/types";

function job(over: Partial<Job> & { id: string }): Job {
	return {
		model_id: "m",
		prompt: "",
		status: "pending",
		created_at: 0,
		started_at: null,
		finished_at: null,
		error: null,
		output_path: null,
		log_file_path: null,
		num_inference_steps: 50,
		num_frames: 240,
		fps: 24,
		height: 768,
		width: 1344,
		guidance_scale: 1,
		seed: 0,
		num_gpus: 1,
		progress: 0,
		progress_msg: "",
		phase: "",
		...over,
	};
}

const SECTIONS = (detailed: string, subjects = "<Subject 1> is a man.") =>
	["subject_definitions:", subjects, "", "detailed_description:", detailed].join("\n");

describe("parseSceneName", () => {
	it.each([
		["wolf-lunch-clip-01", "wolf-lunch", 1],
		["wolf-lunch-clip-15", "wolf-lunch", 15],
		["closeup-fictional-actor-monologue-shot2", "closeup-fictional-actor-monologue", 2],
		["Rooftop Standoff part 3", "Rooftop Standoff", 3],
		["scene_a_07", "scene_a", 7],
	])("%s -> stem %s, clip %i", (name, title, order) => {
		expect(parseSceneName(name)).toEqual({ key: title.toLowerCase(), title, order });
	});

	it("treats a name with no trailing number as its own scene", () => {
		expect(parseSceneName("Rooftop Standoff")).toEqual({ key: "rooftop standoff", title: "Rooftop Standoff", order: null });
	});

	it("keeps the word scene as part of the name", () => {
		expect(parseSceneName("rooftop-scene-2")).toEqual({ key: "rooftop-scene", title: "rooftop-scene", order: 2 });
	});

	it("does not treat a bare number or number-only suffix as a stem", () => {
		expect(parseSceneName("42").order).toBeNull();
		expect(parseSceneName("clip-3").order).toBeNull();
	});

	it("puts blank or missing names in the unnamed group", () => {
		expect(parseSceneName(undefined).key).toBe(UNNAMED_SCENE);
		expect(parseSceneName("   ").key).toBe(UNNAMED_SCENE);
	});

	it("groups names that differ only in case", () => {
		expect(parseSceneName("Wolf-Lunch-Clip-1").key).toBe(parseSceneName("wolf-lunch-clip-2").key);
	});
});

describe("shotsOfJob", () => {
	it("reads shots and dialogue out of a six-section prompt", () => {
		const shots = shotsOfJob({
			prompt: SECTIONS(
				"[Shot 1] Wide shot. A dock.\n[Shot 2] At 00:02.000, the shot cuts to a close-up shot. (S1) <d>[English] Hi.</d>",
			),
		});
		expect(shots).toHaveLength(2);
		expect(shots[1].type).toBe("close-up");
		expect(shotLines(shots[1])[0].text).toBe("Hi.");
		expect(shots[0].subjectIds).toEqual([]);
	});

	it("reads shots out of a base-format integrated_multimodal_description", () => {
		const shots = shotsOfJob({
			prompt: "For the target video, <Picture 1> is used.\n\nintegrated_multimodal_description: [Shot 1] He (S1) says: <d>[English] Sorry.</d> <cutoff>\n\noverall_soundscape: Quiet.",
		});
		expect(shots).toHaveLength(1);
		expect(shotLines(shots[0]).map((l) => l.text)).toEqual(["Sorry."]);
	});

	it("shows a plain prompt as a single shot", () => {
		const shots = shotsOfJob({ prompt: "A toy car drives into a plush dog." });
		expect(shots).toHaveLength(1);
		expect(shotProse(shots[0])).toBe("A toy car drives into a plush dog.");
	});

	it("returns no shots for an empty prompt", () => {
		expect(shotsOfJob({ prompt: "" })).toEqual([]);
	});
});

describe("buildScenes", () => {
	const three = [
		job({ id: "c", name: "wolf-lunch-clip-03", created_at: 3, status: "pending", prompt: "[Shot 1] Wide shot. C." }),
		job({ id: "a", name: "wolf-lunch-clip-01", created_at: 1, status: "completed", num_frames: 240, prompt: "[Shot 1] Wide shot. A.\n[Shot 2] At 00:02.000, the shot cuts to a close-up shot. A2." }),
		job({ id: "b", name: "wolf-lunch-clip-02", created_at: 2, status: "completed", num_frames: 120, fps: 24, prompt: "[Shot 1] Close-up shot. B." }),
	];

	it("groups by stem and orders clips by their number, not creation order", () => {
		const [scene] = buildScenes(three);
		expect(scene.title).toBe("wolf-lunch");
		expect(scene.clips.map((c) => c.job.id)).toEqual(["a", "b", "c"]);
		expect(scene.clips.map((c) => c.position)).toEqual([1, 2, 3]);
	});

	it("lays every shot on one timeline using each clip's frames and fps", () => {
		const [scene] = buildScenes(three);
		expect(scene.clips.map((c) => c.startSeconds)).toEqual([0, 10, 15]);
		expect(scene.totalSeconds).toBe(25); // 240/24 + 120/24 + 240/24 = 10 + 5 + 10
	});

	it("counts shots and statuses across the scene", () => {
		const [scene] = buildScenes(three);
		expect(scene.shotCount).toBe(4);
		expect(scene.statusCounts).toEqual({ completed: 2, pending: 1 });
	});

	it("offsets each shot's cut time by the clip's start", () => {
		const [scene] = buildScenes(three);
		const second = scene.clips[1];
		expect(second.shotStarts).toEqual([10]);
		const first = scene.clips[0];
		expect(first.shotStarts[0]).toBe(0);
		expect(first.shotStarts[1]).toBe(2);
	});

	it("keeps separate scenes apart, newest first, with unnamed jobs last", () => {
		const scenes = buildScenes([
			job({ id: "1", name: "old-scene-1", created_at: 5 }),
			job({ id: "2", name: "new-scene-1", created_at: 50 }),
			job({ id: "3", name: "", created_at: 100 }),
		]);
		expect(scenes.map((s) => s.title)).toEqual(["new-scene", "old-scene", "Unnamed jobs"]);
	});

	it("puts clips without a number after numbered ones, oldest first", () => {
		const [scene] = buildScenes([
			job({ id: "x", name: "wolf-1", created_at: 9 }),
			job({ id: "y", name: "wolf", created_at: 1 }),
			job({ id: "z", name: "wolf-2", created_at: 2 }),
		]);
		// "wolf" shares the stem with wolf-1 / wolf-2, so it joins them -- last, having no number
		expect(scene.clips.map((c) => c.job.id)).toEqual(["x", "z", "y"]);
	});

	it("flags a clip that continues the previous shot", () => {
		const [scene] = buildScenes([
			job({ id: "a", name: "s-1", prompt: "[Shot 1] Close-up shot. A speech. <cutoff>" }),
			job({ id: "b", name: "s-2", prompt: "[Shot 1] Close-up shot, dolly. Continuing the same shot on <Subject 1>, still speaking." }),
			job({ id: "c", name: "s-3", prompt: "summary:\n[reference generation] It continues the previous shot without a cut.\n\ndetailed_description:\n[Shot 1] Wide shot. X." }),
			job({ id: "d", name: "s-4", prompt: "[Shot 1] Wide shot. Fresh." }),
		]);
		expect(scene.clips.map((c) => c.continuation)).toEqual([false, true, true, false]);
	});

	it("falls back to 24 fps when a job has none", () => {
		expect(clipSeconds({ num_frames: 240, fps: undefined })).toBe(10);
		expect(clipSeconds({ num_frames: 240, fps: 0 })).toBe(10);
		expect(clipSeconds({ num_frames: 240, fps: 30 })).toBe(8);
	});

	it("returns no scenes for no jobs", () => {
		expect(buildScenes([])).toEqual([]);
	});
});

describe("formatClock", () => {
	it("formats m:ss and h:mm:ss", () => {
		expect(formatClock(0)).toBe("0:00");
		expect(formatClock(9.6)).toBe("0:10");
		expect(formatClock(65)).toBe("1:05");
		expect(formatClock(3725)).toBe("1:02:05");
		expect(formatClock(-4)).toBe("0:00");
	});
});

describe("takes of the same clip", () => {
	const take = (id: string, n: number, over: Partial<Job> = {}) =>
		job({ id, name: `wolf-lunch-clip-0${n}`, prompt: `[Shot 1] Wide shot. ${id}.`, num_frames: 240, ...over });

	it("counts re-runs of a clip number once, not as extra clips", () => {
		const [scene] = buildScenes([
			take("a1", 1, { created_at: 1, status: "completed" }),
			take("a2", 1, { created_at: 2, status: "completed" }),
			take("a3", 1, { created_at: 3, status: "completed" }),
			take("b1", 2, { created_at: 4, status: "completed" }),
		]);
		expect(scene.clips).toHaveLength(2);
		expect(scene.totalSeconds).toBe(20);
		expect(scene.shotCount).toBe(2);
		expect(scene.statusCounts).toEqual({ completed: 2 });
		expect(scene.clips[0].takes.map((j) => j.id)).toEqual(["a1", "a2", "a3"]);
	});

	it("shows the most recent completed take, even over a newer failed one", () => {
		const [scene] = buildScenes([
			take("old", 1, { created_at: 1, status: "completed", finished_at: 10 }),
			take("good", 1, { created_at: 2, status: "completed", finished_at: 20 }),
			take("bad", 1, { created_at: 3, status: "failed", finished_at: 30 }),
		]);
		expect(scene.clips[0].job.id).toBe("good");
	});

	it("falls back to the newest take when none completed", () => {
		const [scene] = buildScenes([
			take("p1", 1, { created_at: 1, status: "failed" }),
			take("p2", 1, { created_at: 2, status: "pending" }),
		]);
		expect(scene.clips[0].job.id).toBe("p2");
	});

	it("shows the take the user chose, and lays the timeline out from it", () => {
		const jobs = [
			take("short", 1, { created_at: 1, status: "completed", num_frames: 120 }),
			take("long", 1, { created_at: 2, status: "completed", num_frames: 480 }),
			take("next", 2, { created_at: 3 }),
		];
		const [auto] = buildScenes(jobs);
		expect(auto.clips[0].job.id).toBe("long");
		expect(auto.clips[1].startSeconds).toBe(20);

		const [chosen] = buildScenes(jobs, { [auto.clips[0].takeKey as string]: "short" });
		expect(chosen.clips[0].job.id).toBe("short");
		expect(chosen.clips[1].startSeconds).toBe(5);
	});

	it("ignores a choice that isn't one of the takes", () => {
		const jobs = [take("a", 1, { status: "completed" })];
		const [scene] = buildScenes(jobs, { "wolf-lunch#1": "no-such-job" });
		expect(scene.clips[0].job.id).toBe("a");
	});

	it("keeps each unnumbered job as its own clip", () => {
		const [scene] = buildScenes([
			job({ id: "u1", name: "solo", created_at: 1 }),
			job({ id: "u2", name: "solo", created_at: 2 }),
		]);
		expect(scene.clips.map((c) => c.job.id)).toEqual(["u1", "u2"]);
		expect(scene.clips.every((c) => c.takeKey === null)).toBe(true);
	});
});
