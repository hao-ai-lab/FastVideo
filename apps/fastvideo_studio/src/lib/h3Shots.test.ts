import { describe, expect, it } from "vitest";
import {
	detailedDescriptionFromShots,
	formatCutTime,
	makeShot,
	newLine,
	parseCutTime,
	parseDetailedDescription,
	parseSubjectLabels,
	resolveShotStarts,
	retentionAnalysisFromShots,
	shotLines,
	shotProse,
	shotsFromDetailedDescription,
	type Shot,
	type ShotInit,
} from "@/lib/h3Shots";

function shot(init: ShotInit): Shot {
	return makeShot(init);
}

describe("parseSubjectLabels", () => {
	it("returns ordered, de-duplicated <Subject N> labels", () => {
		const text = "<Subject 1> is the dog.\n<Subject 2> is the cat. <Subject 1> again.";
		expect(parseSubjectLabels(text)).toEqual(["<Subject 1>", "<Subject 2>"]);
	});

	it("returns an empty list when there are none", () => {
		expect(parseSubjectLabels("no subjects here")).toEqual([]);
	});
});

describe("detailedDescriptionFromShots", () => {
	it("serializes shot type, movement, description, and dialogue", () => {
		const shots = [
			shot({
				type: "close-up",
				movement: "handheld",
				description: "She looks up.",
				lines: [newLine({ speaker: "S1", text: "I found it." })],
			}),
		];
		expect(detailedDescriptionFromShots(shots)).toBe(
			"[Shot 1] Close-up shot, handheld. She looks up. (S1) <d>[English] I found it.</d>",
		);
	});

	it("omits the movement clause for a static camera", () => {
		const shots = [shot({ type: "wide", movement: "static", description: "An empty street." })];
		expect(detailedDescriptionFromShots(shots)).toBe("[Shot 1] Wide shot. An empty street.");
	});

	it("numbers shots in order, one per line, and marks later shots with a cut time", () => {
		const shots = [shot({ description: "First." }), shot({ description: "Second." })];
		const out = detailedDescriptionFromShots(shots);
		expect(out).toBe(
			"[Shot 1] Medium shot. First.\n[Shot 2] At 00:02.000, the shot cuts to a medium shot. Second.",
		);
	});

	it("uses the right article and movement in the cut wording", () => {
		const shots = [
			shot({ type: "wide" }),
			shot({ type: "establishing", startSeconds: 3 }),
			shot({ type: "extreme-close-up", movement: "dolly", startSeconds: 5.25 }),
			shot({ type: "pov", startSeconds: 7 }),
		];
		const out = detailedDescriptionFromShots(shots).split("\n");
		expect(out[1]).toBe("[Shot 2] At 00:03.000, the shot cuts to an establishing shot.");
		expect(out[2]).toBe("[Shot 3] At 00:05.250, the shot cuts to an extreme close-up shot, dolly.");
		expect(out[3]).toBe("[Shot 4] At 00:07.000, the shot cuts to a POV shot.");
	});

	it("writes the preamble before the first shot", () => {
		const out = detailedDescriptionFromShots([shot({ description: "A." })], "Keep the film-noir look.");
		expect(out).toBe("Keep the film-noir look.\n[Shot 1] Medium shot. A.");
	});
});

describe("cut times", () => {
	it("formats and parses MM:SS.mmm", () => {
		expect(formatCutTime(3.5)).toBe("00:03.500");
		expect(formatCutTime(75.04)).toBe("01:15.040");
		expect(parseCutTime("00:03.500")).toBe(3.5);
		expect(parseCutTime("0:03.5")).toBe(3.5);
		expect(parseCutTime("3.5")).toBe(3.5);
		expect(parseCutTime("1:15")).toBe(75);
		expect(parseCutTime("soon")).toBeNull();
	});

	it("starts the opening shot at 0 and follows estimated shot lengths", () => {
		const shots = [
			shot({ lines: [newLine({ text: "A short line." })] }),
			shot({}),
			shot({}),
		];
		const starts = resolveShotStarts(shots);
		expect(starts[0]).toBe(0);
		expect(starts[1]).toBeGreaterThanOrEqual(2);
		expect(starts[2]).toBeCloseTo(starts[1] + 2, 3);
	});

	it("honors an explicit time and keeps times strictly increasing", () => {
		const shots = [shot({}), shot({ startSeconds: 4 }), shot({ startSeconds: 1 })];
		const starts = resolveShotStarts(shots);
		expect(starts[1]).toBe(4);
		expect(starts[2]).toBeGreaterThan(starts[1]);
	});
});

describe("shotsFromDetailedDescription", () => {
	it("round-trips type, movement, dialogue, speaker, and subjects", () => {
		const original = [
			shot({
				type: "extreme-close-up",
				movement: "dolly",
				description: "Her hand trembles.",
				lines: [newLine({ speaker: "S1", text: "Wait." })],
				subjectIds: ["<Subject 1>"],
			}),
		];
		const serialized = detailedDescriptionFromShots(original);
		const withSubject = serialized.replace("Her hand trembles.", "Her hand trembles, <Subject 1> hesitates.");
		const parsed = shotsFromDetailedDescription(withSubject, ["<Subject 1>", "<Subject 2>"]);

		expect(parsed).toHaveLength(1);
		expect(parsed[0].type).toBe("extreme-close-up");
		expect(parsed[0].movement).toBe("dolly");
		expect(shotLines(parsed[0])).toHaveLength(1);
		expect(shotLines(parsed[0])[0].speaker).toBe("S1");
		expect(shotLines(parsed[0])[0].text).toBe("Wait.");
		expect(parsed[0].subjectIds).toEqual(["<Subject 1>"]);
	});

	it("keeps every dialogue line of an exchange, in order", () => {
		const original = [
			shot({
				description: "They talk.",
				lines: [
					newLine({ speaker: "S1", text: "Ready?" }),
					newLine({ speaker: "S2", text: "<pause> Always." }),
					newLine({ speaker: "S1", text: "Then go." }),
				],
			}),
		];
		const serialized = detailedDescriptionFromShots(original);
		expect(serialized).toBe(
			"[Shot 1] Medium shot. They talk. (S1) <d>[English] Ready?</d> (S2) <d>[English] <pause> Always.</d> (S1) <d>[English] Then go.</d>",
		);

		const parsed = shotsFromDetailedDescription(serialized, []);
		expect(shotProse(parsed[0])).toBe("They talk.");
		expect(shotLines(parsed[0]).map((l) => [l.speaker, l.text])).toEqual([
			["S1", "Ready?"],
			["S2", "<pause> Always."],
			["S1", "Then go."],
		]);
	});

	it("round-trips a non-English language marker instead of relabelling it English", () => {
		const original = [shot({ lines: [newLine({ speaker: "S1", text: "你好。", language: "Chinese" })] })];
		const serialized = detailedDescriptionFromShots(original);
		expect(serialized).toContain("<d>[Chinese] 你好。</d>");
		expect(shotLines(shotsFromDetailedDescription(serialized, [])[0])[0].language).toBe("Chinese");
	});

	it("round-trips a multi-shot list without pinning the automatic cut times", () => {
		const original = [
			shot({ type: "establishing", description: "A rooftop." }),
			shot({
				type: "close-up",
				movement: "handheld",
				description: "He turns.",
				lines: [newLine({ speaker: "S2", text: "I had no choice." })],
			}),
			shot({ type: "wide", movement: "dolly", description: "They stand apart." }),
		];
		const serialized = detailedDescriptionFromShots(original);
		const parsed = shotsFromDetailedDescription(serialized, []);

		expect(parsed.map((p) => [p.type, p.movement, p.freeLead])).toEqual([
			["establishing", "static", false],
			["close-up", "handheld", false],
			["wide", "dolly", false],
		]);
		expect(parsed.every((p) => p.startSeconds === null)).toBe(true);
		expect(detailedDescriptionFromShots(parsed)).toBe(serialized);
	});

	it("pins a cut time that differs from the automatic one", () => {
		const parsed = shotsFromDetailedDescription(
			"[Shot 1] Wide shot. A dock.\n[Shot 2] At 00:07.400, the shot cuts to a close-up shot. A face.",
			[],
		);
		expect(parsed[1].startSeconds).toBe(7.4);
	});

	it("keeps hand-written shots verbatim, with the preamble and cut times", () => {
		const handWritten = [
			"The target video keeps the live-action style of <Video 1> unchanged.",
			"[Shot 1] A handheld medium shot holds <Subject 1> at frame left, half-occluded by plastic sheeting.",
			"[Shot 2] At 00:03.800, the shot cuts to a low tracking shot behind the boots of <Subject 1> as he sprints.",
			"[Shot 3] At 00:07.400, the camera cuts to a close front-on tracking shot. He (S1) says: <d>[English] Go.</d> <cutoff>",
		].join("\n");
		const { preamble, shots } = parseDetailedDescription(handWritten, ["<Subject 1>"]);

		expect(preamble).toBe("The target video keeps the live-action style of <Video 1> unchanged.");
		expect(shots).toHaveLength(3);
		expect(shots.every((s) => s.freeLead)).toBe(true);
		expect(shots.map((s) => s.startSeconds)).toEqual([null, 3.8, 7.4]);
		expect(shots[0].subjectIds).toEqual(["<Subject 1>"]);
		// Dialogue in a hand-written shot is pulled out into an editable line...
		expect(shotLines(shots[2]).map((l) => [l.speaker, l.text])).toEqual([["", "Go."]]);
		expect(detailedDescriptionFromShots(shots, preamble)).toBe(handWritten);
	});

	it("keeps a trailing tag after the dialogue it modifies", () => {
		const text =
			"[Shot 1] He (S1) says: <d>[English] <breath> I think you're... <i>sorry</i>.</d> <cutoff>";
		const [shot1] = shotsFromDetailedDescription(text, []);
		expect(shot1.freeLead).toBe(true);
		expect(shot1.body.map((p) => p.kind)).toEqual(["text", "line", "text"]);
		expect(shotLines(shot1)[0].text).toBe("<breath> I think you're... <i>sorry</i>.");
		expect(detailedDescriptionFromShots([shot1])).toBe(text);
	});

	it("keeps interleaved dialogue in place, whitespace and all, in hand-written shots", () => {
		const text = "[Shot 1] She waits.\n  (S1) <d>[English] Ready?</d>\nHe nods.  (S2) <d>[English] Ready.</d>  The door opens.";
		const [s1] = shotsFromDetailedDescription(text, []);
		expect(s1.body.map((p) => p.kind)).toEqual(["text", "line", "text", "line", "text"]);
		expect(detailedDescriptionFromShots([s1])).toBe(text);
	});

	it("keeps dialogue between prose in shots this editor wrote too", () => {
		const text = "[Shot 1] Wide shot. He looks up. (S1) <d>[English] Hello.</d> Then he leaves.";
		const [s1] = shotsFromDetailedDescription(text, []);
		expect(s1.freeLead).toBe(false);
		expect(s1.body.map((p) => p.kind)).toEqual(["text", "line", "text"]);
		expect(detailedDescriptionFromShots([s1])).toBe(text);
	});

	it("edits a line without moving the text or lines around it", () => {
		const text = "[Shot 1] Wide shot. A. (S1) <d>[English] One.</d> B. (S2) <d>[English] Two.</d> C.";
		const [s1] = shotsFromDetailedDescription(text, []);
		const second = shotLines(s1)[1];
		second.text = "Changed.";
		expect(detailedDescriptionFromShots([s1])).toBe(
			"[Shot 1] Wide shot. A. (S1) <d>[English] One.</d> B. (S2) <d>[English] Changed.</d> C.",
		);
	});

	it("puts a newly added line at the end and separates it from the prose", () => {
		const text = "[Shot 1] Wide shot. He speaks. (S1) <d>[English] First.</d> He pauses.";
		const [s1] = shotsFromDetailedDescription(text, []);
		s1.body.push({ kind: "line", line: newLine({ speaker: "S1", text: "Later." }) });
		expect(detailedDescriptionFromShots([s1])).toBe(
			"[Shot 1] Wide shot. He speaks. (S1) <d>[English] First.</d> He pauses. (S1) <d>[English] Later.</d>",
		);
	});

	it("reordering blocks changes the written order", () => {
		const [s1] = shotsFromDetailedDescription(
			"[Shot 1] Wide shot. (S1) <d>[English] One.</d> (S2) <d>[English] Two.</d>",
			[],
		);
		const lineBlocks = s1.body.filter((p) => p.kind === "line");
		s1.body = [...s1.body.filter((p) => p.kind !== "line"), lineBlocks[1], lineBlocks[0]];
		expect(detailedDescriptionFromShots([s1])).toBe(
			"[Shot 1] Wide shot. (S2) <d>[English] Two.</d> (S1) <d>[English] One.</d>",
		);
	});

	it("recovers type and movement from the guide's cut wording", () => {
		const parsed = shotsFromDetailedDescription(
			"[Shot 1] Wide shot.\n[Shot 2] At 00:04.000, the camera cuts to an over-the-shoulder shot, handheld. Two people talk.",
			[],
		);
		expect(parsed[1].type).toBe("over-the-shoulder");
		expect(parsed[1].movement).toBe("handheld");
		expect(shotProse(parsed[1])).toBe("Two people talk.");
		expect(parsed[1].freeLead).toBe(false);
	});

	it("falls back to defaults for arbitrary prose, keeping it as the description", () => {
		const parsed = shotsFromDetailedDescription("A toy car drives into a plush dog.", []);
		expect(parsed).toHaveLength(1);
		expect(parsed[0].type).toBe("medium");
		expect(parsed[0].movement).toBe("static");
		expect(shotProse(parsed[0])).toBe("A toy car drives into a plush dog.");
	});

	it("returns an empty list for blank text", () => {
		expect(shotsFromDetailedDescription("", [])).toEqual([]);
	});
});

describe("retentionAnalysisFromShots", () => {
	it("generates an appears-in line per subject with shots", () => {
		const shots = [
			shot({ subjectIds: ["<Subject 1>"] }),
			shot({ subjectIds: ["<Subject 1>", "<Subject 2>"] }),
		];
		const out = retentionAnalysisFromShots(shots, ["<Subject 1>", "<Subject 2>"], "");
		expect(out).toBe(
			[
				"<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - what is retained.",
				"<Subject 2> (appears in [Shot 2]): fully_preserved - what is retained.",
			].join("\n"),
		);
	});

	it("preserves an existing retention note instead of overwriting it", () => {
		const shots = [shot({ subjectIds: ["<Subject 1>"] })];
		const previous = "<Subject 1> (appears in [Shot 1]): fully_preserved - fur and collar retained.";
		const out = retentionAnalysisFromShots(shots, ["<Subject 1>"], previous);
		expect(out).toBe("<Subject 1> (appears in [Shot 1]): fully_preserved - fur and collar retained.");
	});

	it("passes through lines it doesn't own, like audio references", () => {
		const shots = [shot({ subjectIds: ["<Subject 1>"] })];
		const previous = [
			"<Subject 1> (appears in [Shot 3]): fully_preserved - old note.",
			"<Audio 1>: reference - how it guides the audio.",
		].join("\n");
		const out = retentionAnalysisFromShots(shots, ["<Subject 1>"], previous);
		expect(out).toContain("<Subject 1> (appears in [Shot 1]): fully_preserved - old note.");
		expect(out).toContain("<Audio 1>: reference - how it guides the audio.");
	});

	it("omits a subject with no shots rather than emitting an empty appears-in", () => {
		const shots = [shot({ subjectIds: [] })];
		const out = retentionAnalysisFromShots(shots, ["<Subject 1>"], "");
		expect(out).toBe("");
	});
});
