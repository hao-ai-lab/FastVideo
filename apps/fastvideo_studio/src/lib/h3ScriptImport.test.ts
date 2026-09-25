import { describe, expect, it } from "vitest";
import {
	buildClips,
	convertDirections,
	parseScript,
	scriptSpeakers,
	type ImportSpeaker,
} from "@/lib/h3ScriptImport";
import { detailedDescriptionFromShots, shotLines, shotProse } from "@/lib/h3Shots";

// Original test material -- not from any real screenplay.
const SCRIPT = [
	"INT. HARBOR OFFICE - DAY",
	"",
	"CAPTAIN: Sit down. (EXHALES) We need to talk about the ledger.",
	"",
	"CLERK: Yes, sir.",
	"",
	"CAPTAIN: Three crates are missing, and nobody on this dock saw a thing.",
	"Not one of you. That does not happen by accident.",
	"",
	"CLERK: I counted twice.",
	"",
	"CAPTAIN: (WHISPERING) Then count a third time.",
	"",
	"CLERK: (TAPPING TABLE) Right away.",
].join("\n");

const SPEAKERS: Record<string, ImportSpeaker> = {
	CAPTAIN: { subject: "<Subject 1>", tag: "S1" },
	CLERK: { subject: "<Subject 2>", tag: "S2" },
};

describe("parseScript", () => {
	it("reads NAME: line turns and ignores headings between turns", () => {
		const turns = parseScript(SCRIPT);
		expect(turns.map((t) => t.speaker)).toEqual(["CAPTAIN", "CLERK", "CAPTAIN", "CLERK", "CAPTAIN", "CLERK"]);
		expect(turns.some((t) => t.text.includes("HARBOR"))).toBe(false);
	});

	it("joins continuation lines directly under a turn", () => {
		const turns = parseScript(SCRIPT);
		expect(turns[2].text).toBe(
			"Three crates are missing, and nobody on this dock saw a thing. Not one of you. That does not happen by accident.",
		);
	});

	it("tolerates markdown bold speaker names", () => {
		expect(parseScript("**MARK:** hello there")).toEqual([{ speaker: "MARK", text: "hello there" }]);
	});

	it("lists speakers in order of first appearance", () => {
		expect(scriptSpeakers(parseScript(SCRIPT))).toEqual(["CAPTAIN", "CLERK"]);
	});

	it("returns nothing for text with no speaker turns", () => {
		expect(parseScript("just some prose without turns")).toEqual([]);
	});
});

describe("convertDirections", () => {
	it("turns known parentheticals into speech tags in place", () => {
		expect(convertDirections("Sit down. (EXHALES) We talk.").text).toBe("Sit down. <exhale> We talk.");
		expect(convertDirections("Well (BEAT) fine.").text).toBe("Well <pause> fine.");
	});

	it("wraps the line for a whispering direction", () => {
		expect(convertDirections("(WHISPERING) Keep it down.").text).toBe("<whisper>Keep it down.</whisper>");
	});

	it("keeps unmapped directions out of the dialogue but reports them", () => {
		const r = convertDirections("(TAPPING TABLE) Right away.");
		expect(r.text).toBe("Right away.");
		expect(r.directions).toEqual(["tapping table"]);
	});
});

describe("buildClips", () => {
	it("opens the first clip with an establishing shot", () => {
		const [clip] = buildClips(parseScript(SCRIPT), { speakers: SPEAKERS, targetSeconds: 60 });
		expect(clip.shots[0].type).toBe("establishing");
		expect(shotLines(clip.shots[0])).toHaveLength(0);
		expect(clip.shots[0].subjectIds).toEqual(["<Subject 1>", "<Subject 2>"]);
	});

	it("folds a short reaction into the shot it reacts to", () => {
		const [clip] = buildClips(parseScript(SCRIPT), { speakers: SPEAKERS, targetSeconds: 60 });
		const first = clip.shots[1];
		expect(shotLines(first).map((l) => [l.speaker, l.text])).toEqual([
			["S1", "Sit down. <exhale> We need to talk about the ledger."],
			["S2", "Yes, sir."],
		]);
		expect(first.type).toBe("medium");
		expect(first.subjectIds).toEqual(["<Subject 1>", "<Subject 2>"]);
	});

	it("uses a close-up when only one person speaks in a shot", () => {
		const [clip] = buildClips([{ speaker: "CAPTAIN", text: "Nobody leaves until the count is done." }], {
			speakers: SPEAKERS,
			targetSeconds: 60,
		});
		const shot = clip.shots[1];
		expect(shot.type).toBe("close-up");
		expect(shot.subjectIds).toEqual(["<Subject 1>"]);
		expect(shotProse(shot)).toBe("Close on <Subject 1>, speaking.");
	});

	it("records unmapped stage directions on the shot they belong to", () => {
		const [clip] = buildClips(parseScript(SCRIPT), { speakers: SPEAKERS, targetSeconds: 60 });
		const last = clip.shots[clip.shots.length - 1];
		expect(shotLines(last).map((l) => l.text)).toEqual(["<whisper>Then count a third time.</whisper>", "Right away."]);
		expect(shotProse(last)).toContain("Stage direction: tapping table.");
	});

	it("splits a long scene into several clips, each with dialogue", () => {
		const clips = buildClips(parseScript(SCRIPT), { speakers: SPEAKERS, targetSeconds: 5 });
		expect(clips.length).toBeGreaterThan(1);
		for (const c of clips) expect(c.shots.some((s) => shotLines(s).length > 0)).toBe(true);
		expect(clips[1].shots[0].type).not.toBe("establishing");
	});

	it("keeps every spoken word across clips", () => {
		const clips = buildClips(parseScript(SCRIPT), { speakers: SPEAKERS, targetSeconds: 5 });
		const all = clips.flatMap((c) => c.shots.flatMap((s) => shotLines(s).map((l) => l.text))).join(" ");
		for (const word of ["ledger", "crates", "accident", "counted", "third", "Right away"]) {
			expect(all).toContain(word);
		}
	});

	it("serializes to valid [Shot N] text with speaker-tagged dialogue", () => {
		const [clip] = buildClips(parseScript(SCRIPT), { speakers: SPEAKERS, targetSeconds: 60 });
		const text = detailedDescriptionFromShots(clip.shots);
		expect(text).toContain("[Shot 1] Establishing shot.");
		expect(text).toContain("(S1) <d>[English] Sit down. <exhale> We need to talk about the ledger.</d>");
		expect(text).toContain("(S2) <d>[English] Yes, sir.</d>");
	});

	describe("speaker tags per clip", () => {
		it("numbers speakers by who speaks first in each clip, not across the whole script", () => {
			const turns = [
				{ speaker: "CAPTAIN", text: "Every one of these lines is long enough to stand as its own turn today." },
				{ speaker: "CLERK", text: "And every reply here is long enough to stand as its own turn as well." },
				{ speaker: "CLERK", text: "The clerk speaks again first when the next clip begins, with a full sentence." },
				{ speaker: "CAPTAIN", text: "And then the captain answers the clerk with another complete sentence here." },
			];
			const clips = buildClips(turns, { speakers: SPEAKERS, targetSeconds: 8 });
			expect(clips.length).toBeGreaterThan(1);
			for (const c of clips) {
				const first = c.shots.flatMap((s) => shotLines(s))[0];
				expect(first.speaker).toBe("S1");
			}
			// a clip that opens on the clerk maps the clerk to S1 and the captain to S2
			const clerkFirst = clips.find((c) => shotLines(c.shots.find((s) => shotLines(s).length)!)[0].text.startsWith("The clerk"));
			const speakers = clerkFirst!.shots.flatMap((s) => shotLines(s).map((l) => l.speaker));
			expect(new Set(speakers).size).toBeLessThanOrEqual(2);
		});

		it("leaves custom speaker tags alone", () => {
			const clips = buildClips(
				[{ speaker: "CLERK", text: "A line for the clerk that is long enough to stand alone here." }],
				{ speakers: { CLERK: { subject: "<Subject 2>", tag: "CLERK" }, CAPTAIN: SPEAKERS.CAPTAIN }, targetSeconds: 11 },
			);
			expect(shotLines(clips[0].shots[1])[0].speaker).toBe("CLERK");
		});
	});

	describe("clip boundaries", () => {
		const longSpeech = Array.from({ length: 12 }, (_, i) => `This is sentence number ${i + 1} of a long speech.`).join(" ");

		it("keeps a long turn as one continuous shot when a clip can hold it", () => {
			const clips = buildClips([{ speaker: "CAPTAIN", text: longSpeech }], { speakers: SPEAKERS, targetSeconds: 60 });
			expect(clips).toHaveLength(1);
			const dialogueShots = clips[0].shots.filter((s) => shotLines(s).length > 0);
			expect(dialogueShots).toHaveLength(1);
			expect(shotLines(dialogueShots[0])[0].text).toBe(longSpeech);
		});

		it("only splits inside a shot when the turn is longer than a whole clip, and marks the follow-on clips", () => {
			const clips = buildClips([{ speaker: "CAPTAIN", text: longSpeech }], { speakers: SPEAKERS, targetSeconds: 11 });
			expect(clips.length).toBeGreaterThan(1);
			expect(clips[0].continuation).toBe(false);
			for (const c of clips.slice(1)) {
				expect(c.continuation).toBe(true);
				expect(shotProse(c.shots[0])).toContain("Continuing the same shot");
				expect(c.shots[0].type).toBe("close-up");
			}
			const spoken = clips.flatMap((c) => c.shots.flatMap((s) => shotLines(s).map((l) => l.text))).join(" ");
			expect(spoken).toBe(longSpeech);
		});

		it("marks speech that runs past the end of a clip with <cutoff>, and only there", () => {
			const clips = buildClips([{ speaker: "CAPTAIN", text: longSpeech }], { speakers: SPEAKERS, targetSeconds: 11 });
			const endings = clips.map((c) => detailedDescriptionFromShots(c.shots));
			for (const text of endings.slice(0, -1)) expect(text).toMatch(/<\/d> <cutoff>$/);
			expect(endings[endings.length - 1]).not.toContain("<cutoff>");
			// the marker is outside the spoken text, so it never leaks into the dialogue itself
			const spoken = clips.flatMap((c) => c.shots.flatMap((s) => shotLines(s).map((l) => l.text))).join(" ");
			expect(spoken).not.toContain("cutoff");
		});

		it("breaks between separate turns at a cut, not as a continuation", () => {
			const turns = Array.from({ length: 6 }, () => ({
				speaker: "CAPTAIN",
				text: "Every one of these turns is a complete thought that takes a while to say aloud.",
			}));
			const clips = buildClips(turns, { speakers: SPEAKERS, targetSeconds: 11 });
			expect(clips.length).toBeGreaterThan(1);
			expect(clips.every((c) => !c.continuation)).toBe(true);
		});

		it("re-wraps a whispered line on every piece so no tag is left unmatched", () => {
			const clips = buildClips([{ speaker: "CAPTAIN", text: `(WHISPERING) ${longSpeech}` }], {
				speakers: SPEAKERS,
				targetSeconds: 11,
			});
			const texts = clips.flatMap((c) => c.shots.flatMap((s) => shotLines(s).map((l) => l.text)));
			expect(texts.length).toBeGreaterThan(1);
			for (const t of texts) {
				expect(t.startsWith("<whisper>")).toBe(true);
				expect(t.endsWith("</whisper>")).toBe(true);
			}
		});

		it("puts a trailing reaction with the last piece of a split speech", () => {
			const clips = buildClips(
				[
					{ speaker: "CAPTAIN", text: longSpeech },
					{ speaker: "CLERK", text: "Understood." },
				],
				{ speakers: SPEAKERS, targetSeconds: 11 },
			);
			const lastShot = clips[clips.length - 1].shots.at(-1)!;
			expect(shotLines(lastShot).map((l) => l.speaker)).toEqual(["S1", "S2"]);
			const earlier = clips.slice(0, -1).flatMap((c) => c.shots.flatMap((s) => shotLines(s).map((l) => l.speaker)));
			expect(earlier.every((sp) => sp === "S1")).toBe(true);
		});
	});
});
