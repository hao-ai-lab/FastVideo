import { describe, expect, it } from "vitest";
import { readField, shotTarget, writeField, writeShotTarget } from "@/lib/h3PromptFields";

const BASE = [
	"For the target video, <Picture 1> is fully referenced.",
	"",
	"integrated_multimodal_description: [Shot 1] A close-up. He (S1) says: <d>[English] Hi.</d> <cutoff>",
	"",
	"overall_soundscape: Room tone.",
	"",
	"non_diegetic_music: N/A",
].join("\n");

const SECTIONS = [
	"subject_definitions:",
	"<Subject 1> is the courier.",
	"",
	"retention_analysis:",
	"<Subject 1> (appears in [Shot 1]): fully_preserved - x.",
	"",
	"detailed_description:",
	"[Shot 1] Wide shot. A rooftop.",
	"[Shot 2] At 00:02.000, the shot cuts to a close-up shot. A face.",
	"",
	"overall_soundscape:",
	"Rain.",
].join("\n");

describe("readField / writeField", () => {
	it("reads a field whose text follows the heading on the same line", () => {
		expect(readField(BASE, "integrated_multimodal_description")).toBe(
			"[Shot 1] A close-up. He (S1) says: <d>[English] Hi.</d> <cutoff>",
		);
	});

	it("reads a field whose text starts on the line after the heading", () => {
		expect(readField(SECTIONS, "detailed_description")).toBe(
			"[Shot 1] Wide shot. A rooftop.\n[Shot 2] At 00:02.000, the shot cuts to a close-up shot. A face.",
		);
		expect(readField(SECTIONS, "subject_definitions")).toBe("<Subject 1> is the courier.");
	});

	it("returns null for a field that isn't there", () => {
		expect(readField(BASE, "detailed_description")).toBeNull();
	});

	it("rewrites only that field, leaving the rest byte-identical (inline style)", () => {
		const out = writeField(BASE, "integrated_multimodal_description", "[Shot 1] New.");
		expect(out).toBe(BASE.replace("[Shot 1] A close-up. He (S1) says: <d>[English] Hi.</d> <cutoff>", "[Shot 1] New."));
	});

	it("rewrites only that field (own-line style)", () => {
		const out = writeField(SECTIONS, "detailed_description", "[Shot 1] Only.");
		expect(out).toContain("detailed_description:\n[Shot 1] Only.\n\noverall_soundscape:\nRain.");
		expect(out.startsWith("subject_definitions:\n<Subject 1> is the courier.")).toBe(true);
	});

	it("writing a field's own text back is a no-op", () => {
		for (const [prompt, field] of [
			[BASE, "integrated_multimodal_description"],
			[SECTIONS, "detailed_description"],
			[SECTIONS, "retention_analysis"],
		] as const) {
			expect(writeField(prompt, field, readField(prompt, field) as string)).toBe(prompt);
		}
	});

	it("ignores a heading that isn't at the start of a line", () => {
		expect(readField("see detailed_description: not a header", "detailed_description")).toBeNull();
	});
});

describe("shotTarget", () => {
	it("prefers integrated_multimodal_description", () => {
		expect(shotTarget(BASE).field).toBe("integrated_multimodal_description");
	});

	it("falls back to detailed_description", () => {
		expect(shotTarget(SECTIONS).field).toBe("detailed_description");
	});

	it("treats a plain prompt as the shot text itself", () => {
		const t = shotTarget("A toy car drives into a plush dog.");
		expect(t).toEqual({ field: null, body: "A toy car drives into a plush dog." });
		expect(writeShotTarget("A toy car drives into a plush dog.", t, "[Shot 1] X.")).toBe("[Shot 1] X.");
	});
});
