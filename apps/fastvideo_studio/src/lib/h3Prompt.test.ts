import { describe, expect, it } from "vitest";
import {
	EMPTY_H3_PROMPT_FIELDS,
	H3_PROMPT_SECTIONS,
	isEmptyPromptFields,
	parseH3Prompt,
	serializeH3Prompt,
	type H3PromptFields,
} from "@/lib/h3Prompt";

const filled: H3PromptFields = {
	subject_definitions: "<Subject 1> is the dog in <Picture 1>.",
	summary: "[reference generation] The target video shows <Subject 1>.",
	retention_analysis: "<Subject 1> (appears in [Shot 1]): fully_preserved - fur retained.",
	detailed_description: "[Shot 1] A medium shot establishes <Subject 1>.",
	overall_soundscape: "Room tone throughout.",
	non_diegetic_music: "N/A",
};

describe("serializeH3Prompt", () => {
	it("emits sections in order, flush left, blank-line separated", () => {
		const out = serializeH3Prompt(filled);
		expect(out).toBe(
			[
				"subject_definitions:\n<Subject 1> is the dog in <Picture 1>.",
				"summary:\n[reference generation] The target video shows <Subject 1>.",
				"retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - fur retained.",
				"detailed_description:\n[Shot 1] A medium shot establishes <Subject 1>.",
				"overall_soundscape:\nRoom tone throughout.",
				"non_diegetic_music:\nN/A",
			].join("\n\n"),
		);
		// content is never indented
		expect(out).not.toMatch(/\n {2}\S/);
	});

	it("fills blank sections with N/A rather than dropping them", () => {
		const out = serializeH3Prompt({ ...EMPTY_H3_PROMPT_FIELDS, summary: "x" });
		for (const s of H3_PROMPT_SECTIONS) expect(out).toContain(`${s}:`);
		expect(out).toContain("non_diegetic_music:\nN/A");
	});
});

describe("parseH3Prompt", () => {
	it("round-trips a serialized prompt", () => {
		expect(parseH3Prompt(serializeH3Prompt(filled))).toEqual(filled);
	});

	it("keeps multi-line section bodies", () => {
		const p = parseH3Prompt("summary:\nline one\nline two\n\ndetailed_description:\nd");
		expect(p?.summary).toBe("line one\nline two");
		expect(p?.detailed_description).toBe("d");
	});

	it("returns null for a plain prompt", () => {
		expect(parseH3Prompt("a toy car drives into a plush dog")).toBeNull();
	});
});

describe("isEmptyPromptFields", () => {
	it("detects blank vs filled", () => {
		expect(isEmptyPromptFields(EMPTY_H3_PROMPT_FIELDS)).toBe(true);
		expect(isEmptyPromptFields(filled)).toBe(false);
	});
});
