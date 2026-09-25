import { describe, expect, it } from "vitest";
import {
	applyH3Preset,
	applyH3Tag,
	H3_PRESETS,
	H3_TAG_CATEGORIES,
	H3_TAGS,
	type H3Preset,
	type H3Tag,
} from "@/lib/h3Tags";

const pointTag = H3_TAGS.find((t) => t.id === "pause") as H3Tag;
const wrapTag = H3_TAGS.find((t) => t.id === "emphasis") as H3Tag;
const emptyWrapTag = H3_TAGS.find((t) => t.id === "smacks-lips") as H3Tag;

describe("H3 tag catalogue", () => {
	it("gives every tag a unique id", () => {
		const ids = H3_TAGS.map((t) => t.id);
		expect(new Set(ids).size).toBe(ids.length);
	});

	it("gives wrap tags a close tag and point tags none", () => {
		for (const tag of H3_TAGS) {
			if (tag.kind === "wrap") expect(tag.close).toBeTruthy();
			else expect(tag.close).toBeUndefined();
		}
	});

	it("flattens to the same tags the categories declare", () => {
		const flattened = H3_TAG_CATEGORIES.flatMap((c) => c.tags);
		expect(H3_TAGS).toEqual(flattened);
	});
});

describe("applyH3Tag: point tags", () => {
	it("inserts at the cursor with padding on both sides", () => {
		const result = applyH3Tag("Okay, so. This is me.", 9, 9, pointTag);
		expect(result.value).toBe("Okay, so. <pause> This is me.");
		// cursor collapses to just past the inserted token
		expect(result.value.slice(result.start, result.end)).toBe("");
		expect(result.value.slice(0, result.start)).toBe("Okay, so. <pause> ".slice(0, result.start));
	});

	it("does not double up whitespace already present", () => {
		const result = applyH3Tag("Okay, so.  This is me.", 10, 10, pointTag);
		expect(result.value).toBe("Okay, so. <pause> This is me.");
	});

	it("inserts cleanly at the very start and end of the text", () => {
		expect(applyH3Tag("Hello.", 0, 0, pointTag).value).toBe("<pause> Hello.");
		expect(applyH3Tag("Hello.", 6, 6, pointTag).value).toBe("Hello. <pause>");
	});

	it("inserts before a selection, keeping the selected text", () => {
		const result = applyH3Tag("Hello there.", 6, 11, pointTag);
		expect(result.value).toBe("Hello <pause> there.");
		expect(result.value.slice(result.start, result.end)).toBe("there");
	});
});

describe("applyH3Tag: wrap tags", () => {
	it("wraps a selection and collapses the cursor after the close tag", () => {
		const result = applyH3Tag("I was not expecting that.", 6, 9, wrapTag);
		expect(result.value).toBe("I was <i>not</i> expecting that.");
		expect(result.start).toBe(result.end);
		expect(result.value.slice(0, result.start)).toBe("I was <i>not</i>");
	});

	it("inserts an empty open+close pair with the cursor between them when nothing is selected", () => {
		const result = applyH3Tag("Don't tell anyone.", 0, 0, wrapTag);
		expect(result.value).toBe("<i></i>Don't tell anyone.");
		expect(result.start).toBe("<i>".length);
		expect(result.end).toBe(result.start);
	});

	it("supports an intentionally empty pair, as used for a sentence-final effect", () => {
		const result = applyH3Tag("Okay.", 0, 0, emptyWrapTag);
		expect(result.value).toBe("<smacks lips></smacks lips>Okay.");
	});
});

describe("applyH3Tag: out-of-range selection", () => {
	it("clamps a selection beyond the text length", () => {
		const result = applyH3Tag("Hi.", 10, 20, pointTag);
		expect(result.value).toBe("Hi. <pause>");
	});
});

describe("applyH3Preset", () => {
	const insertPreset = H3_PRESETS.find((p) => p.label === "Nervous") as H3Preset;
	const wrapPreset = H3_PRESETS.find((p) => p.label === "Intimate") as H3Preset;

	it("inserts with a leading space when the cursor follows text", () => {
		const r = applyH3Preset("Well", 4, 4, insertPreset);
		expect(r.value).toBe("Well <uh> <stutter> ");
		expect(r.start).toBe(r.value.length);
		expect(r.end).toBe(r.value.length);
	});

	it("adds no leading space at the start or after whitespace", () => {
		expect(applyH3Preset("", 0, 0, insertPreset).value).toBe("<uh> <stutter> ");
		expect(applyH3Preset("Well ", 5, 5, insertPreset).value).toBe("Well <uh> <stutter> ");
	});

	it("wraps a selection and keeps it selected", () => {
		const r = applyH3Preset("say it softly now", 4, 13, wrapPreset);
		expect(r.value).toBe("say <whisper>it softly</whisper> now");
		expect(r.value.slice(r.start, r.end)).toBe("it softly");
	});

	it("wraps an empty selection with the caret between the tags", () => {
		const r = applyH3Preset("hi ", 3, 3, wrapPreset);
		expect(r.value).toBe("hi <whisper></whisper>");
		expect(r.start).toBe(r.end);
		expect(r.value.slice(0, r.start)).toBe("hi <whisper>");
	});
});
