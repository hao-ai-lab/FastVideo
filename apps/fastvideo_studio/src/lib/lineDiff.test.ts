import { describe, expect, it } from "vitest";

import { diffLines } from "./lineDiff";

describe("diffLines", () => {
	it("marks a changed line as removed then added, keeping the rest", () => {
		expect(diffLines("a\nslap on each four\nc", "a\nslap on the first four\nc")).toEqual([
			{ kind: "same", text: "a" },
			{ kind: "removed", text: "slap on each four" },
			{ kind: "added", text: "slap on the first four" },
			{ kind: "same", text: "c" },
		]);
	});

	it("handles added and removed lines at either end", () => {
		expect(diffLines("x\ny", "y\nz")).toEqual([
			{ kind: "removed", text: "x" },
			{ kind: "same", text: "y" },
			{ kind: "added", text: "z" },
		]);
	});

	it("returns only kept lines for identical texts", () => {
		expect(diffLines("a\nb", "a\nb").every((l) => l.kind === "same")).toBe(true);
	});
});
