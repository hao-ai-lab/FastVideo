import { describe, expect, it } from "vitest";

import {
	DEFAULT_CAPABILITIES,
	capabilitiesOf,
	formatSource,
	recipeChip,
	recipeRows,
} from "@/lib/modelCapabilities";
import { OMNI_RECIPE } from "@/test/factories";


describe("capabilitiesOf", () => {
	it("keeps the old behaviour for servers that report no capabilities", () => {
		expect(capabilitiesOf({})).toBe(DEFAULT_CAPABILITIES);
		expect(capabilitiesOf(undefined).references).toBeNull();
	});
});

describe("recipeRows", () => {
	it("states PDD blocks and forwards, so PDD-8 is not read as a nine-point ladder", () => {
		const rows = Object.fromEntries(recipeRows(OMNI_RECIPE, 1).map((r) => [r.label, r.value]));
		expect(rows.Sampling).toBe("8 fused blocks of a 32-interval grid = 8 transformer forwards");
		expect(rows["Base model"]).toBe("MiniMaxAI/MiniMax-H3@9bfb6693f2cf");
		expect(rows.Attention).toContain("128-token tiles");
		expect(rows.Parallelism).toBe("1 GPU");
	});

	it("describes multi-GPU sharding", () => {
		const rows = recipeRows(OMNI_RECIPE, 4);
		expect(rows.find((r) => r.label === "Parallelism")?.value).toMatch(/4 GPUs.*FSDP/);
	});
});

describe("formatting", () => {
	it("shortens commit hashes and leaves branch names", () => {
		expect(formatSource("repo", "main")).toBe("repo@main");
		expect(formatSource("/local/dir", null)).toBe("/local/dir");
	});

	it("summarises a recipe for job cards", () => {
		expect(recipeChip(OMNI_RECIPE)).toBe("8 forwards · VSA 0.9 · tile 128");
	});
});
