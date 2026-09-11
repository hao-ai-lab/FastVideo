import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useGenerationCapabilities } from "./useGenerationCapabilities";

describe("generation capabilities", () => {
	afterEach(() => vi.unstubAllGlobals());
	it("enables only the modes the runtime advertises and identifies mock playback", async () => {
		vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ model_id: "mock", modes: ["t2va", "fl2va", "ref2va"], mock: true }) })));
		const { result } = renderHook(() => useGenerationCapabilities());
		await waitFor(() => expect(result.current.loadingCapabilities).toBe(false));
		expect(result.current.capabilities.modes).toEqual(["t2va", "fl2va", "ref2va"]);
		expect(result.current.capabilities.mock).toBe(true);
	});
	it("keeps old runtimes text-only when the capabilities endpoint is missing", async () => {
		vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 })));
		const { result } = renderHook(() => useGenerationCapabilities());
		await waitFor(() => expect(result.current.loadingCapabilities).toBe(false));
		expect(result.current.capabilities.modes).toEqual(["t2va"]);
		expect(result.current.capabilityNotice).toMatch(/Text-only compatibility/);
	});
});
