import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAssetLibrary } from "./useAssetLibrary";

const image = { asset_id: "asset-1", kind: "image", name: "frame.png", mime_type: "image/png", size: 5, url: "/assets/asset-1" };

describe("asset library lifecycle", () => {
	beforeEach(() => localStorage.clear());
	afterEach(() => vi.unstubAllGlobals());
	it("uploads the raw file and keeps assignment state separate from uploaded assets", async () => {
		const fetchMock = vi.fn(async () => ({ ok: true, json: async () => image }));
		vi.stubGlobal("fetch", fetchMock);
		const { result } = renderHook(() => useAssetLibrary());
		const file = new File(["image"], "first frame.png", { type: "image/png" });
		await act(async () => { await result.current.uploadAssets([file]); });
		expect(fetchMock).toHaveBeenCalledWith("/assets", expect.objectContaining({ body: file, headers: { "Content-Type": "image/png", "X-Asset-Name": "first%20frame.png" } }));
		act(() => result.current.assignAsset("asset-1", "first_frame"));
		expect(result.current.conditioningAssets).toEqual([{ asset_id: "asset-1", role: "first_frame" }]);
		act(() => result.current.clearConditioning());
		expect(result.current.assets).toHaveLength(1);
		expect(result.current.conditioningAssets).toEqual([]);
	});
	it("identifies stale server assets without downloading their contents", async () => {
		localStorage.setItem("dreamverse-asset-library-v1", JSON.stringify([image]));
		const fetchMock = vi.fn(async () => ({ ok: false, status: 404 }));
		vi.stubGlobal("fetch", fetchMock);
		const { result } = renderHook(() => useAssetLibrary());
		await waitFor(() => expect(result.current.assets).toHaveLength(1));
		act(() => result.current.assignAsset("asset-1", "first_frame"));
		let message: string | null = null;
		await act(async () => { message = await result.current.verifySelectedAssets(); });
		expect(message).toMatch(/Upload it again/);
		expect(result.current.assets[0].missing).toBe(true);
		expect(fetchMock).toHaveBeenCalledWith("/assets/asset-1", expect.objectContaining({ method: "HEAD" }));
	});
	it("deletes both the library entry and its selected references", async () => {
		localStorage.setItem("dreamverse-asset-library-v1", JSON.stringify([image]));
		vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true })));
		const { result } = renderHook(() => useAssetLibrary());
		await waitFor(() => expect(result.current.assets).toHaveLength(1));
		act(() => result.current.assignAsset("asset-1", "reference"));
		await act(async () => { await result.current.removeAsset("asset-1"); });
		expect(result.current.assets).toEqual([]);
		expect(result.current.conditioningAssets).toEqual([]);
	});

	it("does not mark a valid upload missing when the browser cannot preview its codec", async () => {
		localStorage.setItem("dreamverse-asset-library-v1", JSON.stringify([image]));
		vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200 })));
		const { result } = renderHook(() => useAssetLibrary());
		await waitFor(() => expect(result.current.assets).toHaveLength(1));
		await act(async () => { await result.current.checkAssetAvailability("asset-1"); });
		expect(result.current.assets[0].missing).not.toBe(true);
	});
});
