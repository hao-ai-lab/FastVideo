"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ConditioningAsset, ConditioningRole, GenerationAsset } from "@/lib/generationMode";

const LIBRARY_KEY = "dreamverse-asset-library-v1";
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

async function responseError(response: Response, fallback: string): Promise<Error> {
	const payload = await response.json().catch(() => ({}));
	return new Error(typeof payload.detail === "string" ? payload.detail : fallback);
}

function isAsset(value: unknown): value is GenerationAsset {
	if (!value || typeof value !== "object") return false;
	const asset = value as Partial<GenerationAsset>;
	return typeof asset.asset_id === "string" && /^[a-zA-Z0-9_-]+$/.test(asset.asset_id)
		&& ["image", "video", "audio"].includes(asset.kind || "")
		&& typeof asset.name === "string" && typeof asset.mime_type === "string" && typeof asset.size === "number";
}

/** Asset ownership lives here so composers and other pickers share the same library. */
export function useAssetLibrary() {
	const [assets, setAssets] = useState<GenerationAsset[]>([]);
	const [conditioningAssets, setConditioningAssets] = useState<ConditioningAsset[]>([]);
	const [uploading, setUploading] = useState(false);
	const [assetError, setAssetError] = useState("");
	const [hydrated, setHydrated] = useState(false);
	const uploadingRef = useRef(false);

	useEffect(() => {
		try {
			const saved = JSON.parse(localStorage.getItem(LIBRARY_KEY) || "[]");
			if (Array.isArray(saved)) setAssets(saved.filter(isAsset).map((asset) => ({
				...asset, url: `/assets/${asset.asset_id}`,
			})));
		} catch { /* Storage is optional; uploads still work in private browsing. */ }
		setHydrated(true);
	}, []);
	useEffect(() => {
		if (!hydrated) return;
		try { localStorage.setItem(LIBRARY_KEY, JSON.stringify(assets)); } catch { /* Optional cache. */ }
	}, [assets, hydrated]);

	const uploadAssets = useCallback(async (files: File[]) => {
		if (uploadingRef.current) return;
		uploadingRef.current = true;
		setUploading(true);
		setAssetError("");
		const errors: string[] = [];
		for (const file of files) {
			try {
				if (!/^(image|video|audio)\//.test(file.type)) throw new Error(`${file.name}: choose an image, video, or audio file.`);
				const maxBytes = file.type.startsWith("image/") ? 15 * 1024 * 1024 : MAX_UPLOAD_BYTES;
				if (!file.size || file.size > maxBytes) throw new Error(`${file.name}: use a non-empty file up to ${maxBytes / 1024 / 1024} MiB.`);
				const response = await fetch("/assets", {
					method: "POST",
					headers: { "Content-Type": file.type, "X-Asset-Name": encodeURIComponent(file.name) },
					body: file,
				});
				if (!response.ok) throw await responseError(response, `Could not upload ${file.name}.`);
				const asset: unknown = await response.json();
				if (!isAsset(asset)) throw new Error("The server returned an invalid asset. Please retry the upload.");
				setAssets((current) => [...current.filter((item) => item.asset_id !== asset.asset_id), {
					...asset, url: `/assets/${asset.asset_id}`, missing: false,
				}]);
			} catch (error) {
				errors.push(error instanceof Error ? error.message : `Could not upload ${file.name}.`);
			}
		}
		setAssetError(errors.join(" "));
		setUploading(false);
		uploadingRef.current = false;
	}, []);

	const assignAsset = useCallback((assetId: string, role: ConditioningRole) => {
		setConditioningAssets((current) => {
			const next = role === "reference" ? current : current.filter((item) => item.role !== role);
			if (!assetId || next.some((item) => item.asset_id === assetId && item.role === role)) return next;
			return [...next, { asset_id: assetId, role }];
		});
	}, []);
	const removeConditioning = useCallback((index: number) => {
		setConditioningAssets((current) => current.filter((_, itemIndex) => itemIndex !== index));
	}, []);
	const moveConditioning = useCallback((from: number, to: number) => {
		setConditioningAssets((current) => {
			if (from < 0 || to < 0 || from >= current.length || to >= current.length) return current;
			const next = [...current];
			next.splice(to, 0, next.splice(from, 1)[0]);
			return next;
		});
	}, []);
	const clearConditioning = useCallback(() => setConditioningAssets([]), []);
	const markAssetMissing = useCallback((assetId: string) => {
		setAssets((current) => current.map((asset) => asset.asset_id === assetId ? { ...asset, missing: true } : asset));
	}, []);
	const checkAssetAvailability = useCallback(async (assetId: string) => {
		try {
			const response = await fetch(`/assets/${assetId}`, { method: "HEAD", signal: AbortSignal.timeout(4000) });
			if (response.status === 404) markAssetMissing(assetId);
		} catch { /* A browser preview failure alone does not mean the upload expired. */ }
	}, [markAssetMissing]);
	const removeAsset = useCallback(async (assetId: string) => {
		setAssetError("");
		try {
			const response = await fetch(`/assets/${assetId}`, { method: "DELETE" });
			if (!response.ok && response.status !== 404) throw await responseError(response, "Could not remove the asset. Retry when the backend is available.");
			setAssets((current) => current.filter((asset) => asset.asset_id !== assetId));
			setConditioningAssets((current) => current.filter((asset) => asset.asset_id !== assetId));
		} catch (error) {
			setAssetError(error instanceof Error ? error.message : "Could not remove the asset.");
		}
	}, []);
	const verifySelectedAssets = useCallback(async (): Promise<string | null> => {
		const selected = [...new Set(conditioningAssets.map((item) => item.asset_id))];
		try {
			for (const assetId of selected) {
				const response = await fetch(`/assets/${assetId}`, { method: "HEAD", signal: AbortSignal.timeout(4000) });
				if (response.status === 404) {
					markAssetMissing(assetId);
					return "A selected asset expired or was removed from the server. Upload it again, then select the new copy.";
				}
				if (!response.ok) return "Could not verify the selected assets. Check the backend and try again.";
			}
			return null;
		} catch {
			return "Could not verify the selected assets. Check the backend and try again.";
		}
	}, [conditioningAssets, markAssetMissing]);

	return {
		assets, conditioningAssets, uploading, assetError, uploadAssets, assignAsset,
		removeConditioning, moveConditioning, clearConditioning, removeAsset, checkAssetAvailability, verifySelectedAssets,
	};
}
