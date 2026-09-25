"use client";

import { useCallback, useEffect, useState } from "react";
import { isGenerationMode, type GenerationCapabilities } from "@/lib/generationMode";

const LEGACY_CAPABILITIES: GenerationCapabilities = { model_id: "legacy", modes: ["t2va"] };

export function useGenerationCapabilities() {
	const [capabilities, setCapabilities] = useState<GenerationCapabilities>(LEGACY_CAPABILITIES);
	const [capabilityNotice, setCapabilityNotice] = useState("");
	const [loadingCapabilities, setLoadingCapabilities] = useState(true);
	const refreshCapabilities = useCallback(async () => {
		try {
			const response = await fetch("/generation-capabilities", { signal: AbortSignal.timeout(4000) });
			if (!response.ok) throw new Error("Capabilities unavailable");
			const payload = await response.json();
			if (typeof payload.model_id !== "string" || !Array.isArray(payload.modes)
				|| !payload.modes.every(isGenerationMode)) throw new Error("Invalid capabilities");
			const next: GenerationCapabilities = {
				model_id: payload.model_id,
				modes: payload.modes,
				mock: payload.mock === true,
			};
			setCapabilities(next);
			setCapabilityNotice("");
			return next;
		} catch {
			setCapabilities(LEGACY_CAPABILITIES);
			setCapabilityNotice("Runtime capabilities unavailable. Text-only compatibility mode is available; check the backend to enable image and reference modes.");
			return LEGACY_CAPABILITIES;
		} finally {
			setLoadingCapabilities(false);
		}
	}, []);
	useEffect(() => { void refreshCapabilities(); }, [refreshCapabilities]);
	return { capabilities, capabilityNotice, loadingCapabilities, refreshCapabilities };
}
