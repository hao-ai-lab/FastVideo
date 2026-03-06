"use client";

import { createContext, useCallback, useContext, useState } from "react";
import type { Job } from "@/lib/types";

interface ActiveJobContextValue {
	activeJobId: string | null;
	setActiveJobId: (id: string | null) => void;
	activeJob: Job | null;
	setActiveJob: (job: Job | null) => void;
}

const ActiveJobContext = createContext<ActiveJobContextValue | undefined>(
	undefined,
);

export function ActiveJobProvider({ children }: { children: React.ReactNode }) {
	const [activeJobId, setActiveJobIdState] = useState<string | null>(null);
	const [activeJob, setActiveJobState] = useState<Job | null>(null);

	const setActiveJobId = useCallback((id: string | null) => {
		setActiveJobIdState(id);
		if (!id) setActiveJobState(null);
	}, []);

	const setActiveJob = useCallback((job: Job | null) => {
		setActiveJobState(job);
	}, []);

	const value: ActiveJobContextValue = {
		activeJobId,
		setActiveJobId,
		activeJob,
		setActiveJob,
	};

	return (
		<ActiveJobContext.Provider value={value}>
			{children}
		</ActiveJobContext.Provider>
	);
}

export function useActiveJob(): ActiveJobContextValue {
	const ctx = useContext(ActiveJobContext);
	if (!ctx) {
		throw new Error("useActiveJob must be used within ActiveJobProvider");
	}
	return ctx;
}
