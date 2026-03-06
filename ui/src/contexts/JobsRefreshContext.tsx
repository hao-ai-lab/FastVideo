"use client";

import { createContext, useCallback, useContext, useRef } from "react";

type RefreshCallback = () => void;

interface JobsRefreshContextValue {
	registerRefresh: (cb: RefreshCallback) => () => void;
	triggerRefresh: () => void;
}

const JobsRefreshContext = createContext<JobsRefreshContextValue | undefined>(
	undefined,
);

export function JobsRefreshProvider({
	children,
}: {
	children: React.ReactNode;
}) {
	const refreshRef = useRef<RefreshCallback | null>(null);

	const registerRefresh = useCallback((cb: RefreshCallback) => {
		refreshRef.current = cb;
		return () => {
			refreshRef.current = null;
		};
	}, []);

	const triggerRefresh = useCallback(() => {
		refreshRef.current?.();
	}, []);

	const value: JobsRefreshContextValue = {
		registerRefresh,
		triggerRefresh,
	};

	return (
		<JobsRefreshContext.Provider value={value}>
			{children}
		</JobsRefreshContext.Provider>
	);
}

export function useJobsRefresh(): JobsRefreshContextValue {
	const ctx = useContext(JobsRefreshContext);
	if (!ctx) {
		throw new Error(
			"useJobsRefresh must be used within JobsRefreshProvider",
		);
	}
	return ctx;
}
