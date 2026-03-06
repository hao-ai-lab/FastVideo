"use client";

import {
	createContext,
	useCallback,
	useContext,
	useEffect,
	useMemo,
	useState,
} from "react";
import {
	DEFAULT_OPTIONS,
	DefaultOptions,
	loadDefaultOptions,
	saveDefaultOptions,
} from "@/lib/defaultOptions";
import { getSettings, updateSettings } from "@/lib/api";

interface DefaultOptionsContextValue {
	options: DefaultOptions;
	updateOption: <K extends keyof DefaultOptions>(
		key: K,
		value: DefaultOptions[K],
	) => void;
	resetToDefaults: () => void;
}

const DefaultOptionsContext = createContext<
	DefaultOptionsContextValue | undefined
>(undefined);

export function DefaultOptionsProvider({
	children,
}: {
	children: React.ReactNode;
}) {
	const [options, setOptions] = useState<DefaultOptions>(loadDefaultOptions);

	useEffect(() => {
		getSettings()
			.then(setOptions)
			.catch(() => {
				setOptions(loadDefaultOptions());
			});
	}, []);

	const updateOption = useCallback(
		<K extends keyof DefaultOptions>(key: K, value: DefaultOptions[K]) => {
			setOptions((prev) => {
				const next = { ...prev, [key]: value };
				updateSettings({ [key]: value }).catch(() => {
					saveDefaultOptions(next);
				});
				return next;
			});
		},
		[],
	);

	const resetToDefaults = useCallback(() => {
		setOptions(DEFAULT_OPTIONS);
		updateSettings(DEFAULT_OPTIONS).catch(() => {
			saveDefaultOptions(DEFAULT_OPTIONS);
		});
	}, []);

	const value = useMemo(
		() => ({ options, updateOption, resetToDefaults }),
		[options, updateOption, resetToDefaults],
	);

	return (
		<DefaultOptionsContext.Provider value={value}>
			{children}
		</DefaultOptionsContext.Provider>
	);
}

export function useDefaultOptions(): DefaultOptionsContextValue {
	const ctx = useContext(DefaultOptionsContext);
	if (!ctx) {
		throw new Error(
			"useDefaultOptions must be used within DefaultOptionsProvider",
		);
	}
	return ctx;
}
