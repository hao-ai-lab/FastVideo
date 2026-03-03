'use client';

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import {
  DEFAULT_OPTIONS,
  DefaultOptions,
  loadDefaultOptions,
  saveDefaultOptions,
} from "@/lib/defaultOptions";

interface DefaultOptionsContextValue {
  options: DefaultOptions;
  updateOption: <K extends keyof DefaultOptions>(
    key: K,
    value: DefaultOptions[K]
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
  const [options, setOptions] = useState(loadDefaultOptions);

  const updateOption = useCallback(
    <K extends keyof DefaultOptions>(key: K, value: DefaultOptions[K]) => {
      setOptions((prev) => {
        const next = { ...prev, [key]: value };
        saveDefaultOptions(next);
        return next;
      });
    },
    []
  );

  const resetToDefaults = useCallback(() => {
    setOptions(DEFAULT_OPTIONS);
    saveDefaultOptions(DEFAULT_OPTIONS);
  }, []);

  const value = useMemo(
    () => ({ options, updateOption, resetToDefaults }),
    [options, updateOption, resetToDefaults]
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
      "useDefaultOptions must be used within DefaultOptionsProvider"
    );
  }
  return ctx;
}
