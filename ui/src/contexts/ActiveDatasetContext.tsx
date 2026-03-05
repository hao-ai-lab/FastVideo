'use client';

import { createContext, useCallback, useContext, useState } from "react";
import type { Dataset } from "@/lib/api";

interface ActiveDatasetContextValue {
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null) => void;
  activeDataset: Dataset | null;
  setActiveDataset: (dataset: Dataset | null) => void;
}

const ActiveDatasetContext = createContext<ActiveDatasetContextValue | undefined>(
  undefined
);

export function ActiveDatasetProvider({ children }: { children: React.ReactNode }) {
  const [activeDatasetId, setActiveDatasetIdState] = useState<string | null>(null);
  const [activeDataset, setActiveDatasetState] = useState<Dataset | null>(null);

  const setActiveDatasetId = useCallback((id: string | null) => {
    setActiveDatasetIdState(id);
    if (!id) setActiveDatasetState(null);
  }, []);

  const setActiveDataset = useCallback((dataset: Dataset | null) => {
    setActiveDatasetState(dataset);
  }, []);

  const value: ActiveDatasetContextValue = {
    activeDatasetId,
    setActiveDatasetId,
    activeDataset,
    setActiveDataset,
  };

  return (
    <ActiveDatasetContext.Provider value={value}>
      {children}
    </ActiveDatasetContext.Provider>
  );
}

export function useActiveDataset(): ActiveDatasetContextValue {
  const ctx = useContext(ActiveDatasetContext);
  if (!ctx) {
    throw new Error("useActiveDataset must be used within ActiveDatasetProvider");
  }
  return ctx;
}
