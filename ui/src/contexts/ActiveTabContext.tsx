'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

export type ActiveTab =
  | "inference"
  | "finetuning"
  | "distillation"
  | "lora"
  | "settings";

interface ActiveTabContextValue {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  headerActions: React.ReactNode[];
  setHeaderActions: (actions: React.ReactNode[]) => void;
}

const ActiveTabContext = createContext<ActiveTabContextValue | undefined>(
  undefined
);

const TAB_TITLES: Record<ActiveTab, string> = {
  inference: "Inference",
  finetuning: "Finetuning",
  distillation: "Distillation",
  lora: "LoRA",
  settings: "Settings",
};

export function ActiveTabProvider({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState<ActiveTab>("inference");
  const [headerActions, setHeaderActionsState] = useState<React.ReactNode[]>(
    []
  );

  const setHeaderActions = useCallback((actions: React.ReactNode[]) => {
    setHeaderActionsState(actions);
  }, []);

  const value: ActiveTabContextValue = {
    activeTab,
    setActiveTab,
    headerActions,
    setHeaderActions,
  };

  return (
    <ActiveTabContext.Provider value={value}>
      {children}
    </ActiveTabContext.Provider>
  );
}

export function useActiveTab(): ActiveTabContextValue {
  const ctx = useContext(ActiveTabContext);
  if (!ctx) {
    throw new Error("useActiveTab must be used within ActiveTabProvider");
  }
  return ctx;
}

export function useHeaderTitle(): string {
  const { activeTab } = useActiveTab();
  return TAB_TITLES[activeTab];
}

/**
 * Registers header action components for the current view. Actions are
 * displayed in the Header and cleared when the view unmounts.
 */
export function useHeaderActions(actions: React.ReactNode[]): void {
  const { setHeaderActions } = useActiveTab();

  useEffect(() => {
    setHeaderActions(actions);
    return () => setHeaderActions([]);
  }, [actions, setHeaderActions]);
}
