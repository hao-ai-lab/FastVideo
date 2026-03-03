'use client';

import { createContext, useContext, useState } from "react";

export type ActiveTab = "job-queue" | "settings";

interface ActiveTabContextValue {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
}

const ActiveTabContext = createContext<ActiveTabContextValue | undefined>(
  undefined
);

const TAB_TITLES: Record<ActiveTab, string> = {
  "job-queue": "Jobs",
  settings: "Settings",
};

export function ActiveTabProvider({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState<ActiveTab>("job-queue");

  const value: ActiveTabContextValue = {
    activeTab,
    setActiveTab,
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
