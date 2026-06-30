'use client';

import * as React from 'react';

interface HeaderActionsContextValue {
  actions: React.ReactNode;
  setActions: (node: React.ReactNode) => void;
}

const HeaderActionsContext =
  React.createContext<HeaderActionsContextValue | null>(null);

export function HeaderActionsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [actions, setActions] = React.useState<React.ReactNode>(null);
  const value = React.useMemo(() => ({ actions, setActions }), [actions]);
  return (
    <HeaderActionsContext.Provider value={value}>
      {children}
    </HeaderActionsContext.Provider>
  );
}

export function useHeaderActions(): HeaderActionsContextValue {
  const ctx = React.useContext(HeaderActionsContext);
  if (!ctx) {
    throw new Error(
      'useHeaderActions must be used within a HeaderActionsProvider',
    );
  }
  return ctx;
}
