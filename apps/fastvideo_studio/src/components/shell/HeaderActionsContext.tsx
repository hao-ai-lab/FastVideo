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

/**
 * Declaratively publish the current page's header actions: renders nothing,
 * registers `children` in the header on mount and clears them on unmount.
 * Pages without actions simply don't render it.
 */
export function HeaderActions({ children }: { children: React.ReactNode }) {
  const { setActions } = useHeaderActions();
  // Mount-only on purpose: pages pass inline JSX, which is referentially new
  // every render and would re-register per render if it were a dependency.
  React.useEffect(() => {
    setActions(children);
    return () => setActions(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setActions]);
  return null;
}
