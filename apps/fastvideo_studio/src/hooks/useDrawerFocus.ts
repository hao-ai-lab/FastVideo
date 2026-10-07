'use client';

import * as React from 'react';

/**
 * Move focus into a drawer when it opens as a modal (mobile), and hand it
 * back to the previously focused element on close. Pairs with `inert` on
 * the background content — together they make `aria-modal` truthful.
 */
export function useDrawerFocus<T extends HTMLElement>(active: boolean) {
  const ref = React.useRef<T | null>(null);

  React.useEffect(() => {
    if (!active) return;
    const previous = document.activeElement;
    ref.current?.focus();
    return () => {
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [active]);

  return ref;
}
