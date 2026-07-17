'use client';

import * as React from 'react';

export type ResizableEdge = 'left' | 'right';

export interface UseResizableOptions {
  /** Which edge is fixed: "left" = drag right grows; "right" = drag left grows. */
  edge: ResizableEdge;
  minWidth: number;
  maxWidth: number;
  /** Return current width (e.g. from parent state). */
  getWidth: () => number;
  /** Called with the new width while the user drags. */
  onWidth: (width: number) => void;
  /** Optional: notified when a drag starts/stops. */
  onDragChange?: (dragging: boolean) => void;
}

export interface UseResizableResult {
  onMouseDown: (event: React.MouseEvent) => void;
}

/**
 * React port of the Svelte `resizable` action. Spread the returned
 * `onMouseDown` onto a drag handle; the parent owns the width state.
 */
export function useResizable(
  options: UseResizableOptions,
): UseResizableResult {
  const optionsRef = React.useRef(options);
  optionsRef.current = options;

  const dragStart = React.useRef({ x: 0, width: 0 });
  const dragging = React.useRef(false);

  const onMouseMove = React.useCallback((e: MouseEvent) => {
    const opts = optionsRef.current;
    const delta = e.clientX - dragStart.current.x;
    const sign = opts.edge === 'left' ? 1 : -1;
    const newWidth = Math.min(
      opts.maxWidth,
      Math.max(opts.minWidth, dragStart.current.width + sign * delta),
    );
    opts.onWidth(newWidth);
  }, []);

  const onMouseUp = React.useCallback(() => {
    dragging.current = false;
    optionsRef.current.onDragChange?.(false);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
  }, [onMouseMove]);

  const onMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const opts = optionsRef.current;
      dragStart.current = { x: e.clientX, width: opts.getWidth() };
      dragging.current = true;
      opts.onDragChange?.(true);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    },
    [onMouseMove, onMouseUp],
  );

  React.useEffect(() => {
    return () => {
      if (dragging.current) {
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
      }
    };
  }, [onMouseMove, onMouseUp]);

  return { onMouseDown };
}
