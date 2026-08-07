'use client';

import * as React from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { getEngineLogs } from '@/lib/api';

const POLL_INTERVAL_MS = 2000;
// Cap the DOM at the last ~500 lines; the server keeps its own ring buffer.
const MAX_LINES = 500;

/**
 * Collapsible tail of the engine's stdout/stderr (driver + relayed worker
 * output). Polls only while open; sticks to the bottom unless the user has
 * scrolled up.
 */
export default function EngineConsole() {
  const [open, setOpen] = React.useState(false);
  const [lines, setLines] = React.useState<string[]>([]);
  // Poll cursor + stick-to-bottom flag live in refs: they must update
  // synchronously from async polls / scroll events, outside React's cycle.
  const afterRef = React.useRef(0);
  const stickRef = React.useRef(true);
  const consoleRef = React.useRef<HTMLPreElement | null>(null);

  React.useEffect(() => {
    if (!open) return;
    let mounted = true;
    let locked = false;

    async function poll() {
      if (!mounted || locked) return;
      locked = true;
      try {
        const data = await getEngineLogs(afterRef.current);
        afterRef.current = data.total;
        if (mounted && data.lines.length > 0) {
          setLines((prev) => [...prev, ...data.lines].slice(-MAX_LINES));
        }
      } catch (e) {
        console.error('Failed to fetch engine logs:', e);
      } finally {
        locked = false;
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [open]);

  // Follow the tail after new lines land, unless the user scrolled up.
  React.useEffect(() => {
    const el = consoleRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  function handleScroll() {
    const el = consoleRef.current;
    if (!el) return;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  return (
    <section
      aria-label="Engine output"
      className="mx-auto w-full max-w-[850px] px-10 pt-3"
    >
      <div className="rounded-lg border border-border bg-background">
        <div className="flex items-center gap-2 px-2 py-1.5">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="flex flex-1 items-center gap-2 rounded-md px-2 py-1 text-sm font-semibold text-foreground hover:bg-accent"
          >
            {open ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            Engine output
          </button>
          {open && (
            <Button
              size="sm"
              variant="ghost"
              // Resets the local view only; the server buffer is untouched.
              onClick={() => setLines([])}
            >
              Clear view
            </Button>
          )}
        </div>
        {open && (
          <pre
            ref={consoleRef}
            onScroll={handleScroll}
            className="m-0 h-64 overflow-auto whitespace-pre-wrap break-words rounded-b-lg border-t border-border bg-zinc-950 p-3 font-mono text-xs leading-normal text-zinc-200"
          >
            {lines.length === 0 ? (
              <span className="italic text-zinc-500">
                Waiting for engine output…
              </span>
            ) : (
              lines.join('\n')
            )}
          </pre>
        )}
      </div>
    </section>
  );
}
