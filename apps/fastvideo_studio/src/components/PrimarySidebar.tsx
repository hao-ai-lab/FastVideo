'use client';

import * as React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { useResizable } from '@/hooks/useResizable';
import { cn } from '@/lib/utils';

const SIDEBAR_MIN_WIDTH = 100;
const SIDEBAR_MAX_WIDTH = 300;
const SIDEBAR_COLLAPSED_WIDTH = 0;
const SIDEBAR_COLLAPSED_VISIBLE_WIDTH = 60;

const JOB_ROUTES = [
  { href: '/inference', label: 'Inference' },
  { href: '/finetuning', label: 'Finetuning' },
  { href: '/distillation', label: 'Distillation' },
] as const;

const TAB_BASE =
  'block px-5 py-[0.65rem] text-left text-sm text-muted-foreground transition-colors hover:bg-accent/5 hover:text-foreground';
const TAB_ACTIVE = 'bg-accent/10 font-medium text-accent';

export default function PrimarySidebar({
  onWidthChange,
}: {
  onWidthChange?: (w: number) => void;
}) {
  const pathname = usePathname();
  const [width, setWidth] = React.useState(220);
  const [isCollapsed, setIsCollapsed] = React.useState(false);
  const [isDragging, setIsDragging] = React.useState(false);
  const [jobsOpen, setJobsOpen] = React.useState(true);

  const effectiveWidth = isCollapsed ? SIDEBAR_COLLAPSED_WIDTH : width;
  const layoutWidth = isCollapsed ? SIDEBAR_COLLAPSED_VISIBLE_WIDTH : width;
  const isJobsActive = JOB_ROUTES.some((r) => pathname === r.href);

  React.useEffect(() => {
    onWidthChange?.(layoutWidth);
  }, [layoutWidth, onWidthChange]);

  React.useEffect(() => {
    if (JOB_ROUTES.some((r) => pathname === r.href)) {
      setJobsOpen(true);
    }
  }, [pathname]);

  const { onMouseDown } = useResizable({
    edge: 'left',
    minWidth: SIDEBAR_MIN_WIDTH,
    maxWidth: SIDEBAR_MAX_WIDTH,
    getWidth: () => width,
    onWidth: setWidth,
    onDragChange: setIsDragging,
  });

  return (
    <aside
      className="fixed bottom-0 left-0 top-[var(--header-height)] z-50 flex max-h-[calc(100vh-var(--header-height))] shrink-0 flex-col border-r border-border bg-card"
      style={{ width: effectiveWidth }}
    >
      {!isCollapsed && (
        <nav className="flex flex-col py-2">
          <div className="flex flex-col">
            <button
              type="button"
              onClick={() => setJobsOpen((v) => !v)}
              aria-expanded={jobsOpen}
              aria-haspopup="true"
              className={cn(
                TAB_BASE,
                'flex w-full cursor-pointer items-center justify-between',
                isJobsActive && TAB_ACTIVE,
              )}
            >
              <span>Jobs</span>
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                className={cn(
                  'h-4 w-4 shrink-0 opacity-85 transition-transform',
                  jobsOpen && 'rotate-180',
                )}
              >
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>
            {jobsOpen && (
              <div className="mb-1 ml-4 flex flex-col border-l-2 border-border pl-2">
                {JOB_ROUTES.map((route) => (
                  <Link
                    key={route.href}
                    href={route.href}
                    className={cn(
                      TAB_BASE,
                      'px-4 py-2 text-[0.85rem]',
                      pathname === route.href && TAB_ACTIVE,
                    )}
                  >
                    {route.label}
                  </Link>
                ))}
              </div>
            )}
          </div>
          <Link
            href="/datasets"
            className={cn(TAB_BASE, pathname === '/datasets' && TAB_ACTIVE)}
          >
            Datasets
          </Link>
          <Link
            href="/gallery"
            className={cn(TAB_BASE, pathname === '/gallery' && TAB_ACTIVE)}
          >
            Gallery
          </Link>
          <Link
            href="/settings"
            className={cn(TAB_BASE, pathname === '/settings' && TAB_ACTIVE)}
          >
            Settings
          </Link>
        </nav>
      )}

      <div
        className={cn(
          'absolute bottom-0 p-2',
          isCollapsed ? '-right-[60px] top-0' : 'right-0',
        )}
      >
        <button
          type="button"
          onClick={() => setIsCollapsed((v) => !v)}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className={cn(
            'flex items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent/10 hover:text-foreground',
            isCollapsed ? 'p-3' : 'p-2',
          )}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            className="h-[18px] w-[18px]"
          >
            <path d={isCollapsed ? 'M9 18l6-6-6-6' : 'M15 18l-6-6 6-6'} />
          </svg>
        </button>
      </div>

      {!isCollapsed && (
        <div
          role="presentation"
          onMouseDown={onMouseDown}
          className={cn(
            'absolute bottom-0 right-0 top-0 z-[1] w-1.5 cursor-col-resize hover:bg-accent/20',
            isDragging && 'bg-accent/20',
          )}
        />
      )}
    </aside>
  );
}
