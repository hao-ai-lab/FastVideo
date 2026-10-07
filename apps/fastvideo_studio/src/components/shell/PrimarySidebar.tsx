'use client';

import * as React from 'react';
import { X } from 'lucide-react';
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
  'block min-h-11 px-5 py-[0.65rem] text-left text-sm text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground';
const TAB_ACTIVE = 'bg-accent-blue/10 font-medium text-accent-blue';

export default function PrimarySidebar({
  isMobile,
  mobileOpen,
  onMobileClose,
  onWidthChange,
}: {
  isMobile: boolean;
  mobileOpen: boolean;
  onMobileClose: () => void;
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
    onWidthChange?.(isMobile ? 0 : layoutWidth);
  }, [isMobile, layoutWidth, onWidthChange]);

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
      id="primary-navigation"
      aria-hidden={isMobile && !mobileOpen}
      inert={isMobile && !mobileOpen ? true : undefined}
      className={cn(
        'fixed bottom-0 left-0 top-[var(--header-height)] z-50 flex max-h-[calc(100dvh-var(--header-height))] shrink-0 flex-col border-r border-border bg-card transition-transform duration-200 md:translate-x-0',
        mobileOpen ? 'translate-x-0' : '-translate-x-full',
      )}
      style={{
        width: isMobile
          ? 'min(18rem, calc(100vw - 3rem))'
          : effectiveWidth,
      }}
    >
      {isMobile && (
        <div className="flex h-14 items-center justify-between border-b border-border px-4">
          <span className="text-sm font-semibold">Navigation</span>
          <button
            type="button"
            onClick={onMobileClose}
            aria-label="Close navigation"
            className="flex size-11 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <X className="size-5" aria-hidden />
          </button>
        </div>
      )}
      {!isCollapsed && (
        <nav
          aria-label="Primary navigation"
          className="flex flex-col overflow-y-auto py-2"
        >
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
                    aria-current={pathname === route.href ? 'page' : undefined}
                    onClick={onMobileClose}
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
            aria-current={pathname === '/datasets' ? 'page' : undefined}
            onClick={onMobileClose}
            className={cn(TAB_BASE, pathname === '/datasets' && TAB_ACTIVE)}
          >
            Datasets
          </Link>
          <Link
            href="/gallery"
            aria-current={pathname === '/gallery' ? 'page' : undefined}
            onClick={onMobileClose}
            className={cn(TAB_BASE, pathname === '/gallery' && TAB_ACTIVE)}
          >
            Gallery
          </Link>
          <Link
            href="/gpus"
            aria-current={pathname === '/gpus' ? 'page' : undefined}
            onClick={onMobileClose}
            className={cn(TAB_BASE, pathname === '/gpus' && TAB_ACTIVE)}
          >
            GPUs
          </Link>
          <Link
            href="/settings"
            aria-current={pathname === '/settings' ? 'page' : undefined}
            onClick={onMobileClose}
            className={cn(TAB_BASE, pathname === '/settings' && TAB_ACTIVE)}
          >
            Settings
          </Link>
        </nav>
      )}

      {!isMobile && <div
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
            'flex size-11 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground',
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
      </div>}

      {!isMobile && !isCollapsed && (
        <div
          role="presentation"
          onMouseDown={onMouseDown}
          className={cn(
            'absolute bottom-0 right-0 top-0 z-[1] w-1.5 cursor-col-resize hover:bg-accent-blue/25',
            isDragging && 'bg-accent-blue/25',
          )}
        />
      )}
    </aside>
  );
}
