'use client';

import * as React from 'react';
import { usePathname } from 'next/navigation';

import DatasetSidebar from '@/components/datasets/DatasetSidebar';
import Header from '@/components/shell/Header';
import { HeaderActionsProvider } from '@/components/shell/HeaderActionsContext';
import PrimarySidebar from '@/components/shell/PrimarySidebar';
import JobDetailsSidebar from '@/components/jobs/JobDetailsSidebar';
import { Toaster } from '@/components/ui/sonner';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { useStore } from '@/hooks/useStore';
import {
  activeDatasetStore,
  setActiveDatasetId,
} from '@/stores/activeDataset';
import { activeJobStore, setActiveJobId } from '@/stores/activeJob';
import { initDefaultOptions } from '@/stores/defaultOptions';

const JOB_ROUTES = ['/inference', '/finetuning', '/distillation'];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { activeJob } = useStore(activeJobStore);
  const { activeDataset } = useStore(activeDatasetStore);
  const isMobile = useMediaQuery('(max-width: 767px)');

  const [primaryWidth, setPrimaryWidth] = React.useState(220);
  const [secondaryWidth, setSecondaryWidth] = React.useState(0);
  const [primaryOpen, setPrimaryOpen] = React.useState(false);

  const jobSidebarOpen = JOB_ROUTES.includes(pathname) && activeJob != null;
  const datasetSidebarOpen =
    pathname === '/datasets' && activeDataset != null;
  const secondaryOpen = jobSidebarOpen || datasetSidebarOpen;
  // Mobile detail drawers claim aria-modal, so everything behind them must
  // actually be inert — the platform enforces what the ARIA claims.
  const drawerModal = isMobile && secondaryOpen;

  React.useEffect(() => {
    initDefaultOptions();
  }, []);

  React.useEffect(() => {
    setPrimaryOpen(false);
  }, [pathname]);

  React.useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key !== 'Escape' || document.querySelector('[data-modal]')) return;
      if (primaryOpen) {
        setPrimaryOpen(false);
        return;
      }
      if (activeJobStore.get().activeJob) setActiveJobId(null);
      if (activeDatasetStore.get().activeDataset) setActiveDatasetId(null);
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [primaryOpen]);

  return (
    <HeaderActionsProvider>
      <div
        style={{ display: 'contents' }}
        inert={drawerModal ? true : undefined}
      >
        <Header
          navigationOpen={primaryOpen}
          onNavigationToggle={() => setPrimaryOpen((open) => !open)}
        />
      </div>
      <div
        className="flex overflow-hidden"
        style={{
          marginTop: 'var(--header-height)',
          height: 'calc(100dvh - var(--header-height))',
        }}
      >
        <PrimarySidebar
          isMobile={isMobile}
          mobileOpen={primaryOpen}
          onMobileClose={() => setPrimaryOpen(false)}
          onWidthChange={setPrimaryWidth}
        />
        {primaryOpen && (
          <button
            type="button"
            aria-label="Close navigation"
            onClick={() => setPrimaryOpen(false)}
            className="fixed inset-x-0 bottom-0 top-[var(--header-height)] z-40 bg-black/55 md:hidden"
          />
        )}
        <main
          className="flex min-w-0 flex-1 flex-col overflow-auto"
          inert={drawerModal ? true : undefined}
          style={{
            marginLeft: isMobile ? 0 : primaryWidth,
            marginRight: isMobile || !secondaryOpen ? 0 : secondaryWidth,
          }}
        >
          {children}
        </main>
        {jobSidebarOpen && activeJob && (
          <JobDetailsSidebar
            job={activeJob}
            isMobile={isMobile}
            onClose={() => setActiveJobId(null)}
            onWidthChange={setSecondaryWidth}
          />
        )}
        {datasetSidebarOpen && activeDataset && (
          <DatasetSidebar
            dataset={activeDataset}
            isMobile={isMobile}
            onClose={() => setActiveDatasetId(null)}
            onWidthChange={setSecondaryWidth}
          />
        )}
      </div>
      <Toaster />
    </HeaderActionsProvider>
  );
}
