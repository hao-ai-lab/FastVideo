'use client';

import * as React from 'react';
import { usePathname } from 'next/navigation';

import DatasetSidebar from '@/components/DatasetSidebar';
import Header from '@/components/Header';
import { HeaderActionsProvider } from '@/components/HeaderActionsContext';
import PrimarySidebar from '@/components/PrimarySidebar';
import SecondarySidebar from '@/components/SecondarySidebar';
import { Toaster } from '@/components/ui/sonner';
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

  const [primaryWidth, setPrimaryWidth] = React.useState(220);
  const [secondaryWidth, setSecondaryWidth] = React.useState(0);

  const jobSidebarOpen = JOB_ROUTES.includes(pathname) && activeJob != null;
  const datasetSidebarOpen =
    pathname === '/datasets' && activeDataset != null;
  const secondaryOpen = jobSidebarOpen || datasetSidebarOpen;

  React.useEffect(() => {
    initDefaultOptions();
  }, []);

  React.useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape' && !document.querySelector('[data-modal]')) {
        if (activeJobStore.get().activeJob) setActiveJobId(null);
        if (activeDatasetStore.get().activeDataset) setActiveDatasetId(null);
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <HeaderActionsProvider>
      <Header />
      <div
        className="flex overflow-hidden"
        style={{
          marginTop: 'var(--header-height)',
          height: 'calc(100vh - var(--header-height))',
        }}
      >
        <PrimarySidebar onWidthChange={setPrimaryWidth} />
        <main
          className="flex min-w-0 flex-1 flex-col overflow-auto"
          style={{
            marginLeft: primaryWidth,
            marginRight: secondaryOpen ? secondaryWidth : 0,
          }}
        >
          {children}
        </main>
        {jobSidebarOpen && activeJob && (
          <SecondarySidebar
            job={activeJob}
            onClose={() => setActiveJobId(null)}
            onWidthChange={setSecondaryWidth}
          />
        )}
        {datasetSidebarOpen && activeDataset && (
          <DatasetSidebar
            dataset={activeDataset}
            onClose={() => setActiveDatasetId(null)}
            onWidthChange={setSecondaryWidth}
          />
        )}
      </div>
      <Toaster />
    </HeaderActionsProvider>
  );
}
