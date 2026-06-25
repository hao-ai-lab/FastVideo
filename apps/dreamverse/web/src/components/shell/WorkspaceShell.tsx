'use client';

import React from 'react';

import { cn } from '@/lib/utils';

interface WorkspaceShellProps {
  devtools?: boolean;
  topbar?: React.ReactNode;
  workspace?: React.ReactNode;
  sidebar?: React.ReactNode;
  composer?: React.ReactNode;
  alerts?: React.ReactNode;
  drawer?: React.ReactNode;
}

export default function WorkspaceShell({
  devtools = false,
  topbar,
  workspace,
  sidebar,
  composer,
  alerts,
  drawer,
}: WorkspaceShellProps) {
  return (
    <main
      className={cn(
        'mx-auto flex min-h-screen w-full max-w-[1880px] flex-col gap-4 px-4 py-4 sm:px-6',
        devtools && 'max-w-[1960px]',
      )}
    >
      {topbar}
      <div
        className={cn(
          'flex min-h-0 flex-1 flex-col gap-4 xl:flex-row',
          sidebar && 'xl:items-stretch',
        )}
      >
        <div className="min-h-0 flex-1">{workspace}</div>
        {sidebar && (
          <div className="min-h-0 w-full xl:max-w-[380px]">{sidebar}</div>
        )}
      </div>
      {composer}
      {alerts}
      {drawer}
    </main>
  );
}
