'use client';

import { usePathname } from 'next/navigation';

import { useHeaderActions } from '@/components/HeaderActionsContext';

const TAB_TITLES: Record<string, string> = {
  '/inference': 'Jobs',
  '/finetuning': 'Jobs',
  '/distillation': 'Jobs',
  '/datasets': 'Datasets',
  '/gallery': 'Gallery',
  '/settings': 'Settings',
};

export default function Header() {
  const pathname = usePathname();
  const { actions } = useHeaderActions();
  const title = TAB_TITLES[pathname] ?? 'FastVideo';

  return (
    <header className="fixed inset-x-0 top-0 z-[100] flex h-[var(--header-height)] items-center gap-6 border-b border-border bg-background px-6">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/logo.svg"
        alt="FastVideo Logo"
        width={100}
        height={42}
        className="block h-[42px] w-[100px]"
      />
      <h1 className="m-0 flex-1 text-xl font-semibold tracking-tight">
        {title}
      </h1>
      {actions ? (
        <div className="flex items-center gap-3">{actions}</div>
      ) : null}
    </header>
  );
}
