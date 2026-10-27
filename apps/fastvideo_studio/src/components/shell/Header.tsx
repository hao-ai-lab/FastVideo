'use client';

import { Menu, X } from 'lucide-react';
import { usePathname } from 'next/navigation';

import { useHeaderActions } from '@/components/shell/HeaderActionsContext';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/ui/theme-toggle';

const TAB_TITLES: Record<string, string> = {
  '/inference': 'Jobs',
  '/finetuning': 'Jobs',
  '/distillation': 'Jobs',
  '/datasets': 'Datasets',
  '/gallery': 'Gallery',
  '/gpus': 'GPUs',
  '/settings': 'Settings',
};

export default function Header({
  navigationOpen,
  onNavigationToggle,
}: {
  navigationOpen: boolean;
  onNavigationToggle: () => void;
}) {
  const pathname = usePathname();
  const { actions } = useHeaderActions();
  const title = TAB_TITLES[pathname] ?? 'FastVideo';

  return (
    <header className="fixed inset-x-0 top-0 z-[100] flex h-[var(--header-height)] items-center gap-2 border-b border-border bg-background/80 px-2 backdrop-blur sm:px-4 md:gap-6 md:px-6">
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label={navigationOpen ? 'Close navigation' : 'Open navigation'}
        aria-controls="primary-navigation"
        aria-expanded={navigationOpen}
        onClick={onNavigationToggle}
        className="shrink-0 md:hidden"
      >
        {navigationOpen ? (
          <X className="size-5" aria-hidden />
        ) : (
          <Menu className="size-5" aria-hidden />
        )}
      </Button>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/logo.svg"
        alt="FastVideo Logo"
        width={100}
        height={42}
        className="hidden h-[42px] w-[78px] shrink-0 object-contain min-[361px]:block md:w-[100px]"
      />
      <h1 className="sr-only m-0 flex-1 text-xl font-semibold tracking-tight md:not-sr-only">
        {title}
      </h1>
      <div className="ml-auto flex min-w-0 items-center gap-2 md:gap-3">
        {actions}
        <ThemeToggle />
      </div>
    </header>
  );
}
