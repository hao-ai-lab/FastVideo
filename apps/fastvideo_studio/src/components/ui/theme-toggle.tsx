'use client';

import * as React from 'react';
import { Moon, Sun } from 'lucide-react';

import { Button } from '@/components/ui/button';

/**
 * Light/dark switch (ported from Dreamverse). The studio defaults to dark —
 * layout.tsx applies the stored choice before paint; this button just
 * mirrors and persists it.
 */
function ThemeToggle({ className }: { className?: string }) {
  const [dark, setDark] = React.useState(true);

  React.useEffect(() => {
    setDark(document.documentElement.classList.contains('dark'));
  }, []);

  const transitionTimer =
    React.useRef<ReturnType<typeof setTimeout>>(undefined);

  function toggle() {
    const next = !dark;
    setDark(next);

    clearTimeout(transitionTimer.current);
    document.documentElement.classList.add('theme-transition');
    document.documentElement.classList.toggle('dark', next);

    transitionTimer.current = setTimeout(() => {
      document.documentElement.classList.remove('theme-transition');
    }, 350);

    try {
      localStorage.setItem('theme', next ? 'dark' : 'light');
    } catch {
      // storage unavailable
    }
  }

  React.useEffect(() => {
    return () => clearTimeout(transitionTimer.current);
  }, []);

  return (
    <Button
      variant="outline"
      size="icon-sm"
      onClick={toggle}
      aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      className={className}
    >
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  );
}

export { ThemeToggle };
