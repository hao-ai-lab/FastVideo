import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog';

// The app header is `fixed` at z-[100] (components/shell/Header.tsx).
const HEADER_Z = 100;

function zOf(el: Element | null): number {
  const m = el?.className.match(/\bz-\[(\d+)\]/);
  return m ? Number(m[1]) : 0;
}

describe('Dialog stacking', () => {
  it('renders its overlay and content above the fixed app header', () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Tall dialog</DialogTitle>
          <DialogDescription>Body</DialogDescription>
        </DialogContent>
      </Dialog>,
    );
    const content = screen.getByRole('dialog');
    const overlay = document.querySelector('[data-state="open"].bg-black\\/70');

    expect(zOf(content)).toBeGreaterThan(HEADER_Z);
    expect(zOf(overlay)).toBeGreaterThan(HEADER_Z);
  });
});
