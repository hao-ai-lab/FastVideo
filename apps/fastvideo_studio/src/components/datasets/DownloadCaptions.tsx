'use client';

import { ChevronDown } from 'lucide-react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

import { Button } from '@/components/ui/button';
import { downloadBlob } from '@/lib/utils';

const MENU_ITEM =
  'block min-h-11 w-full cursor-pointer select-none px-4 py-2 text-left text-sm font-medium text-foreground outline-none transition-colors data-[highlighted]:bg-muted';

export default function DownloadCaptions({
  fileNames,
  captions,
}: {
  fileNames: string[];
  captions: Record<string, string>;
}) {
  const disabled = fileNames.length === 0;

  // Sorted lazily in the click handlers: this component re-renders with the
  // sidebar on every caption keystroke, and the sort is only needed on click.
  const sortedFileNames = () => [...fileNames].sort();

  function handleDownloadJson() {
    const data = sortedFileNames().map((path) => ({
      path,
      cap: captions[path] ?? '',
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    downloadBlob(blob, 'videos2caption.json');
  }

  function handleDownloadTxt() {
    const sortedNames = sortedFileNames();
    const videosContent = sortedNames.join('\n');
    const promptContent = sortedNames
      .map((fn) => captions[fn] ?? '')
      .join('\n');
    downloadBlob(
      new Blob([videosContent], { type: 'text/plain' }),
      'videos.txt',
    );
    setTimeout(() => {
      downloadBlob(
        new Blob([promptContent], { type: 'text/plain' }),
        'captions.txt',
      );
    }, 100);
  }

  function handleDownloadCsv() {
    const escape = (s: string) =>
      s.includes('"') || s.includes(',') || s.includes('\n')
        ? `"${s.replace(/"/g, '""')}"`
        : s;
    const rows = sortedFileNames().map(
      (fn) => `${escape(fn)},${escape(captions[fn] ?? '')}`,
    );
    const csv = ['video_name,caption', ...rows].join('\n');
    downloadBlob(new Blob([csv], { type: 'text/csv' }), 'captions.csv');
  }

  // Same click/keyboard-accessible menu idiom as CreateJobButton — hover-only
  // menus exclude touch and keyboard users.
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled}
          className="gap-1.5"
        >
          Download Captions
          <ChevronDown className="h-3.5 w-3.5 opacity-85" aria-hidden />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={4}
          collisionPadding={8}
          className="z-[200] min-w-40 overflow-hidden rounded-lg border border-border bg-popover py-1 text-popover-foreground shadow-lg"
        >
          <DropdownMenu.Item className={MENU_ITEM} onSelect={handleDownloadJson}>
            JSON
          </DropdownMenu.Item>
          <DropdownMenu.Item className={MENU_ITEM} onSelect={handleDownloadTxt}>
            TXT
          </DropdownMenu.Item>
          <DropdownMenu.Item className={MENU_ITEM} onSelect={handleDownloadCsv}>
            CSV
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
