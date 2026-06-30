'use client';

import { ChevronDown } from 'lucide-react';

import { Button } from '@/components/ui/button';

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const MENU_ITEM =
  'block w-full cursor-pointer px-4 py-2 text-left text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50';

export default function DownloadCaptions({
  fileNames,
  captions,
}: {
  fileNames: string[];
  captions: Record<string, string>;
}) {
  const sortedNames = [...fileNames].sort();
  const disabled = fileNames.length === 0;

  function handleDownloadJson() {
    const data = sortedNames.map((path) => ({
      path,
      cap: captions[path] ?? '',
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    downloadBlob(blob, 'videos2caption.json');
  }

  function handleDownloadTxt() {
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
    const rows = sortedNames.map(
      (fn) => `${escape(fn)},${escape(captions[fn] ?? '')}`,
    );
    const csv = ['video_name,caption', ...rows].join('\n');
    downloadBlob(new Blob([csv], { type: 'text/csv' }), 'captions.csv');
  }

  return (
    <div className="group relative inline-block">
      <Button
        type="button"
        variant="outline"
        size="sm"
        disabled={disabled}
        aria-haspopup="menu"
        className="gap-1.5"
      >
        Download Captions
        <ChevronDown className="h-3.5 w-3.5 opacity-85" />
      </Button>
      {!disabled && (
        <div
          role="menu"
          className="invisible absolute right-0 top-full z-[200] min-w-full -translate-y-1 pt-1 opacity-0 transition-all group-focus-within:visible group-focus-within:translate-y-0 group-focus-within:opacity-100 group-hover:visible group-hover:translate-y-0 group-hover:opacity-100"
        >
          <div className="overflow-hidden rounded-lg border border-border bg-popover py-1 shadow-lg">
            <button
              type="button"
              role="menuitem"
              className={MENU_ITEM}
              onClick={handleDownloadJson}
            >
              JSON
            </button>
            <button
              type="button"
              role="menuitem"
              className={MENU_ITEM}
              onClick={handleDownloadTxt}
            >
              TXT
            </button>
            <button
              type="button"
              role="menuitem"
              className={MENU_ITEM}
              onClick={handleDownloadCsv}
            >
              CSV
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
