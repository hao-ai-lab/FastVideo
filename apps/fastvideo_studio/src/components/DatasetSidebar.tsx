'use client';

import * as React from 'react';
import { X } from 'lucide-react';

import DownloadCaptions from '@/components/DownloadCaptions';
import { Textarea } from '@/components/ui/textarea';
import { useResizable } from '@/hooks/useResizable';
import {
  getDatasetFiles,
  getDatasetMediaUrl,
  updateDatasetCaption,
  type Dataset,
} from '@/lib/api';
import { cn } from '@/lib/utils';

const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_MAX_WIDTH = 900;
const INITIAL_PAGE_SIZE = 24;
const PAGE_SIZE = 24;
const SCROLL_THRESHOLD = 200;

export default function DatasetSidebar({
  dataset,
  onClose,
  onWidthChange,
}: {
  dataset: Dataset;
  onClose: () => void;
  onWidthChange?: (w: number) => void;
}) {
  const [width, setWidth] = React.useState(400);
  const [isDragging, setIsDragging] = React.useState(false);
  const [fileNames, setFileNames] = React.useState<string[]>([]);
  const [captions, setCaptions] = React.useState<Record<string, string>>({});
  const [visibleCount, setVisibleCount] = React.useState(INITIAL_PAGE_SIZE);
  const [isLoading, setIsLoading] = React.useState(true);
  const [thumbLoaded, setThumbLoaded] = React.useState<
    Record<string, boolean>
  >({});

  const saveTimeout = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    onWidthChange?.(width);
  }, [width, onWidthChange]);

  React.useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    getDatasetFiles(dataset.id)
      .then((data) => {
        if (cancelled) return;
        setFileNames(data.file_names);
        setCaptions(data.captions);
        setVisibleCount(INITIAL_PAGE_SIZE);
        setThumbLoaded({});
      })
      .catch((err) => console.error('Failed to load dataset files:', err))
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  React.useEffect(() => {
    return () => {
      if (saveTimeout.current) clearTimeout(saveTimeout.current);
    };
  }, []);

  const { onMouseDown } = useResizable({
    edge: 'right',
    minWidth: SIDEBAR_MIN_WIDTH,
    maxWidth: SIDEBAR_MAX_WIDTH,
    getWidth: () => width,
    onWidth: setWidth,
    onDragChange: setIsDragging,
  });

  function handleCaptionChange(fileName: string, value: string) {
    setCaptions((prev) => ({ ...prev, [fileName]: value }));
    if (saveTimeout.current) clearTimeout(saveTimeout.current);
    saveTimeout.current = setTimeout(() => {
      saveTimeout.current = null;
      updateDatasetCaption(dataset.id, fileName, value).catch((err) =>
        console.error('Failed to save caption:', err),
      );
    }, 500);
  }

  function handleScroll() {
    const el = scrollRef.current;
    if (!el || isLoading || visibleCount >= fileNames.length) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    if (distanceFromBottom < SCROLL_THRESHOLD) {
      setVisibleCount((c) => Math.min(c + PAGE_SIZE, fileNames.length));
    }
  }

  function markThumbLoaded(fileName: string) {
    setThumbLoaded((prev) => ({ ...prev, [fileName]: true }));
  }

  const visibleFiles = fileNames.slice(0, visibleCount);

  return (
    <aside
      className="fixed bottom-0 right-0 top-[var(--header-height)] z-50 flex max-h-[calc(100vh-var(--header-height))] min-w-[320px] shrink-0 flex-col border-l border-border bg-card"
      style={{ width, maxWidth: SIDEBAR_MAX_WIDTH }}
    >
      <div className="flex shrink-0 items-center justify-between border-b border-border px-5 py-4">
        <h2 className="m-0 min-w-0 truncate text-base font-semibold text-foreground">
          {dataset.name}
        </h2>
        <div className="flex shrink-0 items-center gap-2">
          <DownloadCaptions fileNames={fileNames} captions={captions} />
          <button
            type="button"
            onClick={onClose}
            title="Close"
            aria-label="Close"
            className="flex items-center justify-center rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-accent/10 hover:text-foreground"
          >
            <X className="h-[18px] w-[18px]" />
          </button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto p-4"
        >
          {isLoading ? (
            <p className="p-8 text-center text-muted-foreground">Loading…</p>
          ) : fileNames.length === 0 ? (
            <p className="p-8 text-center text-muted-foreground">
              No media files
            </p>
          ) : (
            <div className="grid gap-4 [grid-template-columns:repeat(auto-fill,minmax(140px,1fr))]">
              {visibleFiles.map((fileName) => (
                <div
                  key={fileName}
                  className="relative flex flex-col overflow-hidden rounded-lg border border-border bg-background"
                >
                  {!thumbLoaded[fileName] && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/70">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-muted-foreground/40 border-t-accent" />
                    </div>
                  )}
                  {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                  <video
                    src={getDatasetMediaUrl(dataset.id, fileName)}
                    className="aspect-video w-full bg-border object-cover"
                    muted
                    autoPlay
                    loop
                    playsInline
                    onLoadedData={() => markThumbLoaded(fileName)}
                    onError={() => markThumbLoaded(fileName)}
                  />
                  <Textarea
                    value={captions[fileName] ?? ''}
                    onChange={(e) =>
                      handleCaptionChange(fileName, e.target.value)
                    }
                    placeholder="Caption"
                    rows={2}
                    className="min-h-[2.5rem] resize-y rounded-none border-0 bg-transparent p-1.5 text-xs shadow-none focus-visible:border-transparent focus-visible:ring-0"
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div
        role="presentation"
        onMouseDown={onMouseDown}
        className={cn(
          'absolute bottom-0 left-0 top-0 z-[1] w-1.5 cursor-col-resize hover:bg-accent/20',
          isDragging && 'bg-accent/20',
        )}
      />
    </aside>
  );
}
