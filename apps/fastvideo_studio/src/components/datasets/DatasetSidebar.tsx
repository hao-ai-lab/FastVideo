'use client';

import * as React from 'react';
import { ImageOff, X } from 'lucide-react';
import { toast } from 'sonner';

import DownloadCaptions from '@/components/datasets/DownloadCaptions';
import { Textarea } from '@/components/ui/textarea';
import { useResizable } from '@/hooks/useResizable';
import {
  getDatasetFiles,
  getDatasetMediaUrl,
  updateDatasetCaption,
  type Dataset,
} from '@/lib/api';
import { cn } from '@/lib/utils';
import { useDrawerFocus } from '@/hooks/useDrawerFocus';

const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_MAX_WIDTH = 900;
const INITIAL_PAGE_SIZE = 24;
const PAGE_SIZE = 24;
const SCROLL_THRESHOLD = 200;
type CaptionSaveState = 'idle' | 'saving' | 'saved' | 'error';

// Memoized so a caption keystroke re-renders only the edited card, not every
// visible <video> in the grid (visibleCount grows unbounded with scrolling).
const DatasetFileCard = React.memo(function DatasetFileCard({
  fileName,
  mediaUrl,
  caption,
  thumbLoaded,
  saveState,
  onCaptionChange,
  onCaptionRetry,
  onThumbLoaded,
}: {
  fileName: string;
  mediaUrl: string;
  caption: string;
  thumbLoaded: boolean;
  saveState: CaptionSaveState;
  onCaptionChange: (fileName: string, value: string) => void;
  onCaptionRetry: (fileName: string, value: string) => void;
  onThumbLoaded: (fileName: string) => void;
}) {
  const [mediaFailed, setMediaFailed] = React.useState(false);

  React.useEffect(() => {
    setMediaFailed(false);
  }, [mediaUrl]);

  return (
    <div className="relative flex flex-col overflow-hidden rounded-lg border border-border bg-background">
      {!thumbLoaded && !mediaFailed && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/70">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-muted-foreground/40 border-t-accent-blue" />
        </div>
      )}
      {mediaFailed ? (
        <div
          role="status"
          className="flex aspect-video w-full flex-col items-center justify-center gap-1 bg-muted px-2 text-center text-muted-foreground"
        >
          <ImageOff className="size-5" aria-hidden />
          <span className="text-xs">Preview unavailable</span>
        </div>
      ) : (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <video
          src={mediaUrl}
          aria-label={`Preview of ${fileName}`}
          className="aspect-video w-full bg-border object-cover"
          muted
          autoPlay
          loop
          playsInline
          onLoadedData={() => onThumbLoaded(fileName)}
          onError={() => {
            setMediaFailed(true);
            onThumbLoaded(fileName);
          }}
        />
      )}
      <Textarea
        aria-label={`Caption for ${fileName}`}
        value={caption}
        onChange={(e) => onCaptionChange(fileName, e.target.value)}
        placeholder="Caption"
        rows={2}
        className="min-h-[2.5rem] resize-y rounded-none border-0 bg-transparent p-1.5 text-xs shadow-none focus-visible:border-transparent focus-visible:ring-0"
      />
      <div
        aria-live="polite"
        className="flex min-h-6 items-center px-1.5 pb-1 text-[0.7rem] text-muted-foreground"
      >
        {saveState === 'saving' && <span>Saving…</span>}
        {saveState === 'saved' && <span>Saved</span>}
        {saveState === 'error' && (
          <span role="alert" className="text-destructive">
            Not saved.{' '}
            <button
              type="button"
              onClick={() => onCaptionRetry(fileName, caption)}
              className="inline-flex min-h-11 items-center font-medium underline underline-offset-2"
            >
              Retry
            </button>
          </span>
        )}
      </div>
    </div>
  );
});

export default function DatasetSidebar({
  dataset,
  isMobile = false,
  onClose,
  onWidthChange,
}: {
  dataset: Dataset;
  isMobile?: boolean;
  onClose: () => void;
  onWidthChange?: (w: number) => void;
}) {
  const drawerRef = useDrawerFocus<HTMLElement>(isMobile);
  const [width, setWidth] = React.useState(400);
  const [isDragging, setIsDragging] = React.useState(false);
  const [fileNames, setFileNames] = React.useState<string[]>([]);
  const [captions, setCaptions] = React.useState<Record<string, string>>({});
  const [visibleCount, setVisibleCount] = React.useState(INITIAL_PAGE_SIZE);
  const [isLoading, setIsLoading] = React.useState(true);
  const [thumbLoaded, setThumbLoaded] = React.useState<
    Record<string, boolean>
  >({});
  const [captionSaveStates, setCaptionSaveStates] = React.useState<
    Record<string, CaptionSaveState>
  >({});

  // Pending debounced caption saves, keyed per file so editing one caption
  // can't cancel another file's pending save.
  const pendingSaves = React.useRef(
    new Map<string, { timer: ReturnType<typeof setTimeout>; save: () => void }>(),
  );
  const captionVersions = React.useRef(new Map<string, number>());
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    onWidthChange?.(isMobile ? 0 : width);
  }, [isMobile, width, onWidthChange]);

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
        setCaptionSaveStates({});
        captionVersions.current.clear();
      })
      .catch((err) => console.error('Failed to load dataset files:', err))
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  const flushPendingSaves = React.useCallback(() => {
    for (const { timer, save } of pendingSaves.current.values()) {
      clearTimeout(timer);
      save();
    }
    pendingSaves.current.clear();
  }, []);

  // Flush (not drop) pending saves when the dataset changes or on unmount, so
  // an edit made within the debounce window of closing isn't lost.
  React.useEffect(() => flushPendingSaves, [dataset.id, flushPendingSaves]);

  const { onMouseDown } = useResizable({
    edge: 'right',
    minWidth: SIDEBAR_MIN_WIDTH,
    maxWidth: SIDEBAR_MAX_WIDTH,
    getWidth: () => width,
    onWidth: setWidth,
    onDragChange: setIsDragging,
  });

  const datasetId = dataset.id;
  const persistCaption = React.useCallback(
    (fileName: string, value: string, version: number) => {
      setCaptionSaveStates((prev) => ({ ...prev, [fileName]: 'saving' }));
      void updateDatasetCaption(datasetId, fileName, value)
        .then(() => {
          if (captionVersions.current.get(fileName) !== version) return;
          setCaptionSaveStates((prev) => ({ ...prev, [fileName]: 'saved' }));
        })
        .catch((error) => {
          if (captionVersions.current.get(fileName) !== version) return;
          console.error('Failed to save caption:', error);
          setCaptionSaveStates((prev) => ({ ...prev, [fileName]: 'error' }));
          toast.error('Caption was not saved', {
            description: `${fileName}: check the Studio API, then retry.`,
          });
        });
    },
    [datasetId],
  );

  const handleCaptionChange = React.useCallback(
    (fileName: string, value: string) => {
      setCaptions((prev) => ({ ...prev, [fileName]: value }));
      setCaptionSaveStates((prev) => ({ ...prev, [fileName]: 'idle' }));
      const pending = pendingSaves.current.get(fileName);
      if (pending) clearTimeout(pending.timer);
      const version = (captionVersions.current.get(fileName) ?? 0) + 1;
      captionVersions.current.set(fileName, version);
      const save = () => persistCaption(fileName, value, version);
      const timer = setTimeout(() => {
        pendingSaves.current.delete(fileName);
        save();
      }, 500);
      pendingSaves.current.set(fileName, { timer, save });
    },
    [persistCaption],
  );

  const handleCaptionRetry = React.useCallback(
    (fileName: string, value: string) => {
      const version = (captionVersions.current.get(fileName) ?? 0) + 1;
      captionVersions.current.set(fileName, version);
      persistCaption(fileName, value, version);
    },
    [persistCaption],
  );

  function handleScroll() {
    const el = scrollRef.current;
    if (!el || isLoading || visibleCount >= fileNames.length) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    if (distanceFromBottom < SCROLL_THRESHOLD) {
      setVisibleCount((c) => Math.min(c + PAGE_SIZE, fileNames.length));
    }
  }

  // When the visible grid doesn't overflow (a wide/tall sidebar can fit the
  // first page without a scrollbar), no scroll event ever fires — so top up
  // visibleCount until it overflows or every file is shown. Re-runs on resize
  // (width) and after each page grows, otherwise files 25..N are unreachable.
  React.useEffect(() => {
    if (isLoading || visibleCount >= fileNames.length) return;
    const el = scrollRef.current;
    if (el && el.scrollHeight <= el.clientHeight) {
      setVisibleCount((c) => Math.min(c + PAGE_SIZE, fileNames.length));
    }
  }, [visibleCount, fileNames.length, isLoading, width]);

  const markThumbLoaded = React.useCallback((fileName: string) => {
    setThumbLoaded((prev) =>
      prev[fileName] ? prev : { ...prev, [fileName]: true },
    );
  }, []);

  const visibleFiles = fileNames.slice(0, visibleCount);

  return (
    <aside
      ref={drawerRef}
      tabIndex={-1}
      role="dialog"
      aria-label={`${dataset.name} dataset details`}
      aria-modal={isMobile || undefined}
      className="fixed bottom-0 right-0 top-[var(--header-height)] z-50 flex max-h-[calc(100dvh-var(--header-height))] min-w-0 shrink-0 flex-col border-l border-border bg-card md:min-w-[320px]"
      style={{
        width: isMobile ? '100%' : width,
        maxWidth: isMobile ? 'none' : SIDEBAR_MAX_WIDTH,
      }}
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
            className="flex size-11 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
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
                <DatasetFileCard
                  key={fileName}
                  fileName={fileName}
                  mediaUrl={getDatasetMediaUrl(dataset.id, fileName)}
                  caption={captions[fileName] ?? ''}
                  thumbLoaded={!!thumbLoaded[fileName]}
                  saveState={captionSaveStates[fileName] ?? 'idle'}
                  onCaptionChange={handleCaptionChange}
                  onCaptionRetry={handleCaptionRetry}
                  onThumbLoaded={markThumbLoaded}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {!isMobile && <div
        role="presentation"
        onMouseDown={onMouseDown}
        className={cn(
          'absolute bottom-0 left-0 top-0 z-[1] w-1.5 cursor-col-resize hover:bg-accent-blue/25',
          isDragging && 'bg-accent-blue/25',
        )}
      />}
    </aside>
  );
}
