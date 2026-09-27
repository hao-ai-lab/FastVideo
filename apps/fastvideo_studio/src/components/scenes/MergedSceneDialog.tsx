'use client';

import * as React from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { downloadMergedVideo, getMergedVideoUrl, type MergedScene } from '@/lib/api';
import { formatClock } from '@/lib/scenes';
import { downloadBlob } from '@/lib/utils';

export interface MergedSceneDialogProps {
  /** the merged video to show, or null when closed */
  merged: (MergedScene & { title: string }) | null;
  onClose: () => void;
}

/** The finished, merged scene: play it, and save it. */
export function MergedSceneDialog({ merged, onClose }: MergedSceneDialogProps) {
  const [downloading, setDownloading] = React.useState(false);

  async function download() {
    if (!merged) return;
    setDownloading(true);
    try {
      downloadBlob(await downloadMergedVideo(merged.filename), merged.filename);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't download the video.");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <Dialog open={!!merged} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>{merged?.title} merged</DialogTitle>
          <DialogDescription>
            {merged &&
              `${merged.clips} clip${merged.clips === 1 ? '' : 's'} joined in order · ${formatClock(merged.seconds)}`}
          </DialogDescription>
        </DialogHeader>
        {merged && (
          <video
            src={getMergedVideoUrl(merged.filename)}
            aria-label={`Merged video of ${merged.title}`}
            className="max-h-[60vh] w-full rounded-md border border-border bg-black"
            controls
            playsInline
            preload="metadata"
          />
        )}
        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button type="button" onClick={download} disabled={downloading}>
            {downloading ? 'Downloading…' : 'Download'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
