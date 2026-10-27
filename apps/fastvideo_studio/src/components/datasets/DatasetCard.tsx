'use client';

import * as React from 'react';

import { Button } from '@/components/ui/button';
import { deleteDataset } from '@/lib/api';
import type { Dataset } from '@/lib/api';
import { cn } from '@/lib/utils';
import { activeDatasetStore, setActiveDatasetId } from '@/stores/activeDataset';
import { useStore } from '@/hooks/useStore';

function formatSize(sizeBytes: number): string {
  if (sizeBytes < 1024) return `${sizeBytes} B`;
  if (sizeBytes < 1024 * 1024) return `${(sizeBytes / 1024).toFixed(1)} KB`;
  if (sizeBytes < 1024 * 1024 * 1024) {
    return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(sizeBytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export default function DatasetCard({
  dataset,
  onUpdated,
  onSelect = () => {},
}: {
  dataset: Dataset;
  onUpdated: () => void;
  onSelect?: () => void;
}) {
  const { activeDatasetId } = useStore(activeDatasetStore);
  const [isLoading, setIsLoading] = React.useState(false);

  const isSelected = activeDatasetId === dataset.id;
  const fileCount = dataset.file_count ?? 0;
  const sizeLabel = formatSize(dataset.size_bytes ?? 0);

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    if (!window.confirm(`Delete dataset "${dataset.name}"?`)) return;
    setIsLoading(true);
    try {
      await deleteDataset(dataset.id);
      if (activeDatasetStore.get().activeDatasetId === dataset.id) {
        setActiveDatasetId(null);
      }
      onUpdated();
    } catch (err) {
      window.alert(
        err instanceof Error ? err.message : 'Failed to delete dataset',
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <article
      className={cn(
        'mb-3 flex items-start gap-3 rounded-lg border border-border bg-background px-[1.15rem] py-4',
        isSelected && 'border-accent-blue bg-accent-blue/5',
      )}
    >
      <button
        type="button"
        aria-pressed={isSelected}
        onClick={onSelect}
        className="flex min-w-0 flex-1 cursor-pointer flex-col gap-[0.6rem] rounded-md text-left"
      >
        <span className="text-[0.95rem] font-semibold">{dataset.name}</span>
        <span className="text-sm text-muted-foreground">
          {fileCount} {fileCount === 1 ? 'file' : 'files'} · {sizeLabel}
        </span>
      </button>
      <Button
        type="button"
        variant="destructive"
        size="sm"
        onClick={handleDelete}
        disabled={isLoading}
      >
        Delete
      </Button>
    </article>
  );
}
