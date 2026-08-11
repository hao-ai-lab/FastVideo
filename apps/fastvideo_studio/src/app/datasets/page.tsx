'use client';

import * as React from 'react';
import { AlertTriangle } from 'lucide-react';

import AddDatasetButton from '@/components/datasets/AddDatasetButton';
import CreateDatasetModal from '@/components/datasets/CreateDatasetModal';
import DatasetCard from '@/components/datasets/DatasetCard';
import { HeaderActions } from '@/components/shell/HeaderActionsContext';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useStore } from '@/hooks/useStore';
import { getDatasets } from '@/lib/api';
import type { Dataset } from '@/lib/api';
import {
  setActiveDataset,
  setActiveDatasetId,
} from '@/stores/activeDataset';
import {
  createDatasetModalStore,
  setCreateDatasetModalOpen,
} from '@/stores/createDatasetModalOpen';

export default function DatasetsPage() {
  const [datasets, setDatasets] = React.useState<Dataset[]>([]);
  const [isInitialLoading, setIsInitialLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const { open } = useStore(createDatasetModalStore);
  const fetchSequence = React.useRef(0);

  const fetchDatasets = React.useCallback(async () => {
    const sequence = ++fetchSequence.current;
    try {
      const next = await getDatasets();
      if (sequence === fetchSequence.current) {
        setDatasets(next);
        setError(null);
      }
    } catch (err) {
      console.error('Failed to fetch datasets:', err);
      if (sequence === fetchSequence.current) {
        setError(
          'Could not load datasets from the Studio API. Check the server and try again.',
        );
      }
    } finally {
      if (sequence === fetchSequence.current) setIsInitialLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  function handleSelectDataset(ds: Dataset) {
    setActiveDataset(ds);
    setActiveDatasetId(ds.id);
  }

  return (
    <>
      <HeaderActions>
        <AddDatasetButton />
      </HeaderActions>
      <div className="mx-auto flex w-full max-w-[850px] flex-col gap-6 px-4 pb-12">
        <Card className="p-6">
          <div aria-busy={isInitialLoading}>
            {isInitialLoading ? (
              <div
                aria-label="Loading datasets"
                className="flex flex-col gap-3 py-2"
              >
                {[0, 1, 2].map((item) => (
                  <div
                    key={item}
                    className="h-24 animate-pulse rounded-lg border border-border bg-muted/50"
                  />
                ))}
              </div>
            ) : error && datasets.length === 0 ? (
              <div
                role="alert"
                className="flex flex-col items-center gap-3 py-8 text-center"
              >
                <AlertTriangle
                  className="size-6 text-destructive"
                  aria-hidden
                />
                <p className="max-w-md text-sm text-muted-foreground">
                  {error}
                </p>
                <Button type="button" variant="outline" onClick={fetchDatasets}>
                  Try Again
                </Button>
              </div>
            ) : (
              <>
                {error && (
                  <p
                    role="status"
                    className="mb-3 rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm text-foreground"
                  >
                    Dataset updates are temporarily unavailable. Showing the
                    most recent results.
                  </p>
                )}
                {datasets.length === 0 ? (
                  <p className="py-8 text-center text-muted-foreground">
                    No datasets yet.
                  </p>
                ) : (
                  datasets.map((ds) => (
                    <DatasetCard
                      key={ds.id}
                      dataset={ds}
                      onUpdated={fetchDatasets}
                      onSelect={() => handleSelectDataset(ds)}
                    />
                  ))
                )}
              </>
            )}
          </div>
        </Card>
      </div>
      <CreateDatasetModal
        isOpen={open}
        onClose={() => setCreateDatasetModalOpen(false)}
        onSuccess={() => {
          fetchDatasets();
          setCreateDatasetModalOpen(false);
        }}
      />
    </>
  );
}
