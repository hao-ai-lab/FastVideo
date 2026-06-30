'use client';

import * as React from 'react';

import AddDatasetButton from '@/components/AddDatasetButton';
import CreateDatasetModal from '@/components/CreateDatasetModal';
import DatasetCard from '@/components/DatasetCard';
import { useHeaderActions } from '@/components/HeaderActionsContext';
import { Card } from '@/components/ui/card';
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
  const { open } = useStore(createDatasetModalStore);
  const { setActions } = useHeaderActions();

  const fetchDatasets = React.useCallback(async () => {
    try {
      setDatasets(await getDatasets());
    } catch (err) {
      console.error('Failed to fetch datasets:', err);
    }
  }, []);

  React.useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  React.useEffect(() => {
    setActions(<AddDatasetButton />);
    return () => setActions(null);
  }, [setActions]);

  function handleSelectDataset(ds: Dataset) {
    setActiveDataset(ds);
    setActiveDatasetId(ds.id);
  }

  return (
    <>
      <main className="mx-auto flex w-full max-w-[850px] flex-col gap-6 px-4 pb-12">
        <Card className="p-6">
          <div>
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
          </div>
        </Card>
      </main>
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
