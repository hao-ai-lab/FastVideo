'use client';

import { getDatasets, type Dataset } from "@/lib/api";
import { useEffect, useState, useCallback } from "react";
import CreateDatasetModal from "./CreateDatasetModal";
import DatasetCard from "./DatasetCard";
import { useHeaderActions } from "@/contexts/ActiveTabContext";
import { useActiveDataset } from "@/contexts/ActiveDatasetContext";
import cardStyles from "@styles/Card.module.css";
import layoutStyles from "@/app/Layout.module.css";
import buttonStyles from "@styles/Button.module.css";

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const { setActiveDataset, setActiveDatasetId } = useActiveDataset();

  useHeaderActions([
    <button
      key="add-dataset"
      type="button"
      className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`}
      onClick={() => setModalOpen(true)}
    >
      Add Dataset
    </button>,
  ]);

  const fetchDatasets = useCallback(async () => {
    try {
      const list = await getDatasets();
      setDatasets(list);
    } catch (err) {
      console.error("Failed to fetch datasets:", err);
    }
  }, []);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  const handleSelectDataset = useCallback(
    (ds: Dataset) => {
      setActiveDataset(ds);
      setActiveDatasetId(ds.id);
    },
    [setActiveDataset, setActiveDatasetId]
  );

  return (
    <main className={layoutStyles.main}>
      <section className={cardStyles.card}>
        <div id="datasets-container">
          {datasets.length === 0 ? (
            <p className={layoutStyles.placeholder}>
              No datasets yet. Add a raw video dataset to use in
              training.
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
      </section>
      <CreateDatasetModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSuccess={() => {
          fetchDatasets();
          setModalOpen(false);
        }}
      />
    </main>
  );
}
