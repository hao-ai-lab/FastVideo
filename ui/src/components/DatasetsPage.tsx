'use client';

import {
  getDatasets,
  deleteDataset,
  startDatasetPreprocess,
  stopDatasetPreprocess,
  type Dataset,
} from "@/lib/api";
import { useEffect, useState, useRef, useCallback } from "react";
import CreateDatasetModal from "./CreateDatasetModal";
import { useHeaderActions } from "@/contexts/ActiveTabContext";
import { useActiveDataset } from "@/contexts/ActiveDatasetContext";
import cardStyles from "@styles/Card.module.css";
import layoutStyles from "@/app/Layout.module.css";
import jobCardStyles from "@styles/JobCard.module.css";
import badgeStyles from "@styles/Badge.module.css";
import buttonStyles from "@styles/Button.module.css";

function DatasetCard({
  dataset,
  onUpdated,
  onSelect,
}: {
  dataset: Dataset;
  onUpdated: () => void;
  onSelect: () => void;
}) {
  const [isLoading, setIsLoading] = useState(false);

  const datasetType = dataset.dataset_type || "raw";

  const badgeClass =
    `badge${dataset.status.charAt(0).toUpperCase() + dataset.status.slice(1)}`;

  const handleRunPreprocess = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (
      isLoading ||
      dataset.status === "preprocessing" ||
      dataset.status === "ready"
    )
      return;
    setIsLoading(true);
    try {
      await startDatasetPreprocess(dataset.id);
      onUpdated();
    } catch (err) {
      console.error("Failed to start preprocessing:", err);
      alert(err instanceof Error ? err.message : "Failed to start preprocessing");
    } finally {
      setIsLoading(false);
    }
  };

  const handleStopPreprocess = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || dataset.status !== "preprocessing") return;
    setIsLoading(true);
    try {
      await stopDatasetPreprocess(dataset.id);
      onUpdated();
    } catch (err) {
      console.error("Failed to stop preprocessing:", err);
      alert(err instanceof Error ? err.message : "Failed to stop preprocessing");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    if (!confirm(`Delete dataset "${dataset.name}"?`)) return;
    setIsLoading(true);
    try {
      await deleteDataset(dataset.id);
      onUpdated();
    } catch (err) {
      console.error("Failed to delete dataset:", err);
      alert(err instanceof Error ? err.message : "Failed to delete dataset");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className={jobCardStyles.jobCard}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
    >
      <div className={jobCardStyles.jobHeader}>
        <span className={jobCardStyles.jobModel}>{dataset.name}</span>
        <span style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap" }}>
          <span className={badgeStyles.badgePending}>
            {datasetType.charAt(0).toUpperCase() + datasetType.slice(1)}
          </span>
          <span className={badgeStyles[badgeClass] || badgeStyles.badge}>
            {dataset.status}
          </span>
        </span>
      </div>
      <div className={jobCardStyles.jobPrompt}>
        {dataset.raw_path ? `Raw: ${dataset.raw_path}` : `Path: ${dataset.output_path}`}
      </div>
      <div className={jobCardStyles.jobMeta}>
        <span>{dataset.workload_type}</span>
        <span>{dataset.model_path.split("/").pop()}</span>
      </div>
      {dataset.error && (
        <div className={jobCardStyles.jobError}>{dataset.error}</div>
      )}
      <div className={jobCardStyles.jobActions}>
        {dataset.status === "pending" && dataset.dataset_type === "raw" && (
          <button
            type="button"
            className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`}
            onClick={handleRunPreprocess}
            disabled={isLoading}
          >
            Run Preprocessing
          </button>
        )}
        {dataset.status === "preprocessing" && (
          <button
            type="button"
            className={`${buttonStyles.btn} ${buttonStyles.btnStop}`}
            onClick={handleStopPreprocess}
            disabled={isLoading}
          >
            Stop
          </button>
        )}
        <button
          type="button"
          className={`${buttonStyles.btn} ${buttonStyles.btnDelete}`}
          onClick={handleDelete}
          disabled={isLoading}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
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

  useEffect(() => {
    const hasActive = datasets.some(
      (d) => d.status === "preprocessing" || d.status === "pending"
    );
    if (hasActive) {
      intervalRef.current = setInterval(fetchDatasets, 2000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [datasets, fetchDatasets]);

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
