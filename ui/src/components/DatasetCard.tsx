'use client';

import { deleteDataset, type Dataset } from "@/lib/api";
import { useState } from "react";
import { useActiveDataset } from "@/contexts/ActiveDatasetContext";
import jobCardStyles from "@styles/JobCard.module.css";
import buttonStyles from "@styles/Button.module.css";

export interface DatasetCardProps {
  dataset: Dataset;
  onUpdated: () => void;
  onSelect: () => void;
}

export default function DatasetCard({
  dataset,
  onUpdated,
  onSelect,
}: DatasetCardProps) {
  const [isLoading, setIsLoading] = useState(false);
  const { activeDatasetId, setActiveDatasetId } = useActiveDataset();
  const isSelected = activeDatasetId === dataset.id;

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    if (!confirm(`Delete dataset "${dataset.name}"?`)) return;
    setIsLoading(true);
    try {
      await deleteDataset(dataset.id);
      if (activeDatasetId === dataset.id) setActiveDatasetId(null);
      onUpdated();
    } catch (err) {
      console.error("Failed to delete dataset:", err);
      alert(err instanceof Error ? err.message : "Failed to delete dataset");
    } finally {
      setIsLoading(false);
    }
  };

  const fileCount = dataset.file_count ?? 0;
  const sizeBytes = dataset.size_bytes ?? 0;
  const sizeLabel =
    sizeBytes < 1024
      ? `${sizeBytes} B`
      : sizeBytes < 1024 * 1024
        ? `${(sizeBytes / 1024).toFixed(1)} KB`
        : sizeBytes < 1024 * 1024 * 1024
          ? `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`
          : `${(sizeBytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;

  return (
    <div
      className={`${jobCardStyles.jobCard} ${
        isSelected ? jobCardStyles.jobCardSelected : ""
      }`}
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
        <button
          type="button"
          className={`${buttonStyles.btn} ${buttonStyles.btnDelete} ${buttonStyles.btnSmall}`}
          onClick={handleDelete}
          disabled={isLoading}
        >
          Delete
        </button>
      </div>
      <div className={jobCardStyles.jobPrompt}>
        {fileCount} {fileCount === 1 ? "file" : "files"} · {sizeLabel}
      </div>
    </div>
  );
}
