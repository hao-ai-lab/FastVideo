'use client';

import {
  createDataset,
  getModels,
  uploadDatasetCaptions,
  uploadDatasetParquet,
  uploadDatasetVideos,
  type Model,
} from "@/lib/api";
import { useEffect, useState } from "react";
import modalStyles from "./styles/Modal.module.css";
import formStyles from "./styles/Form.module.css";
import cardStyles from "./styles/Card.module.css";
import buttonStyles from "./styles/Button.module.css";
import UploadZone from "./UploadZone";

export type DatasetSourceType = "raw" | "parquet" | "hf";

interface AddDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  sourceType: DatasetSourceType;
}

export default function AddDatasetModal({
  isOpen,
  onClose,
  onSuccess,
  sourceType,
}: AddDatasetModalProps) {
  const [name, setName] = useState("");
  const [models, setModels] = useState<Model[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [workloadType, setWorkloadType] = useState("t2v");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  // Raw: captions + videos
  const [captionsFile, setCaptionsFile] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [rawPath, setRawPath] = useState("");
  const [videoFiles, setVideoFiles] = useState<File[]>([]);
  const [isUploadingCaptions, setIsUploadingCaptions] = useState(false);
  const [isUploadingVideos, setIsUploadingVideos] = useState(false);

  // Parquet: file upload
  const [parquetPath, setParquetPath] = useState("");
  const [parquetFiles, setParquetFiles] = useState<File[]>([]);
  const [isUploadingParquet, setIsUploadingParquet] = useState(false);

  // HuggingFace: text path
  const [hfPath, setHfPath] = useState("");

  useEffect(() => {
    if (isOpen) {
      setIsLoadingModels(true);
      getModels("t2v")
        .then((fetched) => {
          setModels(fetched);
          if (fetched.length > 0 && !modelPath) {
            setModelPath(fetched[0].id);
          }
        })
        .finally(() => setIsLoadingModels(false));
    }
  }, [isOpen, modelPath]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen && !isSubmitting) onClose();
    };
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      return () => document.removeEventListener("keydown", handleEscape);
    }
  }, [isOpen, isSubmitting, onClose]);

  const handleClose = () => {
    if (!isSubmitting) {
      setName("");
      setModelPath("");
      setCaptionsFile(null);
      setUploadId(null);
      setRawPath("");
      setVideoFiles([]);
      setParquetPath("");
      setParquetFiles([]);
      setHfPath("");
      onClose();
    }
  };

  const handleCaptionsChange = async (files: File[]) => {
    const file = files[0];
    if (!file) {
      setCaptionsFile(null);
      setUploadId(null);
      setRawPath("");
      return;
    }
    setCaptionsFile(file);
    setIsUploadingCaptions(true);
    try {
      const res = await uploadDatasetCaptions(file);
      setUploadId(res.upload_id);
      setRawPath(res.path);
    } catch (err) {
      console.error("Captions upload failed:", err);
      setUploadId(null);
      setRawPath("");
    } finally {
      setIsUploadingCaptions(false);
    }
  };

  const handleVideosChange = async (files: File[]) => {
    setVideoFiles(files);
    if (files.length === 0 || !uploadId) return;
    setIsUploadingVideos(true);
    try {
      const res = await uploadDatasetVideos(uploadId, files);
      setRawPath(res.path);
    } catch (err) {
      console.error("Videos upload failed:", err);
    } finally {
      setIsUploadingVideos(false);
    }
  };

  const handleParquetChange = async (files: File[]) => {
    setParquetFiles(files);
    if (files.length === 0) {
      setParquetPath("");
      return;
    }
    setIsUploadingParquet(true);
    try {
      const res = await uploadDatasetParquet(files);
      setParquetPath(`${res.path}/combined_parquet_dataset`);
    } catch (err) {
      console.error("Parquet upload failed:", err);
      setParquetPath("");
    } finally {
      setIsUploadingParquet(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !modelPath) return;

    let rawPathVal = "";
    let outputPathVal: string | undefined;

    if (sourceType === "raw") {
      if (!rawPath || videoFiles.length === 0) {
        alert("Upload captions JSON first, then at least one video.");
        return;
      }
      rawPathVal = rawPath;
    } else if (sourceType === "parquet") {
      if (!parquetPath) {
        alert("Upload parquet files first.");
        return;
      }
      outputPathVal = parquetPath;
    } else {
      if (!hfPath.trim()) {
        alert("Enter a HuggingFace dataset path (e.g. org/dataset-name).");
        return;
      }
      rawPathVal = hfPath.trim();
    }

    setIsSubmitting(true);
    try {
      await createDataset({
        name: name.trim(),
        raw_path: rawPathVal,
        output_path: outputPathVal,
        workload_type: workloadType,
        model_path: modelPath,
        dataset_type: "merged",
      });
      onSuccess();
      handleClose();
    } catch (err) {
      console.error("Failed to create dataset:", err);
      alert(err instanceof Error ? err.message : "Failed to create dataset");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  const sourceLabel =
    sourceType === "raw"
      ? "Raw (captions + videos)"
      : sourceType === "parquet"
        ? "Parquet"
        : "HuggingFace";

  return (
    <div className={modalStyles.modal} data-modal>
      <div className={modalStyles.modalBackdrop} onClick={handleClose} />
      <div className={`${modalStyles.modalContent} ${modalStyles.modalForm}`}>
        <button
          className={modalStyles.modalClose}
          onClick={handleClose}
          disabled={isSubmitting}
          aria-label="Close"
        >
          ×
        </button>
        <div className={cardStyles.card} style={{ margin: 0, border: "none" }}>
          <h2>Add Dataset — {sourceLabel}</h2>
          <form onSubmit={handleSubmit} autoComplete="off">
            <div className={formStyles.formRow}>
              <label htmlFor="add-dataset-name">Name</label>
              <input
                id="add-dataset-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My dataset"
                required
                disabled={isSubmitting}
              />
            </div>
            <div className={formStyles.formRow}>
              <label htmlFor="add-dataset-model">Model</label>
              <select
                id="add-dataset-model"
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
                required
                disabled={isSubmitting || isLoadingModels}
              >
                <option value="">
                  {isLoadingModels
                    ? "Loading models…"
                    : models.length === 0
                      ? "No models available"
                      : "Select a model…"}
                </option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label} ({m.id})
                  </option>
                ))}
              </select>
            </div>
            <div className={formStyles.formRow}>
              <label htmlFor="add-dataset-workload">Workload type</label>
              <select
                id="add-dataset-workload"
                value={workloadType}
                onChange={(e) => setWorkloadType(e.target.value)}
                disabled={isSubmitting}
              >
                <option value="t2v">t2v</option>
                <option value="i2v">i2v</option>
              </select>
            </div>

            {sourceType === "raw" && (
              <>
                <div className={formStyles.formRow}>
                  <label>Captions (JSON)</label>
                  <UploadZone
                    label="Upload captions JSON"
                    hint="videos2caption.json format"
                    accept=".json"
                    value={rawPath}
                    fileName={captionsFile?.name}
                    onFileChange={handleCaptionsChange}
                    onClear={() => {
                      setCaptionsFile(null);
                      setUploadId(null);
                      setRawPath("");
                      setVideoFiles([]);
                    }}
                    disabled={isSubmitting}
                    uploading={isUploadingCaptions}
                  />
                </div>
                {uploadId && (
                  <div className={formStyles.formRow}>
                    <label>Videos</label>
                    <UploadZone
                      label="Upload video files"
                      hint="Add videos to the dataset"
                      accept=".mp4,.webm,.avi,.mov"
                      multiple
                      directory
                      value={rawPath}
                      fileName={
                        videoFiles.length > 0
                          ? `${videoFiles.length} file(s)`
                          : undefined
                      }
                      onFileChange={handleVideosChange}
                      onClear={() => setVideoFiles([])}
                      disabled={isSubmitting}
                      uploading={isUploadingVideos}
                    />
                  </div>
                )}
              </>
            )}

            {sourceType === "parquet" && (
              <div className={formStyles.formRow}>
                <label>Parquet files</label>
                <UploadZone
                  label="Upload parquet file(s)"
                  hint="Preprocessed parquet dataset"
                  accept=".parquet"
                  multiple
                  value={parquetPath}
                  fileName={
                    parquetFiles.length > 0
                      ? `${parquetFiles.length} file(s)`
                      : undefined
                  }
                  onFileChange={handleParquetChange}
                  onClear={() => {
                    setParquetFiles([]);
                    setParquetPath("");
                  }}
                  disabled={isSubmitting}
                  uploading={isUploadingParquet}
                />
              </div>
            )}

            {sourceType === "hf" && (
              <div className={formStyles.formRow}>
                <label htmlFor="add-dataset-hf">HuggingFace dataset path</label>
                <input
                  id="add-dataset-hf"
                  type="text"
                  value={hfPath}
                  onChange={(e) => setHfPath(e.target.value)}
                  placeholder="org/dataset-name"
                  disabled={isSubmitting}
                />
              </div>
            )}

            <div className={formStyles.formRow} style={{ marginTop: "1rem" }}>
              <button
                type="submit"
                className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`}
                disabled={
                  isSubmitting ||
                  isLoadingModels ||
                  isUploadingCaptions ||
                  isUploadingVideos ||
                  isUploadingParquet
                }
              >
                {isSubmitting ? "Creating…" : "Create Dataset"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
