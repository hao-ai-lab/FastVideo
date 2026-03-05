'use client';

import { createDataset, uploadRawDataset } from "@/lib/api";
import { useEffect, useState } from "react";
import modalStyles from "./styles/Modal.module.css";
import formStyles from "./styles/Form.module.css";
import cardStyles from "./styles/Card.module.css";
import buttonStyles from "./styles/Button.module.css";
import UploadZone from "./UploadZone";

const ALLOWED_VIDEO_EXT = ".mp4,.webm,.avi,.mov,.mkv";

interface CreateDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function CreateDatasetModal({
  isOpen,
  onClose,
  onSuccess,
}: CreateDatasetModalProps) {
  const [name, setName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [rawPath, setRawPath] = useState("");
  const [fileNames, setFileNames] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

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
      setRawPath("");
      setFileNames([]);
      setValidationError(null);
      onClose();
    }
  };

  const handleMediaChange = async (files: File[]) => {
    setValidationError(null);
    if (files.length === 0) {
      setRawPath("");
      setFileNames([]);
      return;
    }
    setIsUploading(true);
    try {
      const res = await uploadRawDataset(files);
      setRawPath(res.path);
      setFileNames(res.file_names);
      if (res.file_names.length === 0) {
        setValidationError(
          `No video files found. Allowed: ${ALLOWED_VIDEO_EXT}`
        );
      }
    } catch (err) {
      console.error("Upload failed:", err);
      setRawPath("");
      setFileNames([]);
      setValidationError(
        err instanceof Error ? err.message : "Upload failed"
      );
    } finally {
      setIsUploading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);
    if (!name.trim()) return;

    if (!rawPath || fileNames.length === 0) {
      setValidationError("No data was found. Upload at least one video.");
      return;
    }

    setIsSubmitting(true);
    try {
      await createDataset({
        name: name.trim(),
        upload_path: rawPath,
        file_names: fileNames,
      });
      onSuccess();
      handleClose();
    } catch (err) {
      console.error("Failed to create dataset:", err);
      setValidationError(
        err instanceof Error ? err.message : "Failed to create dataset"
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

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
          <h2>Add Dataset — Raw</h2>
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
              <label>Videos</label>
              <UploadZone
                label="Upload video files"
                hint="Click or drop one or more videos (.mp4, .webm, .avi, .mov, .mkv)"
                accept={ALLOWED_VIDEO_EXT}
                multiple
                value={rawPath}
                fileName={
                  fileNames.length > 0
                    ? `${fileNames.length} file(s)`
                    : undefined
                }
                onFileChange={handleMediaChange}
                onClear={() => {
                  setRawPath("");
                  setFileNames([]);
                  setValidationError(null);
                }}
                disabled={isSubmitting}
                uploading={isUploading}
              />
            </div>
            {validationError && (
              <div
                className={formStyles.formRow}
                style={{ color: "var(--red)", fontSize: "0.9rem" }}
              >
                {validationError}
              </div>
            )}
            <div className={formStyles.formRow} style={{ marginTop: "1rem" }}>
              <button
                type="submit"
                className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`}
                disabled={isSubmitting || isUploading || fileNames.length === 0}
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
