'use client';

import { createDataset, uploadRawDataset } from "@/lib/api";
import { useEffect, useState } from "react";
import modalStyles from "./styles/Modal.module.css";
import formStyles from "./styles/Form.module.css";
import cardStyles from "./styles/Card.module.css";
import buttonStyles from "./styles/Button.module.css";
import UploadZone from "./UploadZone";

const ALLOWED_VIDEO_EXT = ".mp4,.webm,.avi,.mov,.mkv";

/** Parse and validate videos2caption.json. Returns { captions, error }. */
function parseVideos2Caption(
  text: string,
  uploadedFileNames: string[]
): { captions: Record<string, string>; error: string | null } {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { captions: {}, error: "Invalid JSON in videos2caption.json." };
  }

  const captions: Record<string, string> = {};
  const uploadedSet = new Set(uploadedFileNames);

  if (Array.isArray(data)) {
    for (let i = 0; i < data.length; i++) {
      const entry = data[i];
      if (entry === null || typeof entry !== "object") {
        return {
          captions: {},
          error: `Invalid entry at index ${i}: must be an object with "path" and "cap".`,
        };
      }
      const path = (entry as Record<string, unknown>)["path"];
      const cap = (entry as Record<string, unknown>)["cap"];
      if (typeof path !== "string") {
        return {
          captions: {},
          error: `Invalid entry at index ${i}: "path" must be a string.`,
        };
      }
      let caption: string;
      if (typeof cap === "string") {
        caption = cap;
      } else if (Array.isArray(cap) && cap.length > 0 && typeof cap[0] === "string") {
        caption = cap[0];
      } else {
        return {
          captions: {},
          error: `Invalid entry at index ${i}: "cap" must be a string or non-empty array of strings.`,
        };
      }
      captions[path] = caption;
    }
  } else if (data !== null && typeof data === "object" && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    for (const [key, value] of Object.entries(obj)) {
      if (typeof key !== "string" || typeof value !== "string") {
        return {
          captions: {},
          error: `Invalid entry for "${String(key)}": keys and values must be strings.`,
        };
      }
      captions[key] = value;
    }
  } else {
    return {
      captions: {},
      error: "videos2caption.json must be an array of { path, cap } or an object mapping file names to captions.",
    };
  }

  if (uploadedFileNames.length > 0) {
    const unknownRefs = Object.keys(captions).filter((k) => !uploadedSet.has(k));
    if (unknownRefs.length > 0) {
      return {
        captions: {},
        error: `videos2caption.json references file(s) not in the uploaded videos: ${unknownRefs.slice(0, 5).join(", ")}${unknownRefs.length > 5 ? "…" : ""}.`,
      };
    }
  }

  return { captions, error: null };
}

/** Parse .txt caption file: one caption per line, order matches videos in alphabetical order. */
function parseCaptionTxt(
  text: string,
  uploadedFileNames: string[]
): { captions: Record<string, string>; error: string | null } {
  if (uploadedFileNames.length === 0) {
    return {
      captions: {},
      error: "Upload videos first so captions can be matched by order (one line per video, alphabetical).",
    };
  }
  const lines = text.split(/\r?\n/).map((s) => s.trim());
  const sortedNames = [...uploadedFileNames].sort();
  const captions: Record<string, string> = {};
  for (let i = 0; i < sortedNames.length; i++) {
    captions[sortedNames[i]] = i < lines.length ? lines[i] : "";
  }
  return { captions, error: null };
}

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
  const [captionMap, setCaptionMap] = useState<Record<string, string> | null>(null);
  const [captionFileName, setCaptionFileName] = useState<string | null>(null);

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
      setCaptionMap(null);
      setCaptionFileName(null);
      onClose();
    }
  };

  const handleCaptionFileChange = async (files: File[]) => {
    setValidationError(null);
    setCaptionMap(null);
    setCaptionFileName(null);
    if (files.length === 0) return;
    const file = files[0];
    const lower = file.name.toLowerCase();
    const isJson = lower.endsWith(".json");
    const isTxt = lower.endsWith(".txt");
    if (!isJson && !isTxt) {
      setValidationError("Caption file must be .json (e.g. videos2caption.json) or .txt (one caption per line, alphabetical order).");
      return;
    }
    let text: string;
    try {
      text = await file.text();
    } catch {
      setValidationError("Could not read the caption file.");
      return;
    }
    const uploaded = fileNames.length > 0 ? fileNames : [];
    const { captions, error } = isTxt
      ? parseCaptionTxt(text, uploaded)
      : parseVideos2Caption(text, uploaded);
    if (error) {
      setValidationError(error);
      return;
    }
    setCaptionMap(captions);
    setCaptionFileName(file.name);
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

    if (captionFileName && !captionMap) {
      setValidationError("Caption file has errors. Fix or remove it before creating the dataset.");
      return;
    }

    if (captionMap && Object.keys(captionMap).length > 0) {
      const missing = fileNames.filter((fn) => !(fn in captionMap));
      if (missing.length > 0) {
        const list = missing.length <= 5
          ? missing.join(", ")
          : `${missing.slice(0, 5).join(", ")} and ${missing.length - 5} more`;
        const ok = window.confirm(
          `The caption file does not include captions for ${missing.length} video(s): ${list}. They will get empty captions. Continue?`
        );
        if (!ok) return;
      }
    }

    setIsSubmitting(true);
    try {
      await createDataset({
        name: name.trim(),
        upload_path: rawPath,
        file_names: fileNames,
        ...(captionMap && Object.keys(captionMap).length > 0 ? { captions: captionMap } : {}),
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
                  setCaptionMap(null);
                  setCaptionFileName(null);
                  setValidationError(null);
                }}
                disabled={isSubmitting}
                uploading={isUploading}
              />
            </div>
            <div className={formStyles.formRow}>
              <label>Captions (optional)</label>
              <UploadZone
                label="Upload captions (.json or .txt)"
                hint=".json: videos2caption format. .txt: one caption per line, same order as videos (A–Z)"
                accept=".json,.txt,application/json,text/plain"
                value={captionFileName ? "1" : ""}
                fileName={captionFileName ?? undefined}
                onFileChange={handleCaptionFileChange}
                onClear={() => {
                  setCaptionMap(null);
                  setCaptionFileName(null);
                  setValidationError(null);
                }}
                disabled={isSubmitting}
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
