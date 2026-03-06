'use client';

import { createDataset, uploadRawDataset } from "@/lib/api";
import { useEffect, useMemo, useState } from "react";
import modalStyles from "./styles/Modal.module.css";
import formStyles from "./styles/Form.module.css";
import cardStyles from "./styles/Card.module.css";
import buttonStyles from "./styles/Button.module.css";
import UploadZone from "./UploadZone";
import TabSwitch from "./TabSwitch";

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

/** Parse videos.txt + captions.txt: line i in videos = path, line i in captions = caption. */
function parseVideosCaptionsTxt(
  videosLines: string[],
  captionsLines: string[],
  uploadedFileNames: string[]
): { captions: Record<string, string>; error: string | null } {
  const captions: Record<string, string> = {};
  const len = Math.min(videosLines.length, captionsLines.length);
  for (let i = 0; i < len; i++) {
    const path = videosLines[i].trim();
    if (path) captions[path] = captionsLines[i].trim();
  }
  if (uploadedFileNames.length > 0) {
    const uploadedSet = new Set(uploadedFileNames);
    const unknownRefs = Object.keys(captions).filter((k) => !uploadedSet.has(k));
    if (unknownRefs.length > 0) {
      return {
        captions: {},
        error: `videos.txt references file(s) not in the uploaded videos: ${unknownRefs.slice(0, 5).join(", ")}${unknownRefs.length > 5 ? "…" : ""}.`,
      };
    }
  }
  return { captions, error: null };
}

/** Parse CSV with video_name,caption columns. */
function parseCaptionCsv(
  text: string,
  uploadedFileNames: string[]
): { captions: Record<string, string>; error: string | null } {
  const lines = text.split(/\r?\n/).filter((s) => s.trim());
  if (lines.length < 2) {
    return { captions: {}, error: "CSV must have a header row and at least one data row." };
  }
  const headerParts = parseCsvLine(lines[0]).map((s) => s.trim().toLowerCase());
  const vidIdx = headerParts.includes("video_name") ? headerParts.indexOf("video_name") : 0;
  const capIdx = headerParts.includes("caption") ? headerParts.indexOf("caption") : 1;
  const captions: Record<string, string> = {};
  for (let i = 1; i < lines.length; i++) {
    const row = parseCsvLine(lines[i]);
    const path = row[vidIdx]?.trim();
    const cap = row[capIdx]?.trim() ?? "";
    if (path) captions[path] = cap;
  }
  if (uploadedFileNames.length > 0) {
    const uploadedSet = new Set(uploadedFileNames);
    const unknownRefs = Object.keys(captions).filter((k) => !uploadedSet.has(k));
    if (unknownRefs.length > 0) {
      return {
        captions: {},
        error: `CSV references file(s) not in the uploaded videos: ${unknownRefs.slice(0, 5).join(", ")}${unknownRefs.length > 5 ? "…" : ""}.`,
      };
    }
  }
  return { captions, error: null };
}

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if ((c === "," && !inQuotes) || c === "\n") {
      out.push(cur);
      cur = "";
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
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
  const [captionFormat, setCaptionFormat] = useState<"json" | "txt" | "csv">("json");
  const [captionMap, setCaptionMap] = useState<Record<string, string> | null>(null);
  const [captionFileName, setCaptionFileName] = useState<string | null>(null);
  const [videosTxtLines, setVideosTxtLines] = useState<string[] | null>(null);
  const [videosTxtFileName, setVideosTxtFileName] = useState<string | null>(null);
  const [captionsTxtLines, setCaptionsTxtLines] = useState<string[] | null>(null);
  const [captionsTxtFileName, setCaptionsTxtFileName] = useState<string | null>(null);

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
      setCaptionFormat("json");
      setCaptionMap(null);
      setCaptionFileName(null);
      setVideosTxtLines(null);
      setVideosTxtFileName(null);
      setCaptionsTxtLines(null);
      setCaptionsTxtFileName(null);
      onClose();
    }
  };

  const txtCaptionMap = useMemo(() => {
    if (!videosTxtLines || !captionsTxtLines) return null;
    const { captions, error } = parseVideosCaptionsTxt(
      videosTxtLines,
      captionsTxtLines,
      fileNames
    );
    if (error) return null;
    return Object.keys(captions).length > 0 ? captions : null;
  }, [videosTxtLines, captionsTxtLines, fileNames]);

  useEffect(() => {
    if (
      captionFormat === "txt" &&
      videosTxtLines &&
      captionsTxtLines &&
      fileNames.length > 0
    ) {
      const { error } = parseVideosCaptionsTxt(
        videosTxtLines,
        captionsTxtLines,
        fileNames
      );
      setValidationError(error ?? null);
    }
  }, [captionFormat, videosTxtLines, captionsTxtLines, fileNames]);

  const effectiveCaptionMap =
    captionFormat === "txt" ? txtCaptionMap : captionMap;

  const handleCaptionJsonChange = async (files: File[]) => {
    setValidationError(null);
    setCaptionMap(null);
    setCaptionFileName(null);
    if (files.length === 0) return;
    const file = files[0];
    let text: string;
    try {
      text = await file.text();
    } catch {
      setValidationError("Could not read the file.");
      return;
    }
    const uploaded = fileNames.length > 0 ? fileNames : [];
    const { captions, error } = parseVideos2Caption(text, uploaded);
    if (error) {
      setValidationError(error);
      return;
    }
    setCaptionMap(captions);
    setCaptionFileName(file.name);
  };

  const handleVideosTxtChange = async (files: File[]) => {
    setValidationError(null);
    setVideosTxtLines(null);
    setVideosTxtFileName(null);
    if (files.length === 0) return;
    try {
      const text = await files[0].text();
      const lines = text.split(/\r?\n/).map((s) => s.trim());
      setVideosTxtLines(lines);
      setVideosTxtFileName(files[0].name);
      if (captionsTxtLines && fileNames.length > 0) {
        const { error } = parseVideosCaptionsTxt(
          lines,
          captionsTxtLines,
          fileNames
        );
        if (error) setValidationError(error);
      }
    } catch {
      setValidationError("Could not read videos.txt.");
    }
  };

  const handleCaptionsTxtChange = async (files: File[]) => {
    setValidationError(null);
    setCaptionsTxtLines(null);
    setCaptionsTxtFileName(null);
    if (files.length === 0) return;
    try {
      const text = await files[0].text();
      const lines = text.split(/\r?\n/).map((s) => s.trim());
      setCaptionsTxtLines(lines);
      setCaptionsTxtFileName(files[0].name);
      if (videosTxtLines && fileNames.length > 0) {
        const { error } = parseVideosCaptionsTxt(
          videosTxtLines,
          lines,
          fileNames
        );
        if (error) setValidationError(error);
      }
    } catch {
      setValidationError("Could not read captions.txt.");
    }
  };

  const handleCaptionCsvChange = async (files: File[]) => {
    setValidationError(null);
    setCaptionMap(null);
    setCaptionFileName(null);
    if (files.length === 0) return;
    const file = files[0];
    let text: string;
    try {
      text = await file.text();
    } catch {
      setValidationError("Could not read the file.");
      return;
    }
    const uploaded = fileNames.length > 0 ? fileNames : [];
    const { captions, error } = parseCaptionCsv(text, uploaded);
    if (error) {
      setValidationError(error);
      return;
    }
    setCaptionMap(captions);
    setCaptionFileName(file.name);
  };

  const handleCaptionFormatChange = (format: string) => {
    setCaptionFormat(format as "json" | "txt" | "csv");
    setValidationError(null);
    setCaptionMap(null);
    setCaptionFileName(null);
    setVideosTxtLines(null);
    setVideosTxtFileName(null);
    setCaptionsTxtLines(null);
    setCaptionsTxtFileName(null);
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

    if (captionFormat === "json" || captionFormat === "csv") {
      if (captionFileName && !captionMap) {
        setValidationError("Caption file has errors. Fix or remove it before creating the dataset.");
        return;
      }
    } else if (captionFormat === "txt" && (videosTxtFileName || captionsTxtFileName)) {
      if (!videosTxtFileName || !captionsTxtFileName) {
        setValidationError("Upload both videos.txt and captions.txt to use TXT captions.");
        return;
      }
      if (validationError) return;
    }

    const finalCaptionMap = effectiveCaptionMap && Object.keys(effectiveCaptionMap).length > 0 ? effectiveCaptionMap : null;
    if (finalCaptionMap) {
      const missing = fileNames.filter((fn) => !(fn in finalCaptionMap));
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
        ...(finalCaptionMap ? { captions: finalCaptionMap } : {}),
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
              <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
                <label style={{ marginBottom: 0 }}>Captions (optional)</label>
                <TabSwitch
                  options={[
                    { id: "json", label: "JSON" },
                    { id: "txt", label: "TXT" },
                    { id: "csv", label: "CSV" },
                  ]}
                  value={captionFormat}
                  onChange={handleCaptionFormatChange}
                  disabled={isSubmitting}
                />
              </div>
              {captionFormat === "json" && (
                <UploadZone
                  label="Upload videos2caption.json"
                  hint="Array of { path, cap } or object mapping file names to captions"
                  accept=".json,application/json"
                  value={captionFileName ? "1" : ""}
                  fileName={captionFileName ?? undefined}
                  onFileChange={handleCaptionJsonChange}
                  onClear={() => {
                    setCaptionMap(null);
                    setCaptionFileName(null);
                    setValidationError(null);
                  }}
                  disabled={isSubmitting}
                />
              )}
              {captionFormat === "txt" && (
                <>
                  <div className={formStyles.row} style={{gap: '15px'}}>
                  <UploadZone
                    label="Upload videos.txt"
                    hint="One video path per line (same order as captions.txt)"
                    accept=".txt,text/plain"
                    value={videosTxtFileName ? "1" : ""}
                    fileName={videosTxtFileName ?? undefined}
                    onFileChange={handleVideosTxtChange}
                    onClear={() => {
                      setVideosTxtLines(null);
                      setVideosTxtFileName(null);
                      setValidationError(null);
                    }}
                    disabled={isSubmitting}
                  />
                  <UploadZone
                    label="Upload captions.txt"
                    hint="One caption per line (same order as videos.txt)"
                    accept=".txt,text/plain"
                    value={captionsTxtFileName ? "1" : ""}
                    fileName={captionsTxtFileName ?? undefined}
                    onFileChange={handleCaptionsTxtChange}
                    onClear={() => {
                      setCaptionsTxtLines(null);
                      setCaptionsTxtFileName(null);
                      setValidationError(null);
                    }}
                    disabled={isSubmitting}
                  />
                  </div>
                </>
              )}
              {captionFormat === "csv" && (
                <UploadZone
                  label="Upload captions.csv"
                  hint="CSV with video_name and caption columns"
                  accept=".csv,text/csv"
                  value={captionFileName ? "1" : ""}
                  fileName={captionFileName ?? undefined}
                  onFileChange={handleCaptionCsvChange}
                  onClear={() => {
                    setCaptionMap(null);
                    setCaptionFileName(null);
                    setValidationError(null);
                  }}
                  disabled={isSubmitting}
                />
              )}
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
                disabled={isSubmitting || isUploading || fileNames.length === 0 || !name.trim()}
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
