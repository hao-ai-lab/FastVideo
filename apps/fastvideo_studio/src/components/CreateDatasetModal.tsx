'use client';

import * as React from 'react';

import UploadZone from '@/components/UploadZone';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { createDataset, uploadRawDataset } from '@/lib/api';
import {
  parseCaptionCsv,
  parseVideos2Caption,
  parseVideosCaptionsTxt,
} from '@/lib/captionParsing';

const ALLOWED_VIDEO_EXT = '.mp4,.webm,.avi,.mov,.mkv';

type CaptionFormat = 'json' | 'txt' | 'csv';

export interface CreateDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function CreateDatasetModal({
  isOpen,
  onClose,
  onSuccess,
}: CreateDatasetModalProps) {
  const [name, setName] = React.useState('');
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [rawPath, setRawPath] = React.useState('');
  const [fileNames, setFileNames] = React.useState<string[]>([]);
  const [isUploading, setIsUploading] = React.useState(false);
  const [validationError, setValidationError] = React.useState<string | null>(
    null,
  );
  const [captionFormat, setCaptionFormat] = React.useState<CaptionFormat>('json');
  const [captionMap, setCaptionMap] = React.useState<Record<
    string,
    string
  > | null>(null);
  const [captionFileName, setCaptionFileName] = React.useState<string | null>(
    null,
  );
  const [videosTxtLines, setVideosTxtLines] = React.useState<string[] | null>(
    null,
  );
  const [videosTxtFileName, setVideosTxtFileName] = React.useState<
    string | null
  >(null);
  const [captionsTxtLines, setCaptionsTxtLines] = React.useState<
    string[] | null
  >(null);
  const [captionsTxtFileName, setCaptionsTxtFileName] = React.useState<
    string | null
  >(null);

  const txtCaptionMap = React.useMemo(() => {
    if (!captionsTxtLines || fileNames.length === 0) return null;
    const { captions, error } = parseVideosCaptionsTxt(
      videosTxtLines ?? null,
      captionsTxtLines,
      fileNames,
    );
    if (error) return null;
    return Object.keys(captions).length > 0 ? captions : null;
  }, [captionsTxtLines, fileNames, videosTxtLines]);

  const effectiveCaptionMap =
    captionFormat === 'txt' ? txtCaptionMap : captionMap;

  // Keep the validation message in sync with the TXT inputs.
  React.useEffect(() => {
    if (captionFormat === 'txt' && captionsTxtLines && fileNames.length > 0) {
      const hasVideosTxt =
        !!videosTxtLines &&
        videosTxtLines.length > 0 &&
        videosTxtLines.some((s) => s.trim());
      if (hasVideosTxt && videosTxtLines) {
        const { error } = parseVideosCaptionsTxt(
          videosTxtLines,
          captionsTxtLines,
          fileNames,
        );
        setValidationError(error);
      } else {
        setValidationError(null);
      }
    }
  }, [captionFormat, captionsTxtLines, fileNames, videosTxtLines]);

  function resetState() {
    setName('');
    setRawPath('');
    setFileNames([]);
    setValidationError(null);
    setCaptionFormat('json');
    setCaptionMap(null);
    setCaptionFileName(null);
    setVideosTxtLines(null);
    setVideosTxtFileName(null);
    setCaptionsTxtLines(null);
    setCaptionsTxtFileName(null);
  }

  function handleClose() {
    if (isSubmitting) return;
    resetState();
    onClose();
  }

  async function handleMediaChange(files: File[]) {
    setValidationError(null);
    if (files.length === 0) {
      setRawPath('');
      setFileNames([]);
      return;
    }
    setIsUploading(true);
    try {
      const res = await uploadRawDataset(files);
      setRawPath(res.path);
      setFileNames(res.file_names);
      if (res.file_names.length === 0) {
        setValidationError(`No video files found. Allowed: ${ALLOWED_VIDEO_EXT}`);
      }
    } catch (err) {
      setRawPath('');
      setFileNames([]);
      setValidationError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }

  async function handleCaptionJsonChange(files: File[]) {
    setValidationError(null);
    setCaptionMap(null);
    setCaptionFileName(null);
    if (files.length === 0) return;
    const file = files[0];
    try {
      const text = await file.text();
      const { captions, error } = parseVideos2Caption(text, fileNames);
      if (error) {
        setValidationError(error);
        return;
      }
      setCaptionMap(captions);
      setCaptionFileName(file.name);
    } catch {
      setValidationError('Could not read the file.');
    }
  }

  async function handleCaptionCsvChange(files: File[]) {
    setValidationError(null);
    setCaptionMap(null);
    setCaptionFileName(null);
    if (files.length === 0) return;
    const file = files[0];
    try {
      const text = await file.text();
      const { captions, error } = parseCaptionCsv(text, fileNames);
      if (error) {
        setValidationError(error);
        return;
      }
      setCaptionMap(captions);
      setCaptionFileName(file.name);
    } catch {
      setValidationError('Could not read the file.');
    }
  }

  async function handleVideosTxtChange(files: File[]) {
    setValidationError(null);
    setVideosTxtLines(null);
    setVideosTxtFileName(null);
    if (files.length === 0) return;
    try {
      const text = await files[0].text();
      setVideosTxtLines(text.split(/\r?\n/).map((s) => s.trim()));
      setVideosTxtFileName(files[0].name);
    } catch {
      setValidationError('Could not read videos.txt.');
    }
  }

  async function handleCaptionsTxtChange(files: File[]) {
    setValidationError(null);
    setCaptionsTxtLines(null);
    setCaptionsTxtFileName(null);
    if (files.length === 0) return;
    try {
      const text = await files[0].text();
      setCaptionsTxtLines(text.split(/\r?\n/).map((s) => s.trim()));
      setCaptionsTxtFileName(files[0].name);
    } catch {
      setValidationError('Could not read captions.txt.');
    }
  }

  function handleCaptionFormatChange(format: CaptionFormat) {
    setCaptionFormat(format);
    setValidationError(null);
    setCaptionMap(null);
    setCaptionFileName(null);
    setVideosTxtLines(null);
    setVideosTxtFileName(null);
    setCaptionsTxtLines(null);
    setCaptionsTxtFileName(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setValidationError(null);
    if (!name.trim()) return;
    if (!rawPath || fileNames.length === 0) {
      setValidationError('No data was found. Upload at least one video.');
      return;
    }
    if (captionFormat === 'json' || captionFormat === 'csv') {
      if (captionFileName && !captionMap) {
        setValidationError(
          'Caption file has errors. Fix or remove it before creating the dataset.',
        );
        return;
      }
    } else if (captionFormat === 'txt') {
      if (videosTxtFileName || captionsTxtFileName) {
        if (!captionsTxtFileName) {
          setValidationError('Upload captions.txt to use TXT captions.');
          return;
        }
        // Re-validate against current state rather than the stale render-time
        // `validationError`, so a videos.txt mismatch both blocks submission
        // and keeps its message visible.
        const hasVideosTxt =
          !!videosTxtLines &&
          videosTxtLines.length > 0 &&
          videosTxtLines.some((s) => s.trim());
        if (hasVideosTxt && videosTxtLines) {
          const { error } = parseVideosCaptionsTxt(
            videosTxtLines,
            captionsTxtLines ?? [],
            fileNames,
          );
          if (error) {
            setValidationError(error);
            return;
          }
        }
      }
    }

    const finalCaptionMap =
      effectiveCaptionMap && Object.keys(effectiveCaptionMap).length > 0
        ? effectiveCaptionMap
        : null;
    if (finalCaptionMap) {
      const missing = fileNames.filter((fn) => !(fn in finalCaptionMap));
      if (missing.length > 0) {
        const list =
          missing.length <= 5
            ? missing.join(', ')
            : `${missing.slice(0, 5).join(', ')} and ${missing.length - 5} more`;
        const ok = window.confirm(
          `The caption file does not include captions for ${missing.length} video(s): ${list}. They will get empty captions. Continue?`,
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
      resetState();
      onClose();
    } catch (err) {
      setValidationError(
        err instanceof Error ? err.message : 'Failed to create dataset',
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) handleClose();
      }}
    >
      <DialogContent
        aria-describedby={undefined}
        className="max-h-[90vh] w-[90vw] max-w-[850px] overflow-y-auto"
      >
        <DialogHeader>
          <DialogTitle>Add Dataset — Raw</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} autoComplete="off">
          <div className="mb-3.5 flex flex-col gap-1.5">
            <Label htmlFor="add-dataset-name">Name</Label>
            <Input
              id="add-dataset-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My dataset"
              required
              disabled={isSubmitting}
            />
          </div>

          <div className="mb-3.5 flex flex-col gap-1.5">
            <Label>Videos</Label>
            <UploadZone
              label="Upload video files"
              hint="Select files or a folder (.mp4, .webm, .avi, .mov, .mkv)"
              accept={ALLOWED_VIDEO_EXT}
              multiple
              directory
              allowBothFileAndDirectory
              value={rawPath}
              fileName={
                fileNames.length > 0 ? `${fileNames.length} file(s)` : undefined
              }
              onFiles={handleMediaChange}
              onClear={() => {
                setRawPath('');
                setFileNames([]);
                setCaptionMap(null);
                setCaptionFileName(null);
                setValidationError(null);
              }}
              disabled={isSubmitting}
              uploading={isUploading}
            />
          </div>

          <div className="mb-3.5 flex flex-col gap-1.5">
            <Tabs
              value={captionFormat}
              onValueChange={(value) =>
                handleCaptionFormatChange(value as CaptionFormat)
              }
            >
              <div className="mb-1.5 flex flex-wrap items-center gap-4">
                <Label>Captions (optional)</Label>
                <TabsList>
                  <TabsTrigger value="json" disabled={isSubmitting}>
                    JSON
                  </TabsTrigger>
                  <TabsTrigger value="txt" disabled={isSubmitting}>
                    TXT
                  </TabsTrigger>
                  <TabsTrigger value="csv" disabled={isSubmitting}>
                    CSV
                  </TabsTrigger>
                </TabsList>
              </div>
              <TabsContent value="json">
                <UploadZone
                  label="Upload videos2caption.json"
                  hint="Array of { path, cap } or object mapping file names to captions"
                  accept=".json,application/json"
                  value={captionFileName ? '1' : ''}
                  fileName={captionFileName ?? undefined}
                  onFiles={handleCaptionJsonChange}
                  onClear={() => {
                    setCaptionMap(null);
                    setCaptionFileName(null);
                    setValidationError(null);
                  }}
                  disabled={isSubmitting}
                />
              </TabsContent>
              <TabsContent value="txt">
                <div className="flex flex-wrap gap-[15px] [&>*]:min-w-[200px] [&>*]:flex-1">
                  <UploadZone
                    label="Upload videos.txt (optional)"
                    hint="One video path per line, or leave empty to match captions to videos in alphabetical order"
                    accept=".txt,text/plain"
                    value={videosTxtFileName ? '1' : ''}
                    fileName={videosTxtFileName ?? undefined}
                    onFiles={handleVideosTxtChange}
                    onClear={() => {
                      setVideosTxtLines(null);
                      setVideosTxtFileName(null);
                      setValidationError(null);
                    }}
                    disabled={isSubmitting}
                  />
                  <UploadZone
                    label="Upload captions.txt"
                    hint="One caption per line (same order as videos.txt or alphabetical)"
                    accept=".txt,text/plain"
                    value={captionsTxtFileName ? '1' : ''}
                    fileName={captionsTxtFileName ?? undefined}
                    onFiles={handleCaptionsTxtChange}
                    onClear={() => {
                      setCaptionsTxtLines(null);
                      setCaptionsTxtFileName(null);
                      setValidationError(null);
                    }}
                    disabled={isSubmitting}
                  />
                </div>
              </TabsContent>
              <TabsContent value="csv">
                <UploadZone
                  label="Upload captions CSV"
                  hint="Header: video_name, caption"
                  accept=".csv,text/csv"
                  value={captionFileName ? '1' : ''}
                  fileName={captionFileName ?? undefined}
                  onFiles={handleCaptionCsvChange}
                  onClear={() => {
                    setCaptionMap(null);
                    setCaptionFileName(null);
                    setValidationError(null);
                  }}
                  disabled={isSubmitting}
                />
              </TabsContent>
            </Tabs>
          </div>

          {validationError && (
            <p className="mb-2 text-sm text-destructive">{validationError}</p>
          )}
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? 'Creating…' : 'Create Dataset'}
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
