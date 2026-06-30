'use client';

import * as React from 'react';

import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';

export interface UploadZoneProps {
  label: string;
  hint?: string;
  accept?: string;
  multiple?: boolean;
  directory?: boolean;
  allowBothFileAndDirectory?: boolean;
  value?: string;
  fileName?: string;
  onFiles?: (files: File[]) => void;
  onClear?: () => void;
  disabled?: boolean;
  uploading?: boolean;
  textInput?: boolean;
  textValue?: string;
  onTextChange?: (value: string) => void;
  textPlaceholder?: string;
}

const linkButtonClass =
  'cursor-pointer bg-transparent p-0 text-accent hover:underline disabled:cursor-not-allowed disabled:no-underline disabled:opacity-50';

export default function UploadZone({
  label,
  hint,
  accept,
  multiple = false,
  directory = false,
  allowBothFileAndDirectory = false,
  value = '',
  fileName,
  onFiles,
  onClear,
  disabled = false,
  uploading = false,
  textInput = false,
  textValue = '',
  onTextChange,
  textPlaceholder,
}: UploadZoneProps) {
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const directoryInputRef = React.useRef<HTMLInputElement>(null);

  const useBoth = directory && allowBothFileAndDirectory;
  const clickable = !textInput && !useBoth;
  const hasContent = textInput ? !!textValue.trim() : !!(value || fileName);

  // `webkitdirectory` is a DOM property without a typed JSX prop, so set it
  // imperatively. The primary input only selects folders when it is the sole
  // input; the dedicated directory input always does.
  React.useEffect(() => {
    if (fileInputRef.current) {
      fileInputRef.current.webkitdirectory = directory && !allowBothFileAndDirectory;
    }
    if (directoryInputRef.current) {
      directoryInputRef.current.webkitdirectory = true;
    }
  }, [directory, allowBothFileAndDirectory]);

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFiles?.(Array.from(files));
    }
    e.target.value = '';
  }

  function handleClick() {
    if (!textInput && !disabled) {
      fileInputRef.current?.click();
    }
  }

  function handleKeyActivate(e: React.KeyboardEvent) {
    if ((e.target as HTMLElement).closest('button')) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  }

  function handleDrop(e: React.DragEvent) {
    if (textInput || disabled) return;
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onFiles?.(Array.from(files));
    }
  }

  function handleDragOver(e: React.DragEvent) {
    if (textInput || disabled) return;
    e.preventDefault();
  }

  function clearInputs() {
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (directoryInputRef.current) directoryInputRef.current.value = '';
  }

  return (
    <div
      className={cn(
        'flex min-h-[150px] grow flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-white/[0.02] px-5 py-6 text-center transition-colors hover:border-accent hover:bg-white/[0.04]',
        clickable ? 'cursor-pointer' : 'cursor-default',
        hasContent && 'border-solid border-accent',
      )}
      onClick={clickable ? handleClick : undefined}
      onKeyDown={clickable ? handleKeyActivate : undefined}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
    >
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept={accept}
        multiple={multiple}
        onChange={handleChange}
        disabled={disabled}
      />
      {useBoth && (
        <input
          ref={directoryInputRef}
          type="file"
          className="hidden"
          multiple
          onChange={handleChange}
          disabled={disabled}
        />
      )}

      <div className="mb-2 text-sm text-muted-foreground">{label}</div>

      {textInput ? (
        <Input
          type="text"
          value={textValue}
          onChange={(e) => onTextChange?.(e.target.value)}
          placeholder={textPlaceholder}
          disabled={disabled}
        />
      ) : (
        <>
          {!hasContent && (
            <span className="mt-1.5 text-xs text-muted-foreground">
              {uploading ? (
                'Uploading…'
              ) : useBoth ? (
                <>
                  <span
                    role="button"
                    tabIndex={0}
                    className="cursor-pointer text-accent hover:underline"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleClick();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleClick();
                      }
                    }}
                  >
                    Select files
                  </span>
                  {' · '}
                  <button
                    type="button"
                    className={linkButtonClass}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      if (!disabled) directoryInputRef.current?.click();
                    }}
                    disabled={disabled}
                  >
                    Select folder
                  </button>
                </>
              ) : directory ? (
                'Click or drop folder'
              ) : (
                'Click or drop file(s)'
              )}
            </span>
          )}
          {fileName && (
            <div className="mt-2 text-sm text-foreground">
              {fileName}
              {onClear && (
                <>
                  {' · '}
                  <button
                    type="button"
                    className={linkButtonClass}
                    onClick={(e) => {
                      e.stopPropagation();
                      onClear();
                      clearInputs();
                    }}
                    disabled={disabled || uploading}
                  >
                    Clear
                  </button>
                </>
              )}
            </div>
          )}
        </>
      )}

      {hint && <div className="mt-1.5 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}
