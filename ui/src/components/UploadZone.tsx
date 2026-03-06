'use client';

import { useRef } from "react";
import uploadStyles from "./styles/UploadZone.module.css";
import formStyles from "./styles/Form.module.css";

interface UploadZoneProps {
  label: string;
  hint?: string;
  accept?: string;
  multiple?: boolean;
  directory?: boolean;
  value?: string;
  fileName?: string;
  onFileChange?: (files: File[]) => void;
  onClear?: () => void;
  disabled?: boolean;
  uploading?: boolean;
  /** For HuggingFace: text input instead of file */
  textInput?: boolean;
  textValue?: string;
  onTextChange?: (value: string) => void;
  textPlaceholder?: string;
  className?: string;
  style?: React.CSSProperties;
}

export default function UploadZone({
  label,
  hint,
  accept,
  multiple = false,
  directory = false,
  value,
  fileName,
  onFileChange,
  onClear,
  disabled = false,
  uploading = false,
  textInput = false,
  textValue = "",
  onTextChange,
  textPlaceholder,
  className,
  style,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const hasContent = textInput
    ? !!textValue.trim()
    : !!(value || fileName);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileChange?.(Array.from(files));
    }
    e.target.value = "";
  };

  const handleClick = () => {
    if (!textInput && !disabled) {
      inputRef.current?.click();
    }
  };

  return (
    <div
      className={`${uploadStyles.uploadZone} ${hasContent ? uploadStyles.hasFile : ""} ${className ?? ""}`.trim()}
      style={style}
      onClick={!textInput ? handleClick : undefined}
      role={!textInput ? "button" : undefined}
      tabIndex={!textInput ? 0 : undefined}
      onKeyDown={
        !textInput
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleClick();
              }
            }
          : undefined
      }
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        {...(directory && { webkitdirectory: "" })}
        onChange={handleChange}
        disabled={disabled}
      />
      <div className={uploadStyles.label}>{label}</div>
      {textInput ? (
        <input
          type="text"
          value={textValue}
          onChange={(e) => onTextChange?.(e.target.value)}
          placeholder={textPlaceholder}
          disabled={disabled}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <>
          {!hasContent && (
            <span className={uploadStyles.hint}>
              {uploading
                ? "Uploading…"
                : directory
                  ? "Click or drop folder"
                  : "Click or drop file(s)"}
            </span>
          )}
          {fileName && (
            <div className={uploadStyles.fileName}>
              {fileName}
              {onClear && (
                <>
                  {" · "}
                  <button
                    type="button"
                    className={formStyles.clearLink}
                    onClick={(e) => {
                      e.stopPropagation();
                      onClear();
                      if (inputRef.current) inputRef.current.value = "";
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
      {hint && <div className={uploadStyles.hint}>{hint}</div>}
    </div>
  );
}
