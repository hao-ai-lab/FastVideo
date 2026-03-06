"use client";

import { useRef } from "react";
import uploadStyles from "./styles/UploadZone.module.css";
import formStyles from "./styles/Form.module.css";

interface UploadZoneProps {
	label: string;
	hint?: string;
	accept?: string;
	multiple?: boolean;
	directory?: boolean;
	/** When true with directory, show both file picker and folder picker options */
	allowBothFileAndDirectory?: boolean;
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
	allowBothFileAndDirectory = false,
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
	const fileInputRef = useRef<HTMLInputElement>(null);
	const directoryInputRef = useRef<HTMLInputElement>(null);
	const useBoth = directory && allowBothFileAndDirectory;

	const hasContent = textInput ? !!textValue.trim() : !!(value || fileName);

	const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const files = e.target.files;
		if (files && files.length > 0) {
			onFileChange?.(Array.from(files));
		}
		e.target.value = "";
	};

	const handleClick = () => {
		if (!textInput && !disabled) {
			fileInputRef.current?.click();
		}
	};

	const clearInputs = () => {
		if (fileInputRef.current) fileInputRef.current.value = "";
		if (directoryInputRef.current) directoryInputRef.current.value = "";
	};

	const handleZoneClick = useBoth ? undefined : () => handleClick();

	return (
		<div
			className={`${uploadStyles.uploadZone} ${hasContent ? uploadStyles.hasFile : ""} ${useBoth ? uploadStyles.allowBoth : ""} ${className ?? ""}`.trim()}
			style={style}
			onClick={!textInput ? handleZoneClick : undefined}
			role={!textInput && !useBoth ? "button" : undefined}
			tabIndex={!textInput && !useBoth ? 0 : undefined}
			onKeyDown={
				!textInput && !useBoth
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
				ref={fileInputRef}
				type="file"
				accept={accept}
				multiple={multiple}
				{...(directory &&
					!allowBothFileAndDirectory && { webkitdirectory: "" })}
				onChange={handleChange}
				disabled={disabled}
			/>
			{useBoth && (
				<input
					ref={directoryInputRef}
					type="file"
					multiple
					{...{ webkitdirectory: "" }}
					onChange={handleChange}
					disabled={disabled}
				/>
			)}
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
							{uploading ? "Uploading…" : null}
							{!uploading && useBoth && (
								<>
									<span
										role="button"
										tabIndex={0}
										className={
											uploadStyles.selectFilesTrigger
										}
										onClick={(e) => {
											e.stopPropagation();
											handleClick();
										}}
										onKeyDown={(e) => {
											if (
												e.key === "Enter" ||
												e.key === " "
											) {
												e.preventDefault();
												e.stopPropagation();
												handleClick();
											}
										}}
									>
										Select files
									</span>
									{" · "}
									<button
										type="button"
										className={formStyles.clearLink}
										onClick={(e) => {
											e.preventDefault();
											e.stopPropagation();
											if (!disabled)
												directoryInputRef.current?.click();
										}}
										disabled={disabled}
									>
										Select folder
									</button>
								</>
							)}
							{!uploading &&
								!useBoth &&
								directory &&
								"Click or drop folder"}
							{!uploading &&
								!useBoth &&
								!directory &&
								"Click or drop file(s)"}
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
			{hint && <div className={uploadStyles.hint}>{hint}</div>}
		</div>
	);
}
