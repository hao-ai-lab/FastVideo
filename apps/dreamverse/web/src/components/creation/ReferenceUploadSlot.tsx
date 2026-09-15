"use client";

import React, { useRef, useState } from "react";
import { ImagePlus } from "lucide-react";

import { REFERENCE_ACCEPT, isReferenceMediaFile } from "@/lib/creationConfig";
import { cn } from "@/lib/utils";

interface ReferenceUploadSlotProps {
	label: string;
	sublabel?: string;
	previewUrl?: string | null;
	required?: boolean;
	optional?: boolean;
	disabled?: boolean;
	onSelect?: (file: File | null) => void;
}

export default function ReferenceUploadSlot({
	label,
	sublabel,
	previewUrl = null,
	required = false,
	optional = false,
	disabled = false,
	onSelect,
}: ReferenceUploadSlotProps) {
	const fileInputRef = useRef<HTMLInputElement>(null);
	const [dragActive, setDragActive] = useState(false);

	function handleFile(file: File | null) {
		if (!file || !isReferenceMediaFile(file)) return;
		onSelect?.(file);
	}

	return (
		<div className="flex flex-col gap-1">
			<button
				type="button"
				aria-label={[label, sublabel].filter(Boolean).join(" ")}
				onClick={() => fileInputRef.current?.click()}
				disabled={disabled}
				onDragEnter={(event) => {
					event.preventDefault();
					event.stopPropagation();
					if (!disabled) setDragActive(true);
				}}
				onDragOver={(event) => {
					event.preventDefault();
					event.stopPropagation();
					if (!disabled) setDragActive(true);
				}}
				onDragLeave={(event) => {
					event.preventDefault();
					event.stopPropagation();
					setDragActive(false);
				}}
				onDrop={(event) => {
					event.preventDefault();
					event.stopPropagation();
					setDragActive(false);
					if (disabled) return;
					handleFile(event.dataTransfer.files?.[0] ?? null);
				}}
				className={cn(
					"studio-control studio-control-press studio-hover-surface relative flex size-[76px] shrink-0 flex-col items-center justify-center gap-1 overflow-hidden rounded-2xl border border-dashed bg-muted/50 px-1 text-center text-[11px] font-medium text-muted-foreground",
					required && !previewUrl ? "border-amber-500/50" : "border-border/60",
					dragActive && "border-accent-blue bg-accent-blue/10 ring-2 ring-accent-blue/30",
					disabled && "pointer-events-none opacity-50",
				)}
			>
				{previewUrl ? (
					<img src={previewUrl} alt="" className="studio-media-outline absolute inset-0 size-full object-cover" />
				) : (
					<>
						<ImagePlus className="size-4" />
						<span>{label}</span>
						{sublabel && <span className="text-[10px] font-normal opacity-70">{sublabel}</span>}
					</>
				)}
			</button>
			{(required || optional) && (
				<span className="text-center text-[10px] text-muted-foreground">{required ? "Required" : "Optional"}</span>
			)}
			<input
				ref={fileInputRef}
				type="file"
				accept={REFERENCE_ACCEPT}
				className="hidden"
				onChange={(event) => {
					handleFile(event.target.files?.[0] ?? null);
					event.target.value = "";
				}}
			/>
		</div>
	);
}
