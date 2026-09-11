"use client";

import { useRef, useState } from "react";
import { ArrowDown, ArrowUp, AudioLines, Check, GripVertical, ImagePlus, Plus, Trash2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { NativeSelect } from "@/components/ui/native-select";
import { cn } from "@/lib/utils";
import type { ConditioningAsset, ConditioningRole, GenerationAsset, GenerationMode } from "@/lib/generationMode";

interface AssetListProps {
	mode: GenerationMode;
	assets: GenerationAsset[];
	conditioning: ConditioningAsset[];
	locked?: boolean;
	uploading?: boolean;
	error?: string;
	validationNotice?: string | null;
	onUpload: (files: File[]) => void;
	onAssign: (assetId: string, role: ConditioningRole) => void;
	onRemove: (assetId: string) => void;
	onUnselect: (index: number) => void;
	onMove: (from: number, to: number) => void;
	onMissing: (assetId: string) => void;
}

function AssetPreview({ asset, onMissing, compact = false }: {
	asset: GenerationAsset;
	onMissing: (assetId: string) => void;
	compact?: boolean;
}) {
	const [previewFailed, setPreviewFailed] = useState(false);
	function previewError() {
		setPreviewFailed(true);
		onMissing(asset.asset_id);
	}
	const className = cn("h-full w-full object-cover", asset.missing && "opacity-25");
	if (asset.missing) return <span className="p-2 text-center text-[10px] text-muted-foreground">Upload again</span>;
	if (previewFailed) return <span className="p-2 text-center text-[10px] text-muted-foreground">Preview unavailable</span>;
	if (asset.kind === "image") {
		return <img src={asset.url} alt={asset.name} className={className} onError={previewError} />;
	}
	if (asset.kind === "video") {
		return <video src={asset.url} aria-label={`Preview ${asset.name}`} className={className} muted playsInline controls={!compact} preload="metadata" onError={previewError} />;
	}
	return (
		<div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-violet-500/10 p-2 text-violet-500">
			<AudioLines className="size-6" />
			{!compact && <audio src={asset.url} aria-label={`Preview ${asset.name}`} controls preload="metadata" className="h-6 w-full min-w-0" onError={previewError} />}
		</div>
	);
}

/** A reusable library/picker. The parent asset store owns uploads and selection. */
export default function AssetList({
	mode, assets, conditioning, locked = false, uploading = false, error = "", validationNotice,
	onUpload, onAssign, onRemove, onUnselect, onMove, onMissing,
}: AssetListProps) {
	const inputRef = useRef<HTMLInputElement>(null);
	const [libraryOpen, setLibraryOpen] = useState(true);
	const [dragIndex, setDragIndex] = useState<number | null>(null);
	const disabled = locked || uploading;
	const imageAssets = assets.filter((asset) => asset.kind === "image");

	return (
		<section aria-label="Asset List" className="overflow-hidden rounded-2xl border border-input bg-card/70 shadow-sm backdrop-blur-sm">
			<div className="flex items-center justify-between gap-3 px-4 py-3">
				<div>
					<h2 className="text-xs font-semibold tracking-wide">{mode === "fl2va" ? "Frame guidance" : "Reference sequence"}</h2>
					<p className="mt-0.5 text-[11px] text-muted-foreground">{locked ? "Inputs are locked for this project." : mode === "fl2va" ? "Choose your opening image and, optionally, the ending." : "Arrange references in the order you want the model to read them."}</p>
				</div>
				<Button type="button" variant="outline" size="sm" disabled={disabled} onClick={() => inputRef.current?.click()} className="shrink-0 gap-1.5 rounded-full text-xs">
					<Upload className="size-3.5" />{uploading ? "Uploading…" : "Upload assets"}
				</Button>
				<input ref={inputRef} type="file" aria-label="Upload assets" className="sr-only" multiple accept={mode === "fl2va" ? "image/*" : "image/*,video/*,audio/*"} disabled={disabled} onChange={(event) => {
					const files = Array.from(event.target.files || []);
					if (files.length) onUpload(files);
					event.target.value = "";
				}} />
			</div>

			<div className="max-h-[min(42vh,350px)] overflow-y-auto px-4 pb-3">
				{mode === "fl2va" ? (
					<div className="grid grid-cols-2 gap-3">
						{(["first_frame", "last_frame"] as const).map((role) => {
							const label = role === "first_frame" ? "First frame" : "Last frame";
							const assetId = conditioning.find((item) => item.role === role)?.asset_id || "";
							const asset = assets.find((item) => item.asset_id === assetId);
							return (
								<div key={role} className="overflow-hidden rounded-xl border border-input bg-background/40 p-2">
									<div className="flex h-20 items-center justify-center overflow-hidden rounded-lg bg-muted/60 sm:h-24">
										{asset ? <AssetPreview key={asset.asset_id} asset={asset} onMissing={onMissing} /> : <ImagePlus className="size-6 text-muted-foreground/45" />}
									</div>
									<label htmlFor={`asset-${role}`} className="mb-1 mt-2 block text-[11px] font-medium">{label} <span className="font-normal text-muted-foreground">{role === "first_frame" ? "· required" : "· optional"}</span></label>
									<NativeSelect id={`asset-${role}`} aria-label={label} value={assetId} disabled={disabled} className="h-8 text-xs" onChange={(event) => onAssign(event.target.value, role)}>
										<option value="">{imageAssets.length ? "Choose an image" : "Upload an image first"}</option>
										{imageAssets.map((item) => <option key={item.asset_id} value={item.asset_id} disabled={item.missing}>{item.name}{item.missing ? " (upload again)" : ""}</option>)}
									</NativeSelect>
								</div>
							);
						})}
					</div>
				) : (
					<>
						{conditioning.length ? (
							<ol aria-label="Ordered references" className="flex flex-col gap-2">
								{conditioning.map((item, index) => {
									const asset = assets.find((entry) => entry.asset_id === item.asset_id);
									if (!asset) return null;
									return (
										<li key={`${item.asset_id}-${index}`} draggable={!disabled} onDragStart={() => setDragIndex(index)} onDragEnd={() => setDragIndex(null)} onDragOver={(event) => { if (!disabled && dragIndex !== null) event.preventDefault(); }} onDrop={(event) => { event.preventDefault(); if (!disabled && dragIndex !== null) onMove(dragIndex, index); setDragIndex(null); }} className={cn("flex items-center gap-2 rounded-xl border border-input bg-background/40 p-2", dragIndex === index && "opacity-50")}>
											<GripVertical className="hidden size-3.5 shrink-0 text-muted-foreground/50 sm:block" aria-hidden />
											<span className="w-4 text-center text-[11px] font-medium text-muted-foreground">{index + 1}</span>
											<div className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-md bg-muted"><AssetPreview asset={asset} onMissing={onMissing} compact /></div>
											<div className="min-w-0 flex-1"><p className="truncate text-xs font-medium">{asset.name}</p><p className="text-[10px] capitalize text-muted-foreground">{asset.kind}{asset.missing ? " · unavailable" : ""}</p></div>
											<Button type="button" variant="ghost" size="icon-sm" aria-label={`Move ${asset.name} up`} disabled={disabled || index === 0} onClick={() => onMove(index, index - 1)}><ArrowUp className="size-3.5" /></Button>
											<Button type="button" variant="ghost" size="icon-sm" aria-label={`Move ${asset.name} down`} disabled={disabled || index === conditioning.length - 1} onClick={() => onMove(index, index + 1)}><ArrowDown className="size-3.5" /></Button>
											<Button type="button" variant="ghost" size="icon-sm" aria-label={`Unselect ${asset.name}`} disabled={disabled} onClick={() => onUnselect(index)}><X className="size-3.5" /></Button>
										</li>
									);
								})}
							</ol>
						) : (
							<div className="flex items-center gap-3 rounded-xl border border-dashed border-input px-4 py-4 text-muted-foreground"><ImagePlus className="size-6 shrink-0 opacity-50" /><p className="text-xs">Add images, video, or audio from your asset library.<br /><span className="text-[11px] opacity-75">At least one image or video is required.</span></p></div>
						)}
						<p className="mt-2 text-[10px] text-muted-foreground">{conditioning.length}/12 selected · up to 9 images, 3 videos, 3 audio clips</p>
					</>
				)}

				{assets.length > 0 && !locked && (
					<div className="mt-3 border-t border-border/60 pt-2">
						<button type="button" className="flex w-full items-center justify-between py-1 text-[11px] font-medium text-muted-foreground" aria-expanded={libraryOpen} onClick={() => setLibraryOpen(!libraryOpen)}><span>Asset library · {assets.length}</span><span>{libraryOpen ? "Hide" : "Show"}</span></button>
						{libraryOpen && <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
							{assets.map((asset) => {
								const selected = conditioning.some((item) => item.asset_id === asset.asset_id);
								return (
									<div key={asset.asset_id} className={cn("overflow-hidden rounded-lg border bg-background/40", selected ? "border-sky-400/70" : "border-input")}>
										<div className="flex h-20 items-center justify-center overflow-hidden bg-muted/50"><AssetPreview asset={asset} onMissing={onMissing} /></div>
										<div className="flex items-center gap-1 p-1.5">
											<div className="min-w-0 flex-1"><p title={asset.name} className="truncate text-[10px] font-medium">{asset.name}</p><p className="text-[9px] capitalize text-muted-foreground">{asset.missing ? "Upload again" : `${asset.kind} · ${(asset.size / 1024 / 1024).toFixed(1)} MB`}</p></div>
											{mode === "ref2va" && <Button type="button" variant="ghost" size="icon-sm" className="size-7" aria-label={`Add ${asset.name} as reference`} disabled={disabled || selected || asset.missing || conditioning.length >= 12} onClick={() => onAssign(asset.asset_id, "reference")}>{selected ? <Check className="size-3.5 text-sky-500" /> : <Plus className="size-3.5" />}</Button>}
											<Button type="button" variant="ghost" size="icon-sm" className="size-7 text-muted-foreground" aria-label={`Remove asset ${asset.name}`} disabled={disabled} onClick={() => onRemove(asset.asset_id)}><Trash2 className="size-3" /></Button>
										</div>
									</div>
								);
							})}
						</div>}
					</div>
				)}
			</div>
			{!locked && <p className="px-4 pb-2 text-[10px] text-muted-foreground">Images ≤15 MiB / 16 MP{mode === "ref2va" ? " / 1:4–4:1 aspect ratio" : ""} · video/audio ≤100 MiB / 30 sec · video up to 4K · mono/stereo audio</p>}
			{(error || validationNotice) && <p role={error ? "alert" : "status"} className={cn("border-t border-border/60 px-4 py-2 text-[11px]", error ? "bg-rose-500/5 text-rose-600 dark:text-rose-300" : "bg-amber-500/5 text-amber-700 dark:text-amber-300")}>{error || validationNotice}</p>}
		</section>
	);
}
