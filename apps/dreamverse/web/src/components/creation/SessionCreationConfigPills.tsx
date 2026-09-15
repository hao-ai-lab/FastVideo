"use client";

import React from "react";
import { Box, ChevronDown, Clock, Monitor, Wand2 } from "lucide-react";

import ConfigPill from "@/components/creation/ConfigPill";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuLabel,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import {
	ASPECT_RATIOS,
	CREATION_MODELS,
	CREATION_MODES,
	RESOLUTIONS,
	type AspectRatioId,
	type CreationModeId,
	type CreationModelId,
	type ResolutionId,
	formatDurationLabel,
	formatResolutionLabel,
} from "@/lib/creationConfig";
import { cn } from "@/lib/utils";

export interface SessionCreationConfig {
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
}

interface SessionCreationConfigPillsProps extends SessionCreationConfig {
	disabled?: boolean;
	readOnly?: boolean;
	onModelChange?: (modelId: CreationModelId) => void;
	onModeChange?: (modeId: CreationModeId) => void;
	onAspectRatioChange?: (aspectRatio: AspectRatioId) => void;
	onResolutionChange?: (resolution: ResolutionId) => void;
	onDurationChange?: (durationSec: number) => void;
}

export default function SessionCreationConfigPills({
	modelId,
	modeId,
	aspectRatio,
	resolution,
	durationSec,
	disabled = false,
	readOnly = false,
	onModelChange,
	onModeChange,
	onAspectRatioChange,
	onResolutionChange,
	onDurationChange,
}: SessionCreationConfigPillsProps) {
	const selectedModel = CREATION_MODELS.find((model) => model.id === modelId) ?? CREATION_MODELS[0];
	const selectedMode = CREATION_MODES.find((mode) => mode.id === modeId) ?? CREATION_MODES[0];
	const isInteractive = !readOnly && !disabled;

	const pillClassName = cn(
		"h-9 min-h-9 px-2 text-[11px]",
		!isInteractive && "pointer-events-none opacity-70",
	);

	if (readOnly) {
		return (
			<div className="flex flex-wrap items-center gap-1.5">
				<ConfigPill disabled className={pillClassName} aria-label="Model">
					<Box className="size-3" />
					{selectedModel.label}
				</ConfigPill>
				<ConfigPill disabled className={pillClassName} aria-label="Mode">
					<Wand2 className="size-3" />
					{selectedMode.label}
				</ConfigPill>
				<ConfigPill disabled className={pillClassName} aria-label="Aspect ratio and resolution">
					<Monitor className="size-3" />
					{aspectRatio} {formatResolutionLabel(resolution)}
				</ConfigPill>
				<ConfigPill disabled className={pillClassName} aria-label="Duration">
					<Clock className="size-3" />
					{formatDurationLabel(durationSec)}
				</ConfigPill>
			</div>
		);
	}

	return (
		<div className="flex flex-wrap items-center gap-1.5">
			<DropdownMenu>
				<DropdownMenuTrigger asChild>
					<ConfigPill disabled={disabled} className={pillClassName} aria-label="Model">
						<Box className="size-3" />
						{selectedModel.label}
						<ChevronDown className="size-2.5 opacity-60" />
					</ConfigPill>
				</DropdownMenuTrigger>
				<DropdownMenuContent align="start" className="w-72">
					<DropdownMenuLabel>Model</DropdownMenuLabel>
					<DropdownMenuSeparator />
					{CREATION_MODELS.map((model) => (
						<DropdownMenuItem key={model.id} onClick={() => onModelChange?.(model.id)} className="flex-col items-start gap-1 py-2.5">
							<span className="flex items-center gap-2 text-sm font-medium">
								{model.label}
								{model.badge && <span className="rounded-full bg-accent-blue/15 px-1.5 py-0.5 text-[10px] text-accent-blue">{model.badge}</span>}
							</span>
							<span className="text-xs text-muted-foreground">{model.description}</span>
						</DropdownMenuItem>
					))}
				</DropdownMenuContent>
			</DropdownMenu>

			<DropdownMenu>
				<DropdownMenuTrigger asChild>
					<ConfigPill disabled={disabled} className={pillClassName} aria-label="Mode">
						<Wand2 className="size-3" />
						{selectedMode.label}
						<ChevronDown className="size-2.5 opacity-60" />
					</ConfigPill>
				</DropdownMenuTrigger>
				<DropdownMenuContent align="start" className="w-64">
					<DropdownMenuLabel>Mode</DropdownMenuLabel>
					<DropdownMenuSeparator />
					{CREATION_MODES.map((mode) => (
						<DropdownMenuItem key={mode.id} onClick={() => onModeChange?.(mode.id)} className="flex-col items-start gap-1 py-2.5">
							<span className="text-sm font-medium">{mode.label}</span>
							<span className="text-xs text-muted-foreground">{mode.description}</span>
						</DropdownMenuItem>
					))}
				</DropdownMenuContent>
			</DropdownMenu>

			<Popover>
				<PopoverTrigger asChild>
					<ConfigPill disabled={disabled} className={pillClassName} aria-label="Aspect ratio and resolution">
						<Monitor className="size-3" />
						{aspectRatio} {formatResolutionLabel(resolution)}
					</ConfigPill>
				</PopoverTrigger>
				<PopoverContent align="start" className="w-80">
					<p className="mb-3 text-xs font-medium text-muted-foreground">Aspect ratio</p>
					<div className="grid grid-cols-3 gap-2">
						{ASPECT_RATIOS.map((ratio) => (
							<button
								key={ratio}
								type="button"
								onClick={() => onAspectRatioChange?.(ratio)}
								className={cn(
									"studio-control studio-control-press studio-hover-surface flex flex-col items-center gap-2 rounded-xl border px-2 py-3 text-xs",
									aspectRatio === ratio ? "border-accent-blue bg-accent-blue/10 text-foreground" : "border-border",
								)}
							>
								<span
									className={cn(
										"rounded-sm border border-current/40 bg-muted/40",
										ratio === "9:16" && "h-7 w-4",
										ratio === "16:9" && "h-4 w-7",
										ratio === "1:1" && "size-5",
										ratio === "4:3" && "h-5 w-6",
										ratio === "3:4" && "h-6 w-5",
										ratio === "21:9" && "h-3 w-8",
									)}
								/>
								{ratio}
							</button>
						))}
					</div>
					<p className="mb-2 mt-4 text-xs font-medium text-muted-foreground">Resolution</p>
					<div className="flex flex-wrap gap-2">
						{RESOLUTIONS.map((item) => (
							<button
								key={item}
								type="button"
								onClick={() => onResolutionChange?.(item)}
								className={cn(
									"studio-control studio-control-press studio-hover-surface rounded-full border px-3 py-1.5 text-xs font-medium",
									resolution === item ? "border-accent-blue bg-accent-blue/10 text-foreground" : "border-border",
								)}
							>
								{formatResolutionLabel(item)}
							</button>
						))}
					</div>
				</PopoverContent>
			</Popover>

			<Popover>
				<PopoverTrigger asChild>
					<ConfigPill disabled={disabled} className={pillClassName} aria-label="Duration">
						<Clock className="size-3" />
						{formatDurationLabel(durationSec)}
					</ConfigPill>
				</PopoverTrigger>
				<PopoverContent align="start" className="w-72">
					<p className="mb-3 text-xs font-medium text-muted-foreground">Total duration</p>
					<Slider min={5} max={15} step={5} value={[durationSec]} onValueChange={(values) => onDurationChange?.(values[0] ?? 5)} />
					<div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground">
						<span>5s</span>
						<span className="rounded-md border border-border px-2 py-1 text-xs font-medium text-foreground">{formatDurationLabel(durationSec)}</span>
						<span>15s</span>
					</div>
				</PopoverContent>
			</Popover>
		</div>
	);
}
