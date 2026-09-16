"use client";

import React, { useMemo, useRef, useState } from "react";
import { ArrowUp, Box, ChevronDown, Clock, Monitor, Wand2 } from "lucide-react";

import ConfigPill from "@/components/creation/ConfigPill";
import HeroTagline from "@/components/HeroTagline";
import ReferenceUploadSlot from "@/components/creation/ReferenceUploadSlot";
import { Button } from "@/components/ui/button";
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
import SpeechToTextButton from "@/components/SpeechToTextButton";
import {
	ASPECT_RATIOS,
	CREATION_MODELS,
	CREATION_MODES,
	RESOLUTIONS,
	UNSUPPORTED_CREATION_MODES,
	UNSUPPORTED_RESOLUTIONS,
	modeRequiresReference,
	modeUsesDualFrames,
	type AspectRatioId,
	type CreationModeId,
	type CreationModelId,
	type MentionOption,
	type ResolutionId,
	formatDurationLabel,
	formatResolutionLabel,
} from "@/lib/creationConfig";
import {
	DEFAULT_LOBBY_CAPABILITIES_BUNDLE,
	isSupportedCreationMode,
	isSupportedResolution,
	resolveModelCapabilities,
	unsupportedModeNotice,
	type LobbyCreationCapabilities,
} from "@/lib/creationCapabilities";
import { cn } from "@/lib/utils";

const PROMPT_MAX_LENGTH = 500;

interface CreationComposerProps {
	value: string;
	disabled?: boolean;
	isGenerating?: boolean;
	canSubmit?: boolean;
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
	referencePreviewUrl?: string | null;
	firstFramePreviewUrl?: string | null;
	lastFramePreviewUrl?: string | null;
	mentionOptions?: MentionOption[];
	onValueChange: (value: string) => void;
	onSubmit: () => void;
	onKeyDown?: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
	onModelChange: (modelId: CreationModelId) => void;
	onModeChange: (modeId: CreationModeId) => void;
	onAspectRatioChange: (aspectRatio: AspectRatioId) => void;
	onResolutionChange: (resolution: ResolutionId) => void;
	onDurationChange: (durationSec: number) => void;
	onReferenceSelect?: (file: File | null) => void;
	onFirstFrameSelect?: (file: File | null) => void;
	onLastFrameSelect?: (file: File | null) => void;
	onSpeechTranscript?: (text: string) => void;
	onSpeechInterimChange?: (text: string) => void;
	capabilities?: LobbyCreationCapabilities;
}

export default function CreationComposer({
	value,
	disabled = false,
	isGenerating = false,
	canSubmit = false,
	modelId,
	modeId,
	aspectRatio,
	resolution,
	durationSec,
	referencePreviewUrl = null,
	firstFramePreviewUrl = null,
	lastFramePreviewUrl = null,
	mentionOptions = [],
	onValueChange,
	onSubmit,
	onKeyDown,
	onModelChange,
	onModeChange,
	onAspectRatioChange,
	onResolutionChange,
	onDurationChange,
	onReferenceSelect,
	onFirstFrameSelect,
	onLastFrameSelect,
	onSpeechTranscript,
	onSpeechInterimChange,
	capabilities = resolveModelCapabilities(DEFAULT_LOBBY_CAPABILITIES_BUNDLE, modelId),
}: CreationComposerProps) {
	const inputRef = useRef<HTMLTextAreaElement>(null);
	const [sttBusy, setSttBusy] = useState(false);
	const [mentionQuery, setMentionQuery] = useState("");
	const [mentionOpen, setMentionOpen] = useState(false);
	const [mentionStart, setMentionStart] = useState<number | null>(null);

	const availableModels = useMemo(
		() => CREATION_MODELS.filter((model) => capabilities.model_ids.includes(model.id)),
		[capabilities.model_ids],
	);
	const availableModes = useMemo(
		() => CREATION_MODES.filter((mode) => isSupportedCreationMode(mode.id, capabilities)),
		[capabilities],
	);
	const unavailableModes = useMemo(
		() =>
			UNSUPPORTED_CREATION_MODES.filter(
				(mode) => unsupportedModeNotice(mode.id, capabilities) !== null,
			),
		[capabilities],
	);
	const availableAspectRatios = useMemo(
		() => ASPECT_RATIOS.filter((ratio) => capabilities.aspect_ratios.includes(ratio)),
		[capabilities.aspect_ratios],
	);
	const availableResolutions = useMemo(
		() => RESOLUTIONS.filter((item) => isSupportedResolution(item, capabilities)),
		[capabilities],
	);
	const unavailableResolutions = useMemo(
		() => UNSUPPORTED_RESOLUTIONS.filter((item) => !isSupportedResolution(item, capabilities)),
		[capabilities],
	);
	const durationMin = capabilities.duration_sec[0] ?? 5;
	const durationMax = capabilities.duration_sec[capabilities.duration_sec.length - 1] ?? 15;

	const selectedModel = availableModels.find((model) => model.id === modelId) ?? availableModels[0];
	const selectedMode = availableModes.find((mode) => mode.id === modeId) ?? availableModes[0];
	const usesDualFrames = modeUsesDualFrames(modeId);
	const requiresReference = modeRequiresReference(modeId);
	const referenceMissing = requiresReference && !referencePreviewUrl;
	const submitDisabled = !canSubmit || disabled || isGenerating || !value.trim() || referenceMissing;

	const filteredMentions = useMemo(() => {
		const query = mentionQuery.trim().toLowerCase();
		if (!query) return mentionOptions.slice(0, 6);
		return mentionOptions
			.filter((option) => option.label.toLowerCase().includes(query) || option.description?.toLowerCase().includes(query))
			.slice(0, 6);
	}, [mentionOptions, mentionQuery]);

	function autoResize() {
		const el = inputRef.current;
		if (!el) return;
		el.style.height = "auto";
		const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 22;
		const maxHeight = lineHeight * 4;
		el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
		el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
	}

	function updateMentionState(nextValue: string, cursorPosition: number) {
		const beforeCursor = nextValue.slice(0, cursorPosition);
		const atIndex = beforeCursor.lastIndexOf("@");
		if (atIndex === -1 || (atIndex > 0 && !/\s/.test(beforeCursor[atIndex - 1] ?? ""))) {
			setMentionOpen(false);
			setMentionStart(null);
			setMentionQuery("");
			return;
		}
		const query = beforeCursor.slice(atIndex + 1);
		if (/\s/.test(query)) {
			setMentionOpen(false);
			setMentionStart(null);
			setMentionQuery("");
			return;
		}
		setMentionStart(atIndex);
		setMentionQuery(query);
		setMentionOpen(true);
	}

	function insertMention(option: MentionOption) {
		if (mentionStart === null) return;
		const before = value.slice(0, mentionStart);
		const after = value.slice(inputRef.current?.selectionStart ?? value.length);
		const mentionText = `@${option.label} `;
		const nextValue = `${before}${mentionText}${after}`.slice(0, PROMPT_MAX_LENGTH);
		onValueChange(nextValue);
		setMentionOpen(false);
		setMentionStart(null);
		setMentionQuery("");
		requestAnimationFrame(() => {
			const el = inputRef.current;
			if (!el) return;
			const cursor = before.length + mentionText.length;
			el.focus();
			el.setSelectionRange(cursor, cursor);
			autoResize();
		});
	}

	function handleInputChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
		const nextValue = event.target.value.slice(0, PROMPT_MAX_LENGTH);
		onValueChange(nextValue);
		updateMentionState(nextValue, event.target.selectionStart ?? nextValue.length);
		requestAnimationFrame(autoResize);
	}

	function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
		if (mentionOpen && filteredMentions.length > 0) {
			if (event.key === "Tab" || (event.key === "Enter" && !event.shiftKey)) {
				event.preventDefault();
				insertMention(filteredMentions[0]);
				return;
			}
			if (event.key === "Escape") {
				setMentionOpen(false);
				return;
			}
		}
		onKeyDown?.(event);
	}

	return (
		<section className="mx-auto flex w-full max-w-3xl flex-col gap-5">
			<HeroTagline />

			<div className="rounded-[32px] border border-border/40 bg-secondary/95 p-4 shadow-[0_24px_80px_-32px_rgba(0,0,0,0.72)] backdrop-blur-xl sm:p-5">
				<div className="flex gap-3.5">
					{usesDualFrames ? (
						<div className="flex shrink-0 gap-2">
							<ReferenceUploadSlot
								label="Asset"
								sublabel="First"
								previewUrl={firstFramePreviewUrl}
								disabled={disabled}
								onSelect={onFirstFrameSelect}
							/>
							<ReferenceUploadSlot
								label="Asset"
								sublabel="Last"
								previewUrl={lastFramePreviewUrl}
								disabled={disabled}
								onSelect={onLastFrameSelect}
							/>
						</div>
					) : (
						<ReferenceUploadSlot
							label="Reference"
							previewUrl={referencePreviewUrl}
							required={requiresReference}
							optional={!requiresReference}
							disabled={disabled}
							onSelect={onReferenceSelect}
						/>
					)}

					<div className="relative min-w-0 flex-1">
						<textarea
							ref={inputRef}
							id="continuation-prompt"
							aria-label="Continuation prompt"
							value={value}
							onChange={handleInputChange}
							onKeyDown={handleKeyDown}
							onClick={(event) => updateMentionState(value, event.currentTarget.selectionStart ?? value.length)}
							placeholder="Describe your video or mention elements"
							disabled={disabled || sttBusy}
							rows={3}
							className={cn(
								"min-h-[92px] w-full resize-none bg-transparent px-0.5 text-base leading-6 text-foreground outline-none placeholder:text-muted-foreground/80 sm:text-sm",
								(disabled || sttBusy) && "cursor-not-allowed opacity-50",
							)}
						/>
						{mentionOpen && filteredMentions.length > 0 && (
							<div className="absolute left-0 right-0 top-full z-20 mt-2 overflow-hidden rounded-2xl border border-border bg-popover/95 p-1 shadow-xl backdrop-blur-md">
								<p className="px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Mention</p>
								{filteredMentions.map((option) => (
									<button
										key={option.id}
										type="button"
										onMouseDown={(event) => {
											event.preventDefault();
											insertMention(option);
										}}
										className="studio-control studio-hover-surface flex w-full items-start gap-2 rounded-xl px-2.5 py-2 text-left"
									>
										<span className="mt-0.5 rounded-md bg-accent px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
											{option.kind}
										</span>
										<span className="min-w-0">
											<span className="block truncate text-sm font-medium text-foreground">{option.label}</span>
											{option.description && <span className="block truncate text-xs text-muted-foreground">{option.description}</span>}
										</span>
									</button>
								))}
							</div>
						)}
					</div>
				</div>

				<div className="mt-4 flex flex-wrap items-center gap-1.5 rounded-2xl bg-muted/35 p-1.5 ring-1 ring-border/25">
					<DropdownMenu>
						<DropdownMenuTrigger asChild>
							<ConfigPill disabled={disabled}>
								<Box className="size-3.5" />
								{selectedModel.label}
								<ChevronDown className="size-3 opacity-60" />
							</ConfigPill>
						</DropdownMenuTrigger>
						<DropdownMenuContent align="start" className="w-72">
							<DropdownMenuLabel>Model</DropdownMenuLabel>
							<DropdownMenuSeparator />
							{availableModels.map((model) => (
								<DropdownMenuItem key={model.id} onClick={() => onModelChange(model.id)} className="flex-col items-start gap-1 py-2.5">
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
							<ConfigPill disabled={disabled}>
								<Wand2 className="size-3.5" />
								{selectedMode.label}
								<ChevronDown className="size-3 opacity-60" />
							</ConfigPill>
						</DropdownMenuTrigger>
						<DropdownMenuContent align="start" className="w-64">
							<DropdownMenuLabel>Mode</DropdownMenuLabel>
							<DropdownMenuSeparator />
							{availableModes.map((mode) => (
								<DropdownMenuItem key={mode.id} onClick={() => onModeChange(mode.id)} className="flex-col items-start gap-1 py-2.5">
									<span className="text-sm font-medium">{mode.label}</span>
									<span className="text-xs text-muted-foreground">{mode.description}</span>
								</DropdownMenuItem>
							))}
							{unavailableModes.length > 0 && <DropdownMenuSeparator />}
							{unavailableModes.map((mode) => (
								<DropdownMenuItem key={mode.id} disabled className="flex-col items-start gap-1 py-2.5 opacity-60">
									<span className="text-sm font-medium">{mode.label}</span>
									<span className="text-xs text-muted-foreground">
										{unsupportedModeNotice(mode.id, capabilities) ?? mode.description}
									</span>
								</DropdownMenuItem>
							))}
						</DropdownMenuContent>
					</DropdownMenu>

					<Popover>
						<PopoverTrigger asChild>
							<ConfigPill disabled={disabled}>
								<Monitor className="size-3.5" />
								{aspectRatio} {formatResolutionLabel(resolution)}
							</ConfigPill>
						</PopoverTrigger>
						<PopoverContent align="start" className="w-80">
							<p className="mb-3 text-xs font-medium text-muted-foreground">Aspect ratio</p>
							<div className="grid grid-cols-3 gap-2">
								{availableAspectRatios.map((ratio) => (
									<button
										key={ratio}
										type="button"
										onClick={() => onAspectRatioChange(ratio)}
										className={cn(
											"studio-control studio-control-press studio-hover-surface flex flex-col items-center gap-2 rounded-xl border px-2 py-3 text-xs",
											aspectRatio === ratio ? "border-accent-blue bg-accent-blue/10 text-foreground" : "border-border",
										)}
									>
										<span className={cn("rounded-sm border border-current/40 bg-muted/40", ratio === "9:16" && "h-7 w-4", ratio === "16:9" && "h-4 w-7", ratio === "1:1" && "size-5", ratio === "4:3" && "h-5 w-6", ratio === "3:4" && "h-6 w-5", ratio === "21:9" && "h-3 w-8")} />
										{ratio}
									</button>
								))}
							</div>
							<p className="mb-2 mt-4 text-xs font-medium text-muted-foreground">Resolution</p>
							<div className="flex flex-wrap gap-2">
								{availableResolutions.map((item) => (
									<button
										key={item}
										type="button"
										onClick={() => onResolutionChange(item)}
										className={cn(
											"studio-control studio-control-press studio-hover-surface rounded-full border px-3 py-1.5 text-xs font-medium",
											resolution === item ? "border-accent-blue bg-accent-blue/10 text-foreground" : "border-border",
										)}
									>
										{formatResolutionLabel(item)}
									</button>
								))}
								{unavailableResolutions.map((item) => (
									<button
										key={item}
										type="button"
										disabled
										className="studio-control rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground opacity-50"
										title="Not supported on FastLTX models yet"
									>
										{formatResolutionLabel(item)}
									</button>
								))}
							</div>
						</PopoverContent>
					</Popover>

					<Popover>
						<PopoverTrigger asChild>
							<ConfigPill disabled={disabled}>
								<Clock className="size-3.5" />
								{formatDurationLabel(durationSec)}
							</ConfigPill>
						</PopoverTrigger>
						<PopoverContent align="start" className="w-72">
							<p className="mb-3 text-xs font-medium text-muted-foreground">Total duration</p>
							<Slider min={durationMin} max={durationMax} step={5} value={[durationSec]} onValueChange={(values) => onDurationChange(values[0] ?? durationMin)} />
							<div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground">
								<span>{formatDurationLabel(durationMin)}</span>
								<span className="rounded-md border border-border px-2 py-1 text-xs font-medium text-foreground">{formatDurationLabel(durationSec)}</span>
								<span>{formatDurationLabel(durationMax)}</span>
							</div>
						</PopoverContent>
					</Popover>

					<div className="ml-auto flex items-center gap-1.5">
						{onSpeechTranscript && (
							<SpeechToTextButton
								disabled={disabled || isGenerating}
								onTranscript={onSpeechTranscript}
								onInterimChange={onSpeechInterimChange}
								onBusyChange={setSttBusy}
							/>
						)}
						<Button
							aria-label="Generate"
							onClick={onSubmit}
							disabled={submitDisabled}
							size="icon"
							className="studio-control-press rounded-full bg-accent-blue text-white shadow-sm hover-capable:hover:bg-accent-blue/90 disabled:bg-muted disabled:text-muted-foreground"
						>
							<ArrowUp className="size-5" />
						</Button>
					</div>
				</div>

				{referenceMissing && value.trim() && (
					<p className="mt-3 text-center text-xs leading-5 text-amber-700 dark:text-amber-400">
						Upload a reference asset to use Omni reference mode.
					</p>
				)}
			</div>
		</section>
	);
}
