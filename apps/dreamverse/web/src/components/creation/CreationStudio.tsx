"use client";

import React from "react";

import AppNavRail, { type AppNavSection } from "@/components/creation/AppNavRail";
import CreationComposer from "@/components/creation/CreationComposer";
import PresetQuickLaunchRail, { type StoryPresetLike } from "@/components/creation/PresetQuickLaunchRail";
import {
	type AspectRatioId,
	type CreationModeId,
	type CreationModelId,
	type MentionOption,
	type ResolutionId,
} from "@/lib/creationConfig";

interface CreationStudioProps {
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
	storyPresets?: StoryPresetLike[];
	activeSection?: AppNavSection;
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
	onPresetGenerate?: (presetId: string) => void;
	onSpeechTranscript?: (text: string) => void;
	onSpeechInterimChange?: (text: string) => void;
	onOpenProjects?: () => void;
}

export default function CreationStudio({
	activeSection = "create",
	onOpenProjects,
	storyPresets = [],
	onPresetGenerate,
	isGenerating = false,
	...composerProps
}: CreationStudioProps) {
	return (
		<div className="flex min-h-0 flex-1">
			<AppNavRail activeSection={activeSection} onOpenProjects={onOpenProjects} />
			<div className="min-w-0 flex-1 overflow-y-auto">
				<div className="mx-auto flex w-full max-w-5xl flex-col gap-5 px-4 py-7 sm:px-6 sm:py-8">
					<CreationComposer {...composerProps} isGenerating={isGenerating} />
					{storyPresets.length > 0 && onPresetGenerate && (
						<PresetQuickLaunchRail storyPresets={storyPresets} disabled={isGenerating} onPresetGenerate={onPresetGenerate} />
					)}
				</div>
			</div>
		</div>
	);
}
