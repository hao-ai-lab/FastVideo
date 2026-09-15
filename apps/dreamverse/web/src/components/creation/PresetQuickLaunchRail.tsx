"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

export interface StoryPresetLike {
	id: string;
	label: string;
	description?: string;
	segmentCount?: number;
	styleTag?: string;
}

interface PresetQuickLaunchRailProps {
	storyPresets: StoryPresetLike[];
	disabled?: boolean;
	onPresetGenerate: (presetId: string) => void;
}

export default function PresetQuickLaunchRail({
	storyPresets,
	disabled = false,
	onPresetGenerate,
}: PresetQuickLaunchRailProps) {
	const scrollRef = useRef<HTMLDivElement>(null);
	const [canScrollLeft, setCanScrollLeft] = useState(false);
	const [canScrollRight, setCanScrollRight] = useState(false);
	const [presetRailDragging, setPresetRailDragging] = useState(false);
	const presetDragStateRef = useRef({
		pointerId: null as number | null,
		startX: 0,
		startScrollLeft: 0,
		moved: false,
	});
	const suppressPresetClickRef = useRef(false);

	const updateScrollState = useCallback(() => {
		const el = scrollRef.current;
		if (!el) return;
		setCanScrollLeft(el.scrollLeft > 2);
		setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 2);
	}, []);

	const scrollByAmount = useCallback(
		(direction: "left" | "right") => {
			const el = scrollRef.current;
			if (!el) return;
			const delta = direction === "left" ? -220 : 220;
			el.scrollBy({ left: delta, behavior: "smooth" });
			window.setTimeout(updateScrollState, 220);
		},
		[updateScrollState],
	);

	const handlePresetWheel = useCallback(
		(event: React.WheelEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			if (!el) return;
			if (el.scrollWidth <= el.clientWidth + 1) return;

			const dominantDelta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
			if (!dominantDelta) return;

			const maxScrollLeft = Math.max(el.scrollWidth - el.clientWidth, 0);
			const nextScrollLeft = Math.min(Math.max(el.scrollLeft + dominantDelta, 0), maxScrollLeft);
			if (nextScrollLeft === el.scrollLeft) return;

			event.preventDefault();
			el.scrollLeft = nextScrollLeft;
			updateScrollState();
		},
		[updateScrollState],
	);

	const finishPresetDrag = useCallback(() => {
		presetDragStateRef.current = {
			pointerId: null,
			startX: 0,
			startScrollLeft: 0,
			moved: false,
		};
		setPresetRailDragging(false);
	}, []);

	const handlePresetPointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
		const el = scrollRef.current;
		if (!el) return;
		if (event.pointerType !== "mouse" || event.button !== 0) return;
		if (el.scrollWidth <= el.clientWidth + 1) return;

		suppressPresetClickRef.current = false;
		presetDragStateRef.current = {
			pointerId: event.pointerId,
			startX: event.clientX,
			startScrollLeft: el.scrollLeft,
			moved: false,
		};
	}, []);

	const handlePresetPointerMove = useCallback(
		(event: React.PointerEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			const dragState = presetDragStateRef.current;
			if (!el || dragState.pointerId !== event.pointerId) return;

			const deltaX = event.clientX - dragState.startX;
			if (!dragState.moved && Math.abs(deltaX) > 4) {
				dragState.moved = true;
				suppressPresetClickRef.current = true;
				setPresetRailDragging(true);
				el.setPointerCapture?.(event.pointerId);
			}
			if (!dragState.moved) return;

			event.preventDefault();
			const maxScrollLeft = Math.max(el.scrollWidth - el.clientWidth, 0);
			el.scrollLeft = Math.min(Math.max(dragState.startScrollLeft - deltaX, 0), maxScrollLeft);
			updateScrollState();
		},
		[updateScrollState],
	);

	const handlePresetPointerUp = useCallback(
		(event: React.PointerEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			if (!el || presetDragStateRef.current.pointerId !== event.pointerId) return;
			if (el.hasPointerCapture?.(event.pointerId)) {
				el.releasePointerCapture(event.pointerId);
			}
			finishPresetDrag();
		},
		[finishPresetDrag],
	);

	const handlePresetClickCapture = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
		if (!suppressPresetClickRef.current) return;
		suppressPresetClickRef.current = false;
		event.preventDefault();
		event.stopPropagation();
	}, []);

	useEffect(() => {
		updateScrollState();
	}, [storyPresets, updateScrollState]);

	useEffect(() => {
		const el = scrollRef.current;
		if (!el) return;

		const observer = new ResizeObserver(() => updateScrollState());
		observer.observe(el);
		return () => observer.disconnect();
	}, [updateScrollState]);

	if (storyPresets.length === 0) return null;

	const scrollMaskStyle =
		canScrollLeft && canScrollRight
			? {
					maskImage: "linear-gradient(to right, transparent, black 20px, black calc(100% - 20px), transparent)",
					WebkitMaskImage: "linear-gradient(to right, transparent, black 20px, black calc(100% - 20px), transparent)",
				}
			: canScrollLeft
				? {
						maskImage: "linear-gradient(to right, transparent, black 20px, black)",
						WebkitMaskImage: "linear-gradient(to right, transparent, black 20px, black)",
					}
				: canScrollRight
					? {
							maskImage: "linear-gradient(to right, black, black calc(100% - 20px), transparent)",
							WebkitMaskImage: "linear-gradient(to right, black, black calc(100% - 20px), transparent)",
						}
					: undefined;

	return (
		<div className={cn("mx-auto w-full max-w-3xl transition-opacity duration-200", disabled && "pointer-events-none opacity-40")}>
			<div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1 sm:gap-2">
				<div className="flex w-8 shrink-0 justify-center">
					{canScrollLeft ? (
						<button
							type="button"
							aria-label="Scroll suggested prompts left"
							onClick={() => scrollByAmount("left")}
							className="studio-control studio-control-press inline-flex size-8 items-center justify-center rounded-full text-muted-foreground hover-capable:hover:bg-muted/60 hover-capable:hover:text-foreground"
						>
							<ChevronLeft className="size-4" />
						</button>
					) : null}
				</div>

				<div
					ref={scrollRef}
					onScroll={updateScrollState}
					onWheel={handlePresetWheel}
					onPointerDown={handlePresetPointerDown}
					onPointerMove={handlePresetPointerMove}
					onPointerUp={handlePresetPointerUp}
					onPointerCancel={handlePresetPointerUp}
					onLostPointerCapture={finishPresetDrag}
					onClickCapture={handlePresetClickCapture}
					style={scrollMaskStyle}
					className={cn(
						"scrollbar-hidden flex gap-2 overflow-x-auto overflow-y-visible py-0.5 select-none",
						presetRailDragging ? "cursor-grabbing" : "cursor-grab",
					)}
				>
					{storyPresets.map((preset) => (
						<button
							key={preset.id}
							type="button"
							disabled={disabled}
							onClick={() => onPresetGenerate(preset.id)}
							className="studio-control studio-control-press studio-hover-surface flex w-[12.5rem] shrink-0 flex-col gap-1 rounded-xl border border-border/50 bg-card/70 px-3 py-2.5 text-left"
						>
							<span className="line-clamp-1 text-sm font-medium text-foreground">{preset.label}</span>
							{preset.description && (
								<span className="text-pretty line-clamp-2 text-xs leading-5 text-muted-foreground">{preset.description}</span>
							)}
						</button>
					))}
				</div>

				<div className="flex w-8 shrink-0 justify-center">
					{canScrollRight ? (
						<button
							type="button"
							aria-label="Scroll suggested prompts right"
							onClick={() => scrollByAmount("right")}
							className="studio-control studio-control-press inline-flex size-8 items-center justify-center rounded-full text-muted-foreground hover-capable:hover:bg-muted/60 hover-capable:hover:text-foreground"
						>
							<ChevronRight className="size-4" />
						</button>
					) : null}
				</div>
			</div>
		</div>
	);
}
