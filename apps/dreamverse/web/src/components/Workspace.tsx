"use client";
import React, { useRef, useMemo, useEffect, useCallback, useState } from "react";
import { motion, useAnimationControls } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Check, Lightbulb, Pencil } from "lucide-react";

export const WORKSPACE_ORIGINAL_SELECTION_KEY = "original";
export const WORKSPACE_CURRENT_SELECTION_KEY = "current";

interface WorkspaceProps {
	promptEvents?: Record<string, any>[];
	currentThumbnail?: string | null;
	originalLabel?: string;
	sessionStarted?: boolean;
	onSelectOriginal?: () => void;
	onSelectEvent?: (event: Record<string, any>) => void;
	onSelectCurrent?: () => void;
	selectedClipId?: string;
	selectedEntryKey?: string;
	originalClipId?: string;
}

function normalizeText(value: any): string {
	return typeof value === "string" ? value.trim() : "";
}

function isEditEvent(event: Record<string, any>): boolean {
	if (!normalizeText(event?.text)) {
		return false;
	}
	return String(event?.source || "").trim() === "user_rewrite";
}

export function getWorkspaceEventSelectionKey(
	event: Record<string, any>,
	fallbackIndex = 0,
): string {
	return `edit:${String(event?.promptId || event?.clipId || fallbackIndex)}`;
}

const CHROMA_PATHS = (prefix: string) => (
	<>
		<path d="M0 584H174V295H0V584Z" fill={`url(#${prefix}0)`} />
		<path d="M174 584H348V184H174V584Z" fill={`url(#${prefix}1)`} />
		<path d="M348 584H522V104H348V584Z" fill={`url(#${prefix}2)`} />
		<path d="M522 584H697V54H522V584Z" fill={`url(#${prefix}3)`} />
		<path d="M697 584H870V0H697V584Z" fill={`url(#${prefix}4)`} />
		<path d="M870 584H1045V54H870V584Z" fill={`url(#${prefix}5)`} />
		<path d="M1045 584H1219V104H1045V584Z" fill={`url(#${prefix}6)`} />
		<path d="M1219 584H1393V184H1219V584Z" fill={`url(#${prefix}7)`} />
		<path d="M1393 584H1567V294H1393V584Z" fill={`url(#${prefix}8)`} />
	</>
);

const CHROMA_CHILD_GRADIENTS = (prefix: string) => (
	<>
		<linearGradient id={`${prefix}0`} href={`#${prefix}b`} x1="87" y1="584" x2="87" y2="295" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}1`} href={`#${prefix}b`} x1="261" y1="584" x2="261" y2="184" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}2`} href={`#${prefix}b`} x1="435" y1="584" x2="435" y2="104" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}3`} href={`#${prefix}b`} x1="609.5" y1="584" x2="609.5" y2="54" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}4`} href={`#${prefix}b`} x1="783.5" y1="584" x2="783.5" y2="0" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}5`} href={`#${prefix}b`} x1="957.5" y1="584" x2="957.5" y2="54" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}6`} href={`#${prefix}b`} x1="1132" y1="584" x2="1132" y2="104" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}7`} href={`#${prefix}b`} x1="1306" y1="584" x2="1306" y2="184" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}8`} href={`#${prefix}b`} x1="1480" y1="584" x2="1480" y2="294" gradientUnits="userSpaceOnUse" />
	</>
);

const CHROMA_PATHS_SM = (prefix: string) => (
	<>
		<path d="M0 584H224V295H0V584Z" fill={`url(#${prefix}0)`} />
		<path d="M224 584H448V184H224V584Z" fill={`url(#${prefix}1)`} />
		<path d="M448 584H672V54H448V584Z" fill={`url(#${prefix}2)`} />
		<path d="M672 584H896V0H672V584Z" fill={`url(#${prefix}3)`} />
		<path d="M896 584H1120V54H896V584Z" fill={`url(#${prefix}4)`} />
		<path d="M1120 584H1344V184H1120V584Z" fill={`url(#${prefix}5)`} />
		<path d="M1344 584H1567V295H1344V584Z" fill={`url(#${prefix}6)`} />
	</>
);

const CHROMA_CHILD_GRADIENTS_SM = (prefix: string) => (
	<>
		<linearGradient id={`${prefix}0`} href={`#${prefix}b`} x1="112" y1="584" x2="112" y2="295" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}1`} href={`#${prefix}b`} x1="336" y1="584" x2="336" y2="184" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}2`} href={`#${prefix}b`} x1="560" y1="584" x2="560" y2="54" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}3`} href={`#${prefix}b`} x1="784" y1="584" x2="784" y2="0" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}4`} href={`#${prefix}b`} x1="1008" y1="584" x2="1008" y2="54" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}5`} href={`#${prefix}b`} x1="1232" y1="584" x2="1232" y2="184" gradientUnits="userSpaceOnUse" />
		<linearGradient id={`${prefix}6`} href={`#${prefix}b`} x1="1455.5" y1="584" x2="1455.5" y2="295" gradientUnits="userSpaceOnUse" />
	</>
);

const LIGHT_GRADIENT_STOPS = (
	<>
		<stop stopColor="#162E6E" />
		<stop offset="0.10" stopColor="#356CFF" />
		<stop offset="0.18" stopColor="#60A5FA" />
		<stop offset="0.28" stopColor="#FFFFFF" />
		<stop offset="0.40" stopColor="#FEF3C7" />
		<stop offset="0.48" stopColor="#FCD34D" />
		<stop offset="0.58" stopColor="#F59E0B" />
		<stop offset="0.70" stopColor="#D97706" stopOpacity="0.35" />
		<stop offset="1" stopColor="#D97706" stopOpacity="0" />
	</>
);

const DARK_GRADIENT_STOPS = (
	<>
		<stop stopColor="#071530" />
		<stop offset="0.10" stopColor="#0E3280" />
		<stop offset="0.18" stopColor="#2260D4" />
		<stop offset="0.28" stopColor="#3596F8" />
		<stop offset="0.40" stopColor="#93BDFF" />
		<stop offset="0.48" stopColor="#DCEAFC" />
		<stop offset="0.58" stopColor="#FDECAA" />
		<stop offset="0.70" stopColor="#FBDD70" stopOpacity="0.35" />
		<stop offset="1" stopColor="#FBDD70" stopOpacity="0" />
	</>
);

function chromaSvg(prefix: string, pathsFn: (p: string) => React.ReactNode, gradientsFn: (p: string) => React.ReactNode, stops: React.ReactNode) {
	return (
		<svg className="h-full w-full" viewBox="0 0 1567 584" preserveAspectRatio="none" fill="none">
			<g clipPath={`url(#${prefix}-clip)`} filter={`url(#${prefix}-blur)`}>
				{pathsFn(prefix)}
			</g>
			<defs>
				<filter id={`${prefix}-blur`} x="-30" y="-80" width="1627" height="744" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
					<feFlood floodOpacity="0" result="BackgroundImageFix" />
					<feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
					<feGaussianBlur stdDeviation="15" result="blur" />
				</filter>
				<linearGradient id={`${prefix}b`} gradientUnits="userSpaceOnUse">
					{stops}
				</linearGradient>
				{gradientsFn(prefix)}
				<clipPath id={`${prefix}-clip`}>
					<rect width="1567" height="584" fill="white" />
				</clipPath>
			</defs>
		</svg>
	);
}

function ChromaGradient({ sessionStarted = false }: { sessionStarted?: boolean }) {
	const controls = useAnimationControls();
	const prevStarted = useRef(sessionStarted);

	useEffect(() => {
		const wasStarted = prevStarted.current;
		prevStarted.current = sessionStarted;

		if (sessionStarted && !wasStarted) {
			controls.start({
				scaleY: [1, 1.3, 0.14],
				transition: { duration: 1.0, times: [0, 0.25, 1], ease: "easeInOut" },
			});
		} else if (sessionStarted) {
			controls.start({
				scaleY: 0.14,
				transition: { duration: 0.4, ease: "easeOut" },
			});
		} else {
			controls.start({
				scaleY: 1,
				transition: { duration: 0.6, ease: "easeOut" },
			});
		}
	}, [sessionStarted, controls]);

	return (
		<>
			{/* Mobile (<md): 7 wider bars */}
			<div className="contents md:hidden">
				<motion.div
					initial={{ scaleY: 1 }}
					animate={controls}
					style={{ transformOrigin: "bottom" }}
					className="pointer-events-none fixed inset-x-0 bottom-0 z-0 h-[42vh] opacity-80 dark:hidden"
					aria-hidden="true"
				>
					{chromaSvg("ml", CHROMA_PATHS_SM, CHROMA_CHILD_GRADIENTS_SM, LIGHT_GRADIENT_STOPS)}
				</motion.div>
				<motion.div
					initial={{ scaleY: 1 }}
					animate={controls}
					style={{ transformOrigin: "bottom" }}
					className="pointer-events-none fixed inset-x-0 bottom-0 z-0 hidden h-[42vh] opacity-55 dark:block"
					aria-hidden="true"
				>
					{chromaSvg("mk", CHROMA_PATHS_SM, CHROMA_CHILD_GRADIENTS_SM, DARK_GRADIENT_STOPS)}
				</motion.div>
			</div>

			{/* Desktop (md+): 9 bars */}
			<div className="hidden md:contents">
				<motion.div
					initial={{ scaleY: 1 }}
					animate={controls}
					style={{ transformOrigin: "bottom" }}
					className="pointer-events-none fixed inset-x-0 bottom-0 z-0 h-[50vh] opacity-80 dark:hidden"
					aria-hidden="true"
				>
					{chromaSvg("cl", CHROMA_PATHS, CHROMA_CHILD_GRADIENTS, LIGHT_GRADIENT_STOPS)}
				</motion.div>
				<motion.div
					initial={{ scaleY: 1 }}
					animate={controls}
					style={{ transformOrigin: "bottom" }}
					className="pointer-events-none fixed inset-x-0 bottom-0 z-0 hidden h-[50vh] opacity-55 dark:block"
					aria-hidden="true"
				>
					{chromaSvg("cd", CHROMA_PATHS, CHROMA_CHILD_GRADIENTS, DARK_GRADIENT_STOPS)}
				</motion.div>
			</div>
		</>
	);
}

export default function Workspace({ promptEvents = [], currentThumbnail = null, originalLabel = "", sessionStarted = false, onSelectOriginal, onSelectEvent, onSelectCurrent, selectedClipId, selectedEntryKey: selectedEntryKeyProp, originalClipId = "" }: WorkspaceProps) {
	const bottomSentinelRef = useRef<HTMLDivElement>(null);
	const topSentinelRef = useRef<HTMLDivElement>(null);
	const [showTopFade, setShowTopFade] = useState(false);

	const conversationEvents = useMemo(() => {
		const all = promptEvents.filter(isEditEvent).slice().reverse();
		if (all.length > 0 && originalLabel && normalizeText(all[0].text) === originalLabel.trim()) {
			return all.slice(1);
		}
		return all;
	}, [promptEvents, originalLabel]);

	const selectedEntryKey = useMemo(() => {
		if (selectedEntryKeyProp !== undefined) return selectedEntryKeyProp;
		if (!selectedClipId) return WORKSPACE_CURRENT_SELECTION_KEY;
		if (selectedClipId === originalClipId) return WORKSPACE_ORIGINAL_SELECTION_KEY;
		const matchIndex = conversationEvents.findIndex((e) => e.clipId === selectedClipId);
		if (matchIndex >= 0 && matchIndex === conversationEvents.length - 1) return WORKSPACE_CURRENT_SELECTION_KEY;
		if (matchIndex >= 0) return getWorkspaceEventSelectionKey(conversationEvents[matchIndex], matchIndex);
		return WORKSPACE_CURRENT_SELECTION_KEY;
	}, [selectedEntryKeyProp, selectedClipId, originalClipId, conversationEvents]);

	const scrollToBottom = useCallback(() => {
		setTimeout(() => {
			bottomSentinelRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
		}, 60);
	}, []);

	useEffect(() => {
		if (conversationEvents.length > 0) scrollToBottom();
	}, [conversationEvents, scrollToBottom]);

	useEffect(() => {
		const el = topSentinelRef.current;
		if (!el) return;
		const observer = new IntersectionObserver(([entry]) => setShowTopFade(!entry.isIntersecting), { threshold: 0.1 });
		observer.observe(el);
		return () => observer.disconnect();
	}, [conversationEvents.length]);

	return (
		<div className="mt-auto flex flex-col">
			<ChromaGradient sessionStarted={sessionStarted} />
			{conversationEvents.length >= 1 && (
				<section className="relative z-10 flex flex-col h-full items-bottom">
					<div
						className={cn(
							"pointer-events-none sticky top-0 z-20 -mb-12 h-12 bg-linear-to-b from-background to-transparent transition-opacity duration-200",
							showTopFade ? "opacity-100" : "opacity-0",
						)}
						aria-hidden="true"
					/>
					<div ref={topSentinelRef} className="h-0 w-0" aria-hidden="true" />
					<div className="flex flex-col gap-4 pt-4 pb-4">
						<div className="vertical gap-2">
							<div
								className={cn(
									"flex items-start gap-3 rounded-xl p-3 transition-colors duration-200",
									originalClipId && onSelectOriginal ? "cursor-pointer hover:dark:bg-slate-800/30 hover:bg-slate-200/50" : "hover:dark:bg-slate-800/30 hover:bg-slate-200/50",
									selectedEntryKey === WORKSPACE_ORIGINAL_SELECTION_KEY && "ring-2 ring-blue-500/40 bg-slate-200/70 dark:bg-slate-800/50",
								)}
								data-selected={selectedEntryKey === WORKSPACE_ORIGINAL_SELECTION_KEY ? "true" : "false"}
								onClick={() => { if (originalClipId && onSelectOriginal) onSelectOriginal(); }}
							>
								<div className="flex min-w-0 flex-1 flex-col gap-2">
									<Badge variant="secondary" className="horizontal gap-2 items-center w-fit">
										<Lightbulb className="size-3 opacity-70" />
										Original
									</Badge>
									{originalLabel && <p className="line-clamp-2 text-sm leading-5 text-muted-foreground">{originalLabel}</p>}
								</div>
								{conversationEvents[0].thumbnail && (
									<img src={conversationEvents[0].thumbnail} alt="" className="mt-0.5 h-12 w-auto shrink-0 rounded-md border border-border object-cover" />
								)}
							</div>

							{conversationEvents.map((event, index) => {
								const isLast = index === conversationEvents.length - 1;
								const eventSelectionKey = isLast
									? WORKSPACE_CURRENT_SELECTION_KEY
									: getWorkspaceEventSelectionKey(event, index);
								const isClickable = isLast
									? Boolean(onSelectCurrent)
									: Boolean(event.clipId && onSelectEvent);
								const isSelected = selectedEntryKey === eventSelectionKey;
								return (
									<div
										key={event.promptId || index}
											className={cn(
												"flex items-start gap-3 rounded-xl p-3 transition-colors duration-200",
												isClickable && "cursor-pointer",
												isSelected && "ring-2 ring-blue-500/40 bg-slate-200/70 dark:bg-slate-800/50",
												!isSelected && "hover:bg-slate-200/50 hover:dark:bg-slate-800/30",
											)}
										data-selected={isSelected ? "true" : "false"}
										onClick={() => {
											if (isLast && onSelectCurrent) onSelectCurrent();
											else if (!isLast && event.clipId && onSelectEvent) onSelectEvent(event);
										}}
									>
										<div className="flex min-w-0 flex-1 flex-col gap-2">
											<Badge variant="secondary" className="horizontal gap-2 items-center w-fit">
												{isLast ? <Check className="size-3 opacity-70" /> : <Pencil className="size-3 opacity-70" />}
												{isLast ? "Current" : "Edit"}
											</Badge>
											<p className="line-clamp-2 text-sm leading-5 text-muted-foreground">{event.text}</p>
										</div>
									{isLast ? (
										currentThumbnail ? (
											<img src={currentThumbnail} alt="" className="mt-0.5 h-12 w-auto shrink-0 rounded-md border border-border object-cover" />
										) : (
											<div className="mt-0.5 h-12 w-[5.3rem] shrink-0 animate-pulse rounded-md border border-border bg-muted" />
										)
									) : (
										(event.resultThumbnail || event.thumbnail) && (
											<img src={event.resultThumbnail || event.thumbnail} alt="" className="mt-0.5 h-12 w-auto shrink-0 rounded-md border border-border object-cover" />
										)
									)}
									</div>
								);
							})}
						</div>
					</div>
					<div ref={bottomSentinelRef} className="h-0 w-0" aria-hidden="true" />
				</section>
			)}
		</div>
	);
}
