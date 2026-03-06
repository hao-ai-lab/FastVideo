"use client";

import {
	type Dataset,
	getDatasetFiles,
	updateDatasetCaption,
	getDatasetMediaUrl,
} from "@/lib/api";
import { useCallback, useEffect, useRef, useState } from "react";
import DownloadCaptions from "./DownloadCaptions";
import datasetStyles from "./styles/DatasetSidebar.module.css";

const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_MAX_WIDTH = 900;
const INITIAL_PAGE_SIZE = 24;
const PAGE_SIZE = 24;

function CloseIcon() {
	return (
		<svg
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
		>
			<path d="M18 6L6 18M6 6l12 12" />
		</svg>
	);
}

interface DatasetSidebarProps {
	dataset: Dataset;
	onClose: () => void;
	onWidthChange?: (width: number) => void;
	onUpdated?: () => void;
}

export default function DatasetSidebar({
	dataset,
	onClose,
	onWidthChange,
}: DatasetSidebarProps) {
	const [width, setWidth] = useState(400);
	const [isDragging, setIsDragging] = useState(false);
	const dragStartRef = useRef({ x: 0, width: 0 });
	const [fileNames, setFileNames] = useState<string[]>([]);
	const [captions, setCaptions] = useState<Record<string, string>>({});
	const [visibleCount, setVisibleCount] = useState(INITIAL_PAGE_SIZE);
	const [isLoading, setIsLoading] = useState(true);
	const [thumbLoaded, setThumbLoaded] = useState<Record<string, boolean>>({});
	const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const scrollRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		onWidthChange?.(width);
	}, [width, onWidthChange]);

	useEffect(() => {
		let cancelled = false;
		setIsLoading(true);
		getDatasetFiles(dataset.id)
			.then((data) => {
				if (!cancelled) {
					setFileNames(data.file_names);
					setCaptions(data.captions);
					setVisibleCount(INITIAL_PAGE_SIZE);
					setThumbLoaded({});
				}
			})
			.catch((err) => console.error("Failed to load dataset files:", err))
			.finally(() => {
				if (!cancelled) setIsLoading(false);
			});
		return () => {
			cancelled = true;
		};
	}, [dataset.id]);

	const handleCaptionChange = useCallback(
		(fileName: string, value: string) => {
			setCaptions((prev) => ({ ...prev, [fileName]: value }));
			if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
			saveTimeoutRef.current = setTimeout(() => {
				saveTimeoutRef.current = null;
				updateDatasetCaption(dataset.id, fileName, value).catch((err) =>
					console.error("Failed to save caption:", err),
				);
			}, 500);
		},
		[dataset.id],
	);

	useEffect(() => {
		return () => {
			if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
		};
	}, []);

	const handleLoadMore = useCallback(() => {
		setVisibleCount((c) => Math.min(c + PAGE_SIZE, fileNames.length));
	}, [fileNames.length]);

	const handleMouseDown = useCallback(
		(e: React.MouseEvent) => {
			e.preventDefault();
			dragStartRef.current = { x: e.clientX, width };
			setIsDragging(true);
		},
		[width],
	);

	useEffect(() => {
		if (!isDragging) return;
		const handleMouseMove = (e: MouseEvent) => {
			const { x, width: startWidth } = dragStartRef.current;
			const delta = e.clientX - x;
			const newWidth = Math.min(
				SIDEBAR_MAX_WIDTH,
				Math.max(SIDEBAR_MIN_WIDTH, startWidth - delta),
			);
			setWidth(newWidth);
		};
		const handleMouseUp = () => setIsDragging(false);
		document.body.style.cursor = "col-resize";
		document.body.style.userSelect = "none";
		document.addEventListener("mousemove", handleMouseMove);
		document.addEventListener("mouseup", handleMouseUp);
		return () => {
			document.body.style.cursor = "";
			document.body.style.userSelect = "";
			document.removeEventListener("mousemove", handleMouseMove);
			document.removeEventListener("mouseup", handleMouseUp);
		};
	}, [isDragging]);

	const visibleFiles = fileNames.slice(0, visibleCount);
	const hasMore = visibleCount < fileNames.length;

	useEffect(() => {
		const el = scrollRef.current;
		if (!el) return;

		const handleScroll = () => {
			if (!hasMore || isLoading) return;
			const { scrollTop, scrollHeight, clientHeight } = el;
			const distanceFromBottom =
				scrollHeight - (scrollTop + clientHeight);
			if (distanceFromBottom < 200) {
				setVisibleCount((c) =>
					Math.min(c + PAGE_SIZE, fileNames.length),
				);
			}
		};

		el.addEventListener("scroll", handleScroll);
		return () => {
			el.removeEventListener("scroll", handleScroll);
		};
	}, [fileNames.length, hasMore, isLoading]);

	return (
		<aside
			className={datasetStyles.sidebar}
			style={{ width, maxWidth: SIDEBAR_MAX_WIDTH }}
		>
			<div className={datasetStyles.header}>
				<h2 className={datasetStyles.title}>{dataset.name}</h2>
				<div className={datasetStyles.headerActions}>
					<DownloadCaptions
						fileNames={fileNames}
						captions={captions}
					/>
					<button
						type="button"
						className={datasetStyles.closeBtn}
						onClick={onClose}
						title="Close"
					>
						<CloseIcon />
					</button>
				</div>
			</div>
			<div className={datasetStyles.gallerySection}>
				<div className={datasetStyles.galleryScroll} ref={scrollRef}>
					{isLoading ? (
						<p className={datasetStyles.galleryEmpty}>Loading…</p>
					) : fileNames.length === 0 ? (
						<p className={datasetStyles.galleryEmpty}>
							No media files
						</p>
					) : (
						<>
							<div className={datasetStyles.galleryGrid}>
								{visibleFiles.map((fileName) => (
									<div
										key={fileName}
										className={datasetStyles.galleryItem}
									>
										{!thumbLoaded[fileName] && (
											<div
												className={
													datasetStyles.thumbLoading
												}
											>
												<div
													className={
														datasetStyles.thumbSpinner
													}
												/>
											</div>
										)}
										<video
											src={getDatasetMediaUrl(
												dataset.id,
												fileName,
											)}
											className={
												datasetStyles.galleryThumb
											}
											muted
											autoPlay
											loop
											playsInline
											onLoadedData={() =>
												setThumbLoaded((prev) => ({
													...prev,
													[fileName]: true,
												}))
											}
											onError={() =>
												setThumbLoaded((prev) => ({
													...prev,
													[fileName]: true,
												}))
											}
										/>
										<div
											className={
												datasetStyles.galleryCaption
											}
										>
											<textarea
												value={captions[fileName] ?? ""}
												onChange={(e) =>
													handleCaptionChange(
														fileName,
														e.target.value,
													)
												}
												placeholder="Caption"
												rows={2}
											/>
										</div>
									</div>
								))}
							</div>
						</>
					)}
				</div>
			</div>
			<div
				className={`${datasetStyles.resizeHandle} ${
					isDragging ? datasetStyles.resizeHandleActive : ""
				}`}
				onMouseDown={handleMouseDown}
			/>
		</aside>
	);
}
