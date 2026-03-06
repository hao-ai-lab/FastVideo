"use client";

import { useCallback, useMemo } from "react";
import buttonStyles from "./styles/Button.module.css";
import dropdownStyles from "./styles/Dropdown.module.css";

function downloadBlob(blob: Blob, filename: string) {
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

export interface DownloadCaptionsProps {
	fileNames: string[];
	captions: Record<string, string>;
}

export default function DownloadCaptions({
	fileNames,
	captions,
}: DownloadCaptionsProps) {
	const sortedNames = useMemo(() => [...fileNames].sort(), [fileNames]);
	const disabled = fileNames.length === 0;

	const handleDownloadJson = useCallback(() => {
		const data = sortedNames.map((path) => ({
			path,
			cap: captions[path] ?? "",
		}));
		const blob = new Blob([JSON.stringify(data, null, 2)], {
			type: "application/json",
		});
		downloadBlob(blob, "videos2caption.json");
	}, [sortedNames, captions]);

	const handleDownloadTxt = useCallback(() => {
		const videosContent = sortedNames.join("\n");
		const promptContent = sortedNames
			.map((fn) => captions[fn] ?? "")
			.join("\n");
		downloadBlob(
			new Blob([videosContent], { type: "text/plain" }),
			"videos.txt",
		);
		setTimeout(() => {
			downloadBlob(
				new Blob([promptContent], { type: "text/plain" }),
				"captions.txt",
			);
		}, 100);
	}, [sortedNames, captions]);

	const handleDownloadCsv = useCallback(() => {
		const escape = (s: string) =>
			s.includes('"') || s.includes(",") || s.includes("\n")
				? `"${s.replace(/"/g, '""')}"`
				: s;
		const rows = sortedNames.map(
			(fn) => `${escape(fn)},${escape(captions[fn] ?? "")}`,
		);
		const header = "video_name,caption";
		const csv = [header, ...rows].join("\n");
		downloadBlob(new Blob([csv], { type: "text/csv" }), "captions.csv");
	}, [sortedNames, captions]);

	return (
		<div className={dropdownStyles.wrapper}>
			<button
				type="button"
				className={`${buttonStyles.btn} ${buttonStyles.btnSmall} ${dropdownStyles.trigger}`}
				disabled={disabled}
			>
				Download Captions
			</button>
			<div className={dropdownStyles.menu} role="menu">
				<button
					className={dropdownStyles.menuItem}
					role="menuitem"
					onClick={handleDownloadJson}
					disabled={disabled}
				>
					JSON
				</button>
				<button
					className={dropdownStyles.menuItem}
					role="menuitem"
					onClick={handleDownloadTxt}
					disabled={disabled}
				>
					TXT
				</button>
				<button
					className={dropdownStyles.menuItem}
					role="menuitem"
					onClick={handleDownloadCsv}
					disabled={disabled}
				>
					CSV
				</button>
			</div>
		</div>
	);
}
