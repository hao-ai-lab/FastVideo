// SPDX-License-Identifier: Apache-2.0
// Caption file parsing for CreateDatasetModal

export function parseVideos2Caption(
	text: string,
	uploadedFileNames: string[],
): { captions: Record<string, string>; error: string | null } {
	let data: unknown;
	try {
		data = JSON.parse(text);
	} catch {
		return { captions: {}, error: "Invalid JSON in videos2caption.json." };
	}
	const captions: Record<string, string> = {};
	const uploadedSet = new Set(uploadedFileNames);

	if (Array.isArray(data)) {
		for (let i = 0; i < data.length; i++) {
			const entry = data[i];
			if (entry === null || typeof entry !== "object") {
				return {
					captions: {},
					error: `Invalid entry at index ${i}: must be an object with "path" and "cap".`,
				};
			}
			const path = (entry as Record<string, unknown>)["path"];
			const cap = (entry as Record<string, unknown>)["cap"];
			if (typeof path !== "string") {
				return {
					captions: {},
					error: `Invalid entry at index ${i}: "path" must be a string.`,
				};
			}
			let caption: string;
			if (typeof cap === "string") {
				caption = cap;
			} else if (
				Array.isArray(cap) &&
				cap.length > 0 &&
				typeof cap[0] === "string"
			) {
				caption = cap[0];
			} else {
				return {
					captions: {},
					error: `Invalid entry at index ${i}: "cap" must be a string or non-empty array of strings.`,
				};
			}
			captions[path] = caption;
		}
	} else if (
		data !== null &&
		typeof data === "object" &&
		!Array.isArray(data)
	) {
		const obj = data as Record<string, unknown>;
		for (const [key, value] of Object.entries(obj)) {
			if (typeof key !== "string" || typeof value !== "string") {
				return {
					captions: {},
					error: `Invalid entry for "${String(key)}": keys and values must be strings.`,
				};
			}
			captions[key] = value;
		}
	} else {
		return {
			captions: {},
			error:
				"videos2caption.json must be an array of { path, cap } or an object mapping file names to captions.",
		};
	}

	if (uploadedFileNames.length > 0) {
		const unknownRefs = Object.keys(captions).filter(
			(k) => !uploadedSet.has(k),
		);
		if (unknownRefs.length > 0) {
			return {
				captions: {},
				error: `videos2caption.json references file(s) not in the uploaded videos: ${unknownRefs.slice(0, 5).join(", ")}${unknownRefs.length > 5 ? "…" : ""}.`,
			};
		}
	}
	return { captions, error: null };
}

export function parseVideosCaptionsTxt(
	videosLines: string[] | null,
	captionsLines: string[],
	uploadedFileNames: string[],
): { captions: Record<string, string>; error: string | null } {
	const captions: Record<string, string> = {};
	const useAlphabetical =
		!videosLines ||
		videosLines.length === 0 ||
		videosLines.every((s) => !s.trim());
	const paths: string[] = useAlphabetical
		? [...uploadedFileNames].sort()
		: videosLines!.map((s) => s.trim()).filter(Boolean);
	const len = Math.min(paths.length, captionsLines.length);
	for (let i = 0; i < len; i++) {
		const path = paths[i];
		if (path) captions[path] = captionsLines[i].trim();
	}
	if (!useAlphabetical && uploadedFileNames.length > 0) {
		const uploadedSet = new Set(uploadedFileNames);
		const unknownRefs = Object.keys(captions).filter(
			(k) => !uploadedSet.has(k),
		);
		if (unknownRefs.length > 0) {
			return {
				captions: {},
				error: `videos.txt references file(s) not in the uploaded videos: ${unknownRefs.slice(0, 5).join(", ")}${unknownRefs.length > 5 ? "…" : ""}.`,
			};
		}
	}
	return { captions, error: null };
}

function parseCsvLine(line: string): string[] {
	const out: string[] = [];
	let cur = "";
	let inQuotes = false;
	for (let i = 0; i < line.length; i++) {
		const c = line[i];
		if (c === '"') {
			if (inQuotes && line[i + 1] === '"') {
				cur += '"';
				i++;
			} else {
				inQuotes = !inQuotes;
			}
		} else if ((c === "," && !inQuotes) || c === "\n") {
			out.push(cur);
			cur = "";
		} else {
			cur += c;
		}
	}
	out.push(cur);
	return out;
}

export function parseCaptionCsv(
	text: string,
	uploadedFileNames: string[],
): { captions: Record<string, string>; error: string | null } {
	const lines = text.split(/\r?\n/).filter((s) => s.trim());
	if (lines.length < 2) {
		return {
			captions: {},
			error: "CSV must have a header row and at least one data row.",
		};
	}
	const headerParts = parseCsvLine(lines[0]).map((s) =>
		s.trim().toLowerCase(),
	);
	const vidIdx = headerParts.includes("video_name")
		? headerParts.indexOf("video_name")
		: 0;
	const capIdx = headerParts.includes("caption")
		? headerParts.indexOf("caption")
		: 1;
	const captions: Record<string, string> = {};
	for (let i = 1; i < lines.length; i++) {
		const row = parseCsvLine(lines[i]);
		const path = row[vidIdx]?.trim();
		const cap = row[capIdx]?.trim() ?? "";
		if (path) captions[path] = cap;
	}
	if (uploadedFileNames.length > 0) {
		const uploadedSet = new Set(uploadedFileNames);
		const unknownRefs = Object.keys(captions).filter(
			(k) => !uploadedSet.has(k),
		);
		if (unknownRefs.length > 0) {
			return {
				captions: {},
				error: `CSV references file(s) not in the uploaded videos: ${unknownRefs.slice(0, 5).join(", ")}${unknownRefs.length > 5 ? "…" : ""}.`,
			};
		}
	}
	return { captions, error: null };
}
