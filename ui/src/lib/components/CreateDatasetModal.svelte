<script lang="ts">
	import { createDataset, uploadRawDataset } from "$lib/api";
	import UploadZone from "./UploadZone.svelte";
	import TabSwitch from "./TabSwitch.svelte";
	import {
		parseVideos2Caption,
		parseVideosCaptionsTxt,
		parseCaptionCsv,
	} from "$lib/captionParsing";

	const ALLOWED_VIDEO_EXT = ".mp4,.webm,.avi,.mov,.mkv";

	let {
		isOpen,
		onClose,
		onSuccess,
	}: {
		isOpen: boolean;
		onClose: () => void;
		onSuccess: () => void;
	} = $props();

	let name = $state("");
	let isSubmitting = $state(false);
	let rawPath = $state("");
	let fileNames = $state<string[]>([]);
	let isUploading = $state(false);
	let validationError = $state<string | null>(null);
	let captionFormat = $state<"json" | "txt" | "csv">("json");
	let captionMap = $state<Record<string, string> | null>(null);
	let captionFileName = $state<string | null>(null);
	let videosTxtLines = $state<string[] | null>(null);
	let videosTxtFileName = $state<string | null>(null);
	let captionsTxtLines = $state<string[] | null>(null);
	let captionsTxtFileName = $state<string | null>(null);

	$effect(() => {
		if (captionFormat === "txt" && captionsTxtLines && fileNames.length > 0) {
			const hasVideosTxt =
				videosTxtLines &&
				videosTxtLines.length > 0 &&
				videosTxtLines.some((s) => s.trim());
			if (hasVideosTxt && videosTxtLines) {
				const { error } = parseVideosCaptionsTxt(
					videosTxtLines,
					captionsTxtLines,
					fileNames,
				);
				validationError = error;
			} else {
				validationError = null;
			}
		}
	});

	const txtCaptionMap = $derived.by(() => {
		if (!captionsTxtLines || fileNames.length === 0) return null;
		const { captions, error } = parseVideosCaptionsTxt(
			videosTxtLines ?? null,
			captionsTxtLines,
			fileNames,
		);
		if (error) return null;
		return Object.keys(captions).length > 0 ? captions : null;
	});

	const effectiveCaptionMap = $derived(
		captionFormat === "txt" ? txtCaptionMap : captionMap,
	);

	$effect(() => {
		if (!isOpen) return;
		function onKey(e: KeyboardEvent) {
			if (e.key === "Escape" && !isSubmitting) handleClose();
		}
		document.addEventListener("keydown", onKey);
		return () => document.removeEventListener("keydown", onKey);
	});

	function handleClose() {
		if (isSubmitting) return;
		name = "";
		rawPath = "";
		fileNames = [];
		validationError = null;
		captionFormat = "json";
		captionMap = null;
		captionFileName = null;
		videosTxtLines = null;
		videosTxtFileName = null;
		captionsTxtLines = null;
		captionsTxtFileName = null;
		onClose();
	}

	async function handleCaptionJsonChange(files: File[]) {
		validationError = null;
		captionMap = null;
		captionFileName = null;
		if (files.length === 0) return;
		const file = files[0];
		try {
			const text = await file.text();
			const uploaded = fileNames.length > 0 ? fileNames : [];
			const { captions, error } = parseVideos2Caption(text, uploaded);
			if (error) {
				validationError = error;
				return;
			}
			captionMap = captions;
			captionFileName = file.name;
		} catch {
			validationError = "Could not read the file.";
		}
	}

	async function handleVideosTxtChange(files: File[]) {
		validationError = null;
		videosTxtLines = null;
		videosTxtFileName = null;
		if (files.length === 0) return;
		try {
			const text = await files[0].text();
			videosTxtLines = text.split(/\r?\n/).map((s) => s.trim());
			videosTxtFileName = files[0].name;
			if (captionsTxtLines && fileNames.length > 0) {
				const { error } = parseVideosCaptionsTxt(
					videosTxtLines,
					captionsTxtLines,
					fileNames,
				);
				if (error) validationError = error;
			}
		} catch {
			validationError = "Could not read videos.txt.";
		}
	}

	async function handleCaptionsTxtChange(files: File[]) {
		validationError = null;
		captionsTxtLines = null;
		captionsTxtFileName = null;
		if (files.length === 0) return;
		try {
			const text = await files[0].text();
			captionsTxtLines = text.split(/\r?\n/).map((s) => s.trim());
			captionsTxtFileName = files[0].name;
			if (videosTxtLines && fileNames.length > 0) {
				const { error } = parseVideosCaptionsTxt(
					videosTxtLines,
					captionsTxtLines,
					fileNames,
				);
				if (error) validationError = error;
			}
		} catch {
			validationError = "Could not read captions.txt.";
		}
	}

	async function handleCaptionCsvChange(files: File[]) {
		validationError = null;
		captionMap = null;
		captionFileName = null;
		if (files.length === 0) return;
		const file = files[0];
		try {
			const text = await file.text();
			const uploaded = fileNames.length > 0 ? fileNames : [];
			const { captions, error } = parseCaptionCsv(text, uploaded);
			if (error) {
				validationError = error;
				return;
			}
			captionMap = captions;
			captionFileName = file.name;
		} catch {
			validationError = "Could not read the file.";
		}
	}

	function handleCaptionFormatChange(format: string) {
		captionFormat = format as "json" | "txt" | "csv";
		validationError = null;
		captionMap = null;
		captionFileName = null;
		videosTxtLines = null;
		videosTxtFileName = null;
		captionsTxtLines = null;
		captionsTxtFileName = null;
	}

	async function handleMediaChange(files: File[]) {
		validationError = null;
		if (files.length === 0) {
			rawPath = "";
			fileNames = [];
			return;
		}
		isUploading = true;
		try {
			const res = await uploadRawDataset(files);
			rawPath = res.path;
			fileNames = res.file_names;
			if (res.file_names.length === 0) {
				validationError = `No video files found. Allowed: ${ALLOWED_VIDEO_EXT}`;
			}
		} catch (err) {
			rawPath = "";
			fileNames = [];
			validationError =
				err instanceof Error ? err.message : "Upload failed";
		} finally {
			isUploading = false;
		}
	}

	async function handleSubmit(e: SubmitEvent) {
		e.preventDefault();
		validationError = null;
		if (!name.trim()) return;
		if (!rawPath || fileNames.length === 0) {
			validationError = "No data was found. Upload at least one video.";
			return;
		}
		if (captionFormat === "json" || captionFormat === "csv") {
			if (captionFileName && !captionMap) {
				validationError =
					"Caption file has errors. Fix or remove it before creating the dataset.";
				return;
			}
		} else if (captionFormat === "txt") {
			if (videosTxtFileName || captionsTxtFileName) {
				if (!captionsTxtFileName) {
					validationError = "Upload captions.txt to use TXT captions.";
					return;
				}
				if (validationError) return;
			}
		}

		const finalCaptionMap =
			effectiveCaptionMap && Object.keys(effectiveCaptionMap).length > 0
				? effectiveCaptionMap
				: null;
		if (finalCaptionMap) {
			const missing = fileNames.filter((fn) => !(fn in finalCaptionMap));
			if (missing.length > 0) {
				const list =
					missing.length <= 5
						? missing.join(", ")
						: `${missing.slice(0, 5).join(", ")} and ${missing.length - 5} more`;
				const ok = confirm(
					`The caption file does not include captions for ${missing.length} video(s): ${list}. They will get empty captions. Continue?`,
				);
				if (!ok) return;
			}
		}

		isSubmitting = true;
		try {
			await createDataset({
				name: name.trim(),
				upload_path: rawPath,
				file_names: fileNames,
				...(finalCaptionMap ? { captions: finalCaptionMap } : {}),
			});
			onSuccess();
			handleClose();
		} catch (err) {
			validationError =
				err instanceof Error ? err.message : "Failed to create dataset";
		} finally {
			isSubmitting = false;
		}
	}
</script>

{#if isOpen}
	<div class="modal" data-modal>
		<div class="modalBackdrop" onclick={handleClose} role="presentation"></div>
		<div class="modalContent modalForm">
			<button
				class="modalClose"
				onclick={handleClose}
				disabled={isSubmitting}
				aria-label="Close"
			>
				×
			</button>
			<div class="card">
				<h2>Add Dataset — Raw</h2>
				<form onsubmit={handleSubmit} autocomplete="off">
					<div class="formRow">
						<label for="add-dataset-name">Name</label>
						<input
							id="add-dataset-name"
							type="text"
							bind:value={name}
							placeholder="My dataset"
							required
							disabled={isSubmitting}
						/>
					</div>
					<div class="formRow">
						<label>Videos</label>
						<UploadZone
							label="Upload video files"
							hint="Select files or a folder (.mp4, .webm, .avi, .mov, .mkv)"
							accept={ALLOWED_VIDEO_EXT}
							multiple={true}
							directory={true}
							allowBothFileAndDirectory={true}
							value={rawPath}
							fileName={fileNames.length > 0 ? `${fileNames.length} file(s)` : undefined}
							onFileChange={handleMediaChange}
							onClear={() => {
								rawPath = "";
								fileNames = [];
								captionMap = null;
								captionFileName = null;
								validationError = null;
							}}
							disabled={isSubmitting}
							uploading={isUploading}
						/>
					</div>
					<div class="formRow">
						<div class="captionRow">
							<label>Captions (optional)</label>
							<TabSwitch
								options={[
									{ id: "json", label: "JSON" },
									{ id: "txt", label: "TXT" },
									{ id: "csv", label: "CSV" },
								]}
								value={captionFormat}
								onChange={handleCaptionFormatChange}
								disabled={isSubmitting}
							/>
						</div>
						{#if captionFormat === "json"}
							<UploadZone
								label="Upload videos2caption.json"
								hint={'Array of { path, cap } or object mapping file names to captions'}
								accept=".json,application/json"
								value={captionFileName ? "1" : ""}
								fileName={captionFileName ?? undefined}
								onFileChange={handleCaptionJsonChange}
								onClear={() => {
									captionMap = null;
									captionFileName = null;
									validationError = null;
								}}
								disabled={isSubmitting}
							/>
						{:else if captionFormat === "txt"}
							<div class="twoCol">
								<UploadZone
									label="Upload videos.txt (optional)"
									hint="One video path per line, or leave empty to match captions to videos in alphabetical order"
									accept=".txt,text/plain"
									value={videosTxtFileName ? "1" : ""}
									fileName={videosTxtFileName ?? undefined}
									onFileChange={handleVideosTxtChange}
									onClear={() => {
										videosTxtLines = null;
										videosTxtFileName = null;
										validationError = null;
									}}
									disabled={isSubmitting}
								/>
								<UploadZone
									label="Upload captions.txt"
									hint="One caption per line (same order as videos.txt or alphabetical)"
									accept=".txt,text/plain"
									value={captionsTxtFileName ? "1" : ""}
									fileName={captionsTxtFileName ?? undefined}
									onFileChange={handleCaptionsTxtChange}
									onClear={() => {
										captionsTxtLines = null;
										captionsTxtFileName = null;
										validationError = null;
									}}
									disabled={isSubmitting}
								/>
							</div>
						{:else}
							<UploadZone
								label="Upload captions CSV"
								hint="Header: video_name, caption"
								accept=".csv,text/csv"
								value={captionFileName ? "1" : ""}
								fileName={captionFileName ?? undefined}
								onFileChange={handleCaptionCsvChange}
								onClear={() => {
									captionMap = null;
									captionFileName = null;
									validationError = null;
								}}
								disabled={isSubmitting}
							/>
						{/if}
					</div>
					{#if validationError}
						<p class="validationError">{validationError}</p>
					{/if}
					<button type="submit" class="btn btnPrimary" disabled={isSubmitting}>
						{isSubmitting ? "Creating…" : "Create Dataset"}
					</button>
				</form>
			</div>
		</div>
	</div>
{/if}

<style>
	.modal {
		position: fixed;
		inset: 0;
		z-index: 1000;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.modalBackdrop {
		position: absolute;
		inset: 0;
		background: rgba(0, 0, 0, 0.7);
	}
	.modalContent {
		position: relative;
		z-index: 1;
		max-width: 90vw;
		max-height: 90vh;
		border-radius: var(--radius);
		overflow: hidden;
		background: var(--surface);
		border: 1px solid var(--border);
		padding: 0;
		max-width: 850px;
		width: 90vw;
		max-height: 90vh;
		overflow-y: auto;
	}
	.modalClose {
		position: absolute;
		top: 0.5rem;
		right: 0.75rem;
		background: none;
		border: none;
		color: var(--text);
		font-size: 1.6rem;
		cursor: pointer;
		z-index: 2;
	}
	.card {
		padding: 1.5rem;
	}
	.card h2 {
		font-size: 1.15rem;
		margin-bottom: 1rem;
	}
	.formRow {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		margin-bottom: 0.85rem;
	}
	.formRow label {
		font-size: 0.8rem;
		color: var(--text-dim);
	}
	.captionRow {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
		margin-bottom: 0.35rem;
	}
	.captionRow label {
		margin-bottom: 0;
	}
	.twoCol {
		display: flex;
		gap: 15px;
		flex-wrap: wrap;
	}
	.validationError {
		color: var(--red);
		font-size: 0.85rem;
		margin-bottom: 0.5rem;
	}
	.btn {
		cursor: pointer;
		border: none;
		border-radius: var(--radius);
		font-family: inherit;
		font-weight: 600;
		padding: 0.55rem 1.2rem;
	}
	.btnPrimary {
		background: var(--accent);
		color: #fff;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>
