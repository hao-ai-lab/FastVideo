<script lang="ts">
	let { fileNames = [], captions = {} }: {
		fileNames?: string[];
		captions?: Record<string, string>;
	} = $props();

	function downloadBlob(blob: Blob, filename: string) {
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = filename;
		a.click();
		URL.revokeObjectURL(url);
	}

	const sortedNames = $derived([...fileNames].sort());
	const disabled = $derived(fileNames.length === 0);

	function handleDownloadJson() {
		const data = sortedNames.map((path) => ({
			path,
			cap: captions[path] ?? "",
		}));
		const blob = new Blob([JSON.stringify(data, null, 2)], {
			type: "application/json",
		});
		downloadBlob(blob, "videos2caption.json");
	}

	function handleDownloadTxt() {
		const videosContent = sortedNames.join("\n");
		const promptContent = sortedNames.map((fn) => captions[fn] ?? "").join("\n");
		downloadBlob(new Blob([videosContent], { type: "text/plain" }), "videos.txt");
		setTimeout(() => {
			downloadBlob(
				new Blob([promptContent], { type: "text/plain" }),
				"captions.txt",
			);
		}, 100);
	}

	function handleDownloadCsv() {
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
	}
</script>

<div class="wrapper">
	<button type="button" class="btn btnSmall trigger" disabled={disabled}>
		Download Captions
	</button>
	<div class="menu" role="menu">
		<button
			class="menuItem"
			role="menuitem"
			onclick={handleDownloadJson}
			disabled={disabled}
		>
			JSON
		</button>
		<button
			class="menuItem"
			role="menuitem"
			onclick={handleDownloadTxt}
			disabled={disabled}
		>
			TXT
		</button>
		<button
			class="menuItem"
			role="menuitem"
			onclick={handleDownloadCsv}
			disabled={disabled}
		>
			CSV
		</button>
	</div>
</div>

<style>
	.wrapper {
		position: relative;
		display: inline-block;
	}
	.trigger {
		display: flex;
		align-items: center;
		gap: 0.35rem;
	}
	.trigger::after {
		content: "";
		width: 0;
		height: 0;
		border-left: 4px solid transparent;
		border-right: 4px solid transparent;
		border-top: 5px solid currentColor;
		opacity: 0.85;
	}
	.menu {
		position: absolute;
		top: 100%;
		right: 0;
		margin-top: 0.25rem;
		min-width: 100%;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
		padding: 0.25rem 0;
		opacity: 0;
		visibility: hidden;
		transform: translateY(-4px);
		transition: opacity 0.15s, visibility 0.15s, transform 0.15s;
		z-index: 200;
	}
	.wrapper:hover .menu {
		opacity: 1;
		visibility: visible;
		transform: translateY(0);
	}
	.menuItem {
		display: block;
		width: 100%;
		padding: 0.5rem 1rem;
		border: none;
		background: transparent;
		color: var(--text);
		font-family: inherit;
		font-size: 0.9rem;
		font-weight: 500;
		text-align: left;
		cursor: pointer;
		transition: background 0.1s;
		white-space: nowrap;
	}
	.menuItem:hover {
		background: var(--border);
	}
	.menuItem:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btn {
		cursor: pointer;
		border: none;
		border-radius: var(--radius);
		font-family: inherit;
		font-weight: 600;
		font-size: 0.85rem;
		padding: 0.55rem 1.2rem;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btnSmall {
		background: transparent;
		color: var(--accent-h);
		border: 1px solid var(--border);
		padding: 0.3rem 0.7rem;
		font-size: 0.8rem;
	}
</style>
