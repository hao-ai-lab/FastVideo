<script lang="ts">
	let {
		label,
		hint,
		accept,
		multiple = false,
		directory = false,
		allowBothFileAndDirectory = false,
		value = "",
		fileName,
		onFileChange,
		onClear,
		disabled = false,
		uploading = false,
		textInput = false,
		textValue = "",
		onTextChange,
		textPlaceholder,
	}: {
		label: string;
		hint?: string;
		accept?: string;
		multiple?: boolean;
		directory?: boolean;
		allowBothFileAndDirectory?: boolean;
		value?: string;
		fileName?: string;
		onFileChange?: (files: File[]) => void;
		onClear?: () => void;
		disabled?: boolean;
		uploading?: boolean;
		textInput?: boolean;
		textValue?: string;
		onTextChange?: (value: string) => void;
		textPlaceholder?: string;
	} = $props();

	let fileInputEl: HTMLInputElement;
	let directoryInputEl: HTMLInputElement;

	const useBoth = directory && allowBothFileAndDirectory;
	const hasContent = textInput ? !!textValue.trim() : !!(value || fileName);

	function handleChange(e: Event) {
		const files = (e.target as HTMLInputElement).files;
		if (files && files.length > 0) {
			onFileChange?.(Array.from(files));
		}
		(e.target as HTMLInputElement).value = "";
	}

	function handleClick() {
		if (!textInput && !disabled) {
			fileInputEl?.click();
		}
	}

	function clearInputs() {
		if (fileInputEl) fileInputEl.value = "";
		if (directoryInputEl) directoryInputEl.value = "";
	}
</script>

<div
	class="uploadZone"
	class:hasFile={hasContent}
	class:allowBoth={useBoth}
	onclick={!textInput && !useBoth ? handleClick : undefined}
	role={!textInput && !useBoth ? "button" : undefined}
	tabindex={!textInput && !useBoth ? 0 : undefined}
	onkeydown={(e) => {
		if (!textInput && !useBoth && (e.key === "Enter" || e.key === " ")) {
			e.preventDefault();
			handleClick();
		}
	}}
>
	<input
		bind:this={fileInputEl}
		type="file"
		{accept}
		{multiple}
		webkitdirectory={directory && !allowBothFileAndDirectory ? "" : undefined}
		onchange={handleChange}
		{disabled}
	/>
	{#if useBoth}
		<input
			bind:this={directoryInputEl}
			type="file"
			multiple
			webkitdirectory=""
			onchange={handleChange}
			{disabled}
		/>
	{/if}
	<div class="label">{label}</div>
	{#if textInput}
		<input
			type="text"
			value={textValue}
			oninput={(e) => onTextChange?.((e.target as HTMLInputElement).value)}
			placeholder={textPlaceholder}
			{disabled}
			onclick={(e) => e.stopPropagation()}
		/>
	{:else}
		{#if !hasContent}
			<span class="hint">
				{#if uploading}
					Uploading…
				{:else if useBoth}
					<span
						role="button"
						tabindex="0"
						class="selectFilesTrigger"
						onclick={(e) => {
							e.stopPropagation();
							handleClick();
						}}
						onkeydown={(e) => {
							if (e.key === "Enter" || e.key === " ") {
								e.preventDefault();
								handleClick();
							}
						}}
					>
						Select files
					</span>
					·
					<button
						type="button"
						class="clearLink"
						onclick={(e) => {
							e.preventDefault();
							e.stopPropagation();
							if (!disabled) directoryInputEl?.click();
						}}
						{disabled}
					>
						Select folder
					</button>
				{:else if directory}
					Click or drop folder
				{:else}
					Click or drop file(s)
				{/if}
			</span>
		{/if}
		{#if fileName}
			<div class="fileName">
				{fileName}
				{#if onClear}
					·
					<button
						type="button"
						class="clearLink"
						onclick={(e) => {
							e.stopPropagation();
							onClear();
							clearInputs();
						}}
						disabled={disabled || uploading}
					>
						Clear
					</button>
				{/if}
			</div>
		{/if}
	{/if}
	{#if hint}
		<div class="hint">{hint}</div>
	{/if}
</div>

<style>
	.uploadZone {
		border: 2px dashed var(--border);
		border-radius: var(--radius);
		padding: 1.5rem 1.25rem;
		min-height: 150px;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		text-align: center;
		background: rgba(255, 255, 255, 0.02);
		cursor: pointer;
		transition: border-color 0.15s, background 0.15s;
		flex-grow: 1;
	}
	.uploadZone.allowBoth {
		cursor: default;
	}
	.uploadZone:hover {
		border-color: var(--accent-h);
		background: rgba(255, 255, 255, 0.04);
	}
	.uploadZone.hasFile {
		border-style: solid;
		border-color: var(--accent);
	}
	.uploadZone input[type="file"] {
		display: none;
	}
	.uploadZone input[type="text"] {
		width: 100%;
		padding: 0.5rem;
		border: 1px solid var(--border);
		border-radius: var(--radius);
		background: var(--bg);
		color: var(--text);
		font-size: 0.9rem;
	}
	.label {
		font-size: 0.85rem;
		color: var(--text-dim);
		margin-bottom: 0.5rem;
	}
	.hint {
		font-size: 0.75rem;
		color: var(--text-dim);
		margin-top: 0.35rem;
	}
	.selectFilesTrigger {
		cursor: pointer;
		color: var(--accent-h);
	}
	.selectFilesTrigger:hover {
		color: var(--accent);
		text-decoration: underline;
	}
	.fileName {
		font-size: 0.85rem;
		color: var(--text);
		margin-top: 0.5rem;
	}
	.clearLink {
		background: none;
		border: none;
		padding: 0;
		font: inherit;
		color: var(--accent-h);
		cursor: pointer;
		text-decoration: none;
	}
	.clearLink:hover:not(:disabled) {
		color: var(--accent);
	}
</style>
