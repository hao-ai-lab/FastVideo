<script lang="ts">
	let {
		id,
		checked = false,
		onChange,
		disabled = false,
		label,
		ariaLabel,
	}: {
		id: string;
		checked?: boolean;
		onChange: (v: boolean) => void;
		disabled?: boolean;
		label?: string;
		ariaLabel?: string;
	} = $props();
</script>

<label
	for={id}
	class="toggle"
	class:toggleDisabled={disabled}
	data-checked={checked}
>
	<input
		type="checkbox"
		{id}
		checked={checked}
		onchange={(e) => onChange((e.target as HTMLInputElement).checked)}
		{disabled}
		aria-label={ariaLabel ?? label}
	/>
	<span class="toggleTrack" aria-hidden="true"></span>
	{#if label != null}
		<span class="toggleLabel">{label}</span>
	{/if}
</label>

<style>
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		cursor: pointer;
		user-select: none;
	}
	.toggleDisabled {
		cursor: not-allowed;
		opacity: 0.6;
	}
	.toggle input {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
	.toggleTrack {
		display: block;
		width: 2.25rem;
		height: 1.25rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 999px;
		transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
		flex-shrink: 0;
	}
	.toggleTrack::after {
		content: "";
		display: block;
		width: 0.9rem;
		height: 0.9rem;
		margin: 0.1rem 0 0 0.1rem;
		background: var(--text-dim);
		border-radius: 50%;
		transition: transform 0.15s, background 0.15s;
	}
	.toggle:hover:not(.toggleDisabled) .toggleTrack {
		border-color: var(--accent-h);
	}
	.toggle[data-checked="true"] .toggleTrack {
		background: var(--accent);
		border-color: var(--accent);
	}
	.toggle[data-checked="true"] .toggleTrack::after {
		transform: translateX(1rem);
		background: #fff;
	}
	.toggle:focus-within .toggleTrack {
		box-shadow: 0 0 0 2px var(--accent);
	}
	.toggleLabel {
		font-size: 0.8rem;
		color: var(--text-dim);
	}
</style>
