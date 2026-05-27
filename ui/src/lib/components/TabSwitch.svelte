<script lang="ts">
	export interface TabSwitchOption {
		id: string;
		label: string;
	}
	let {
		options = [],
		value = "",
		onChange = () => {},
		disabled = false,
	}: {
		options?: TabSwitchOption[];
		value?: string;
		onChange?: (id: string) => void;
		disabled?: boolean;
	} = $props();
</script>

<div class="tabs" role="tablist">
	{#each options as opt}
		<button
			type="button"
			role="tab"
			aria-selected={value === opt.id}
			class="tab"
			class:tabActive={value === opt.id}
			onclick={() => onChange(opt.id)}
			{disabled}
		>
			{opt.label}
		</button>
	{/each}
</div>

<style>
	.tabs {
		display: inline-flex;
		align-items: center;
		gap: 0;
		padding: 2px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
	}
	.tab {
		padding: 0.4rem 0.85rem;
		font-size: 0.85rem;
		font-weight: 500;
		color: var(--text-dim);
		border: none;
		background: transparent;
		border-radius: calc(var(--radius) - 2px);
		cursor: pointer;
		font-family: inherit;
		transition: color 0.15s, background 0.15s;
	}
	.tab:hover:not(:disabled) {
		color: var(--text);
		background: rgba(99, 102, 241, 0.08);
	}
	.tabActive {
		color: var(--text);
		background: var(--surface);
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
	}
	.tab:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>
