<script lang="ts">
	let {
		id,
		min,
		max,
		step,
		value,
		onChange,
		disabled = false,
		showValue = true,
		formatValue = (v: number) => String(v),
		ariaLabel,
	}: {
		id: string;
		min: number;
		max: number;
		step: number;
		value: number;
		onChange: (v: number) => void;
		disabled?: boolean;
		showValue?: boolean;
		formatValue?: (v: number) => string;
		ariaLabel?: string;
	} = $props();
</script>

<div class="sliderWrapper">
	<input
		type="range"
		{id}
		{min}
		{max}
		{step}
		value={value}
		oninput={(e) => onChange(parseFloat((e.target as HTMLInputElement).value))}
		{disabled}
		aria-label={ariaLabel}
	/>
	{#if showValue}
		<span class="sliderValue" aria-hidden="true">{formatValue(value)}</span>
	{/if}
</div>

<style>
	.sliderWrapper {
		display: flex;
		align-items: center;
		min-width: 0;
	}
	.sliderValue {
		font-size: 0.85rem;
		color: var(--text-dim);
		min-width: 2.5rem;
		text-align: right;
		flex-shrink: 0;
	}
	.sliderWrapper input[type="range"] {
		appearance: none;
		-webkit-appearance: none;
		flex: 1;
		min-width: 80px;
		height: 6px;
		background: var(--border);
		border-radius: 999px;
		outline: none;
	}
	.sliderWrapper input[type="range"]::-webkit-slider-thumb {
		appearance: none;
		-webkit-appearance: none;
		width: 16px;
		height: 16px;
		background: var(--accent);
		border-radius: 50%;
		cursor: pointer;
		transition: background 0.15s, transform 0.15s;
	}
	.sliderWrapper input[type="range"]::-webkit-slider-thumb:hover {
		background: var(--accent-h);
	}
	.sliderWrapper input[type="range"]:disabled::-webkit-slider-thumb {
		cursor: not-allowed;
		opacity: 0.6;
	}
	.sliderWrapper input[type="range"]::-moz-range-thumb {
		width: 16px;
		height: 16px;
		background: var(--accent);
		border: none;
		border-radius: 50%;
		cursor: pointer;
	}
	.sliderWrapper input[type="range"]::-moz-range-track {
		height: 6px;
		background: var(--border);
		border-radius: 999px;
	}
</style>
