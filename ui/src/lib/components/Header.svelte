<script lang="ts">
	import { page } from "$app/stores";
	import { headerActions } from "$lib/stores/headerActions";
	import type { HeaderAction } from "$lib/stores/headerActions";

	const TAB_TITLES: Record<string, string> = {
		"/inference": "Jobs",
		"/finetuning": "Jobs",
		"/distillation": "Jobs",
		"/datasets": "Datasets",
		"/settings": "Settings",
	};

	const title = $derived(TAB_TITLES[$page.url.pathname] ?? "FastVideo");

	let actions = $state<HeaderAction[]>([]);
	$effect(() => {
		const unsub = headerActions.subscribe((v) => (actions = v));
		return unsub;
	});
</script>

<header class="header">
	<img src="/logo.svg" alt="FastVideo Logo" width="100" height="42" class="logo" />
	<h1 class="title">{title}</h1>
	{#if actions.length > 0}
		<div class="actions">
			{#each actions as action, i (i)}
				<svelte:component this={action.component} {...(action.props ?? {})} />
			{/each}
		</div>
	{/if}
</header>

<style>
	.header {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		z-index: 100;
		display: flex;
		align-items: center;
		gap: 1.5rem;
		padding: 1rem 1.5rem;
		border-bottom: 1px solid var(--border);
		background: var(--bg);
	}
	.title {
		font-size: 1.25rem;
		font-weight: 600;
		letter-spacing: -0.02em;
		margin: 0;
		flex: 1;
	}
	.logo {
		width: 100px;
		height: 42px;
		display: block;
	}
	.header h1 {
		font-size: 1.75rem;
		font-weight: 700;
		letter-spacing: -0.02em;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}
</style>
