<script lang="ts">
	import { onMount } from "svelte";
	import { goto } from "$app/navigation";

	onMount(() => {
		goto("/inference", { replaceState: true });
	});
</script>

<p class="redirect-msg">Redirecting to Inference…</p>

<style>
	.redirect-msg {
		padding: 2rem;
		color: var(--text-dim);
	}
</style>
