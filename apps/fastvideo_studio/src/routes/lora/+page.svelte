<script lang="ts">
	import { goto } from "$app/navigation";

	goto("/finetuning", { replaceState: true });
</script>

<p class="redirect-msg">Redirecting to Finetuning…</p>

<style>
	.redirect-msg {
		text-align: center;
		color: var(--text-dim);
		padding: 2rem;
	}
</style>
