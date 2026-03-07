<script lang="ts">
	import JobQueue from "$lib/components/JobQueue.svelte";
	import CreateJobButton from "$lib/components/CreateJobButton.svelte";
	import { setHeaderActions } from "$lib/stores/headerActions";
	import { onMount, onDestroy } from "svelte";

	onMount(() => {
		setHeaderActions([{ component: CreateJobButton, props: { jobType: "lora" } }]);
	});
	onDestroy(() => setHeaderActions([]));
</script>

<JobQueue jobType="lora" />
