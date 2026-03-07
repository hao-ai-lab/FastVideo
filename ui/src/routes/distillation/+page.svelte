<script lang="ts">
	import JobQueue from "$lib/components/JobQueue.svelte";
	import CreateJobButton from "$lib/components/CreateJobButton.svelte";
	import { setHeaderActions } from "$lib/stores/headerActions";
	import { onDestroy } from "svelte";

	onDestroy(() => setHeaderActions([]));
	setHeaderActions([
		{ component: CreateJobButton, props: { jobType: "distillation" } },
	]);
</script>

<JobQueue jobType="distillation" />
