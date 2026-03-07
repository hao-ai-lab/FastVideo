<script lang="ts">
	import { onMount, onDestroy } from "svelte";
	import { getModels } from "$lib/api";
	import type { Model } from "$lib/api";
	import type { WorkloadType } from "$lib/defaultOptions";
	import { defaultOptions, updateOption, resetToDefaults } from "$lib/stores/defaultOptions";
	import { setHeaderActions } from "$lib/stores/headerActions";
	import Toggle from "$lib/components/Toggle.svelte";
	import Slider from "$lib/components/Slider.svelte";

	onMount(() => setHeaderActions([]));
	onDestroy(() => setHeaderActions([]));

	let modelsT2v = $state<Model[]>([]);
	let modelsI2v = $state<Model[]>([]);
	let modelsT2i = $state<Model[]>([]);

	onMount(() => {
		Promise.all([
			getModels("t2v"),
			getModels("i2v"),
			getModels("t2i"),
		])
			.then(([t2v, i2v, t2i]) => {
				modelsT2v = t2v;
				modelsI2v = i2v;
				modelsT2i = t2i;
			})
			.catch((e) => console.error("Failed to load models:", e));
	});
</script>

<main class="main">
	<section class="card">
		<h2>Behavior</h2>
		<div class="row">
			<label for="settings-auto-start-job">Auto Start Job on Create</label>
			<Toggle
				id="settings-auto-start-job"
				checked={$defaultOptions.autoStartJob}
				onChange={(v) => updateOption("autoStartJob", v)}
			/>
		</div>
		<hr class="hr" />
		<h2>Paths</h2>
		<div class="formRow">
			<label for="settings-dataset-upload-path">Dataset Upload Path</label>
			<input
				id="settings-dataset-upload-path"
				type="text"
				value={$defaultOptions.datasetUploadPath ?? ""}
				oninput={(e) =>
					updateOption("datasetUploadPath", (e.target as HTMLInputElement).value)}
				placeholder="outputs/ui_data/uploads/datasets"
				style="font-family: monospace; font-size: 0.9rem"
			/>
		</div>
		<hr class="hr" />
		<div class="sectionHeader">
			<h2>Default Options</h2>
			<button type="button" class="btn btnSmall" onclick={resetToDefaults}>
				Reset to Defaults
			</button>
		</div>
		<p class="helperP">
			These values are used as defaults when creating new jobs.
		</p>
		<div class="settingsGrid">
			<div class="formRow">
				<label for="settings-default-model-t2v">Default Model (T2V)</label>
				<select
					id="settings-default-model-t2v"
					value={$defaultOptions.defaultModelIdT2v}
					onchange={(e) =>
						updateOption(
							"defaultModelIdT2v",
							(e.target as HTMLSelectElement).value,
						)}
				>
					<option value="">None (select when creating job)</option>
					{#each modelsT2v as model}
						<option value={model.id}>{model.label} ({model.id})</option>
					{/each}
				</select>
			</div>
			<div class="formRow">
				<label for="settings-default-model-i2v">Default Model (I2V)</label>
				<select
					id="settings-default-model-i2v"
					value={$defaultOptions.defaultModelIdI2v}
					onchange={(e) =>
						updateOption(
							"defaultModelIdI2v",
							(e.target as HTMLSelectElement).value,
						)}
				>
					<option value="">None</option>
					{#each modelsI2v as model}
						<option value={model.id}>{model.label} ({model.id})</option>
					{/each}
				</select>
			</div>
			<div class="formRow">
				<label for="settings-default-model-t2i">Default Model (T2I)</label>
				<select
					id="settings-default-model-t2i"
					value={$defaultOptions.defaultModelIdT2i}
					onchange={(e) =>
						updateOption(
							"defaultModelIdT2i",
							(e.target as HTMLSelectElement).value,
						)}
				>
					<option value="">None</option>
					{#each modelsT2i as model}
						<option value={model.id}>{model.label} ({model.id})</option>
					{/each}
				</select>
			</div>
			<div class="formRow">
				<label for="settings-num-frames">Frames</label>
				<Slider
					id="settings-num-frames"
					min={1}
					max={500}
					step={1}
					value={$defaultOptions.numFrames}
					onChange={(v) => updateOption("numFrames", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-height">Height</label>
				<Slider
					id="settings-height"
					min={64}
					max={1080}
					step={16}
					value={$defaultOptions.height}
					onChange={(v) => updateOption("height", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-width">Width</label>
				<Slider
					id="settings-width"
					min={64}
					max={1920}
					step={16}
					value={$defaultOptions.width}
					onChange={(v) => updateOption("width", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-num-steps">Inference Steps</label>
				<Slider
					id="settings-num-steps"
					min={1}
					max={200}
					step={1}
					value={$defaultOptions.numInferenceSteps}
					onChange={(v) => updateOption("numInferenceSteps", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-vsa-sparsity" title="VSA sparsity (0–1)">VSA Sparsity</label>
				<Slider
					id="settings-vsa-sparsity"
					min={0}
					max={1}
					step={0.05}
					value={$defaultOptions.vsaSparsity}
					onChange={(v) => updateOption("vsaSparsity", v)}
					formatValue={(v) => v.toFixed(2)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-guidance">Guidance Scale</label>
				<Slider
					id="settings-guidance"
					min={0}
					max={20}
					step={0.1}
					value={$defaultOptions.guidanceScale}
					onChange={(v) => updateOption("guidanceScale", v)}
					formatValue={(v) => v.toFixed(1)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-guidance-rescale" title="0 = disabled">Guidance Rescale</label>
				<Slider
					id="settings-guidance-rescale"
					min={0}
					max={1}
					step={0.05}
					value={$defaultOptions.guidanceRescale ?? 0}
					onChange={(v) => updateOption("guidanceRescale", v)}
					formatValue={(v) => v.toFixed(2)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-tp-size" title="-1 = auto">TP Size</label>
				<Slider
					id="settings-tp-size"
					min={-1}
					max={8}
					step={1}
					value={$defaultOptions.tpSize}
					onChange={(v) => updateOption("tpSize", v)}
					formatValue={(v) => (v === -1 ? "Auto" : String(v))}
				/>
			</div>
			<div class="formRow">
				<label for="settings-sp-size" title="-1 = auto">SP Size</label>
				<Slider
					id="settings-sp-size"
					min={-1}
					max={8}
					step={1}
					value={$defaultOptions.spSize}
					onChange={(v) => updateOption("spSize", v)}
					formatValue={(v) => (v === -1 ? "Auto" : String(v))}
				/>
			</div>
			<div class="formRow">
				<label for="settings-fps">FPS</label>
				<Slider
					id="settings-fps"
					min={1}
					max={60}
					step={1}
					value={$defaultOptions.fps ?? 24}
					onChange={(v) => updateOption("fps", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-dit-cpu-offload">DiT CPU Offload</label>
				<Toggle
					id="settings-dit-cpu-offload"
					checked={$defaultOptions.ditCpuOffload}
					onChange={(v) => updateOption("ditCpuOffload", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-text-encoder-cpu-offload">Text Encoder CPU Offload</label>
				<Toggle
					id="settings-text-encoder-cpu-offload"
					checked={$defaultOptions.textEncoderCpuOffload}
					onChange={(v) => updateOption("textEncoderCpuOffload", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-use-fsdp-inference">Use FSDP Inference</label>
				<Toggle
					id="settings-use-fsdp-inference"
					checked={$defaultOptions.useFsdpInference}
					onChange={(v) => updateOption("useFsdpInference", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-vae-cpu-offload">VAE CPU Offload</label>
				<Toggle
					id="settings-vae-cpu-offload"
					checked={$defaultOptions.vaeCpuOffload}
					onChange={(v) => updateOption("vaeCpuOffload", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-image-encoder-cpu-offload">Image Encoder CPU Offload</label>
				<Toggle
					id="settings-image-encoder-cpu-offload"
					checked={$defaultOptions.imageEncoderCpuOffload}
					onChange={(v) => updateOption("imageEncoderCpuOffload", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-enable-torch-compile">Torch Compile</label>
				<Toggle
					id="settings-enable-torch-compile"
					checked={$defaultOptions.enableTorchCompile}
					onChange={(v) => updateOption("enableTorchCompile", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-num-gpus">GPUs</label>
				<Slider
					id="settings-num-gpus"
					min={1}
					max={8}
					step={1}
					value={$defaultOptions.numGpus}
					onChange={(v) => updateOption("numGpus", v)}
				/>
			</div>
			<div class="formRow">
				<label for="settings-seed">Seed</label>
				<input
					id="settings-seed"
					type="number"
					value={$defaultOptions.seed}
					oninput={(e) =>
						updateOption(
							"seed",
							parseInt((e.target as HTMLInputElement).value, 10),
						)}
					min={0}
				/>
			</div>
		</div>
	</section>
</main>

<style>
	.main {
		max-width: 850px;
		margin: 0 auto;
		padding: 0 1rem 3rem;
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
		width: 100%;
	}
	.card {
		padding: 1.5rem;
	}
	.card h2 {
		font-size: 1.15rem;
		margin-bottom: 1rem;
	}
	.row {
		display: flex;
		flex-direction: row;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.85rem;
	}
	.row label {
		font-size: 0.8rem;
		color: var(--text-dim);
	}
	.hr {
		margin: 1rem 0;
		border: none;
		border-top: 1px solid var(--border);
	}
	.formRow {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		margin-bottom: 0.85rem;
	}
	.formRow label {
		font-size: 0.8rem;
		color: var(--text-dim);
	}
	.sectionHeader {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1rem;
	}
	.sectionHeader h2 {
		margin-bottom: 0;
	}
	.helperP {
		color: var(--text-dim);
		font-size: 0.9rem;
		margin-bottom: 1rem;
	}
	.settingsGrid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
		gap: 0.5rem 0.75rem;
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
	.btnSmall {
		background: transparent;
		color: var(--accent-h);
		border: 1px solid var(--border);
		padding: 0.3rem 0.7rem;
		font-size: 0.8rem;
	}
</style>
