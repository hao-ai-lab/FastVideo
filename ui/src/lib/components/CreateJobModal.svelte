<script lang="ts">
	import {
		createJob,
		getModels,
		getDatasets,
		uploadImage,
		type Model,
	} from "$lib/api";
	import { getDefaultModelForWorkload } from "$lib/defaultOptions";
	import { defaultOptions } from "$lib/stores/defaultOptions";
	import { WORKLOAD_OPTIONS } from "$lib/jobConfig";
	import type { JobType } from "$lib/types";
	import Toggle from "./Toggle.svelte";
	import Slider from "./Slider.svelte";

	let {
		isOpen,
		onClose,
		onSuccess,
		jobType,
		workloadType,
	}: {
		isOpen: boolean;
		onClose: () => void;
		onSuccess: () => void;
		jobType: JobType;
		workloadType: string;
	} = $props();

	function getModelWorkloadForTraining(_w: string): string {
		return "t2v";
	}

	const isInference = $derived(jobType === "inference");
	const inferenceWorkload = $derived(
		isInference ? workloadType : getModelWorkloadForTraining(workloadType),
	);

	let models = $state<Model[]>([]);
	let modelId = $state("");
	let prompt = $state("");
	let imagePath = $state("");
	let imageFileName = $state("");
	let isUploadingImage = $state(false);
	let negativePrompt = $state("");
	let numInferenceSteps = $state(50);
	let numFrames = $state(81);
	let height = $state(480);
	let width = $state(832);
	let guidanceScale = $state(5);
	let guidanceRescale = $state(0);
	let fps = $state(24);
	let seed = $state(1024);
	let numGpus = $state(1);
	let ditCpuOffload = $state(false);
	let textEncoderCpuOffload = $state(false);
	let vaeCpuOffload = $state(false);
	let imageEncoderCpuOffload = $state(false);
	let useFsdpInference = $state(false);
	let enableTorchCompile = $state(false);
	let vsaSparsity = $state(0);
	let tpSize = $state(-1);
	let spSize = $state(-1);
	let selectedDatasetId = $state("");
	let readyDatasets = $state<Awaited<ReturnType<typeof getDatasets>>>([]);
	let maxTrainSteps = $state(1000);
	let trainBatchSize = $state(1);
	let learningRate = $state(5e-5);
	let numLatentT = $state(20);
	let selectedValidationDatasetId = $state("");
	let loraRank = $state(32);
	let ltx2FirstFrameConditioningP = $state(0.1);
	let dmdUseVsa = $state(false);
	let dmdVsaSparsity = $state(0.8);
	let dmdDenoisingSteps = $state("1000,757,522");
	let minTimestepRatio = $state(0.02);
	let maxTimestepRatio = $state(0.98);
	let realScoreGuidanceScale = $state(3.5);
	let generatorUpdateInterval = $state(5);
	let realScoreModelPath = $state("");
	let fakeScoreModelPath = $state("");
	let isSubmitting = $state(false);
	let isLoadingModels = $state(false);
	let imageInputEl: HTMLInputElement;

	$effect(() => {
		if (!isOpen) return;
		const opts = $defaultOptions;
		numInferenceSteps = opts.numInferenceSteps;
		numFrames = workloadType === "t2i" ? 1 : opts.numFrames;
		height = opts.height;
		width = opts.width;
		guidanceScale = opts.guidanceScale;
		guidanceRescale = opts.guidanceRescale;
		fps = opts.fps;
		seed = opts.seed;
		numGpus = opts.numGpus;
		ditCpuOffload = opts.ditCpuOffload;
		textEncoderCpuOffload = opts.textEncoderCpuOffload;
		vaeCpuOffload = opts.vaeCpuOffload;
		imageEncoderCpuOffload = opts.imageEncoderCpuOffload;
		useFsdpInference = opts.useFsdpInference;
		enableTorchCompile = opts.enableTorchCompile;
		vsaSparsity = opts.vsaSparsity;
		tpSize = opts.tpSize;
		spSize = opts.spSize;
		modelId = getDefaultModelForWorkload(
			opts,
			inferenceWorkload as "t2v" | "i2v" | "t2i",
		);
		imagePath = "";
		imageFileName = "";
		selectedDatasetId = "";
		selectedValidationDatasetId = "";
		if (workloadType === "dmd_t2v") {
			dmdUseVsa = false;
			dmdVsaSparsity = 0.8;
			dmdDenoisingSteps = "1000,757,522";
			minTimestepRatio = 0.02;
			maxTimestepRatio = 0.98;
			realScoreGuidanceScale = 3.5;
			generatorUpdateInterval = 5;
			realScoreModelPath = "";
			fakeScoreModelPath = "";
		}
	});

	$effect(() => {
		if (!isOpen) return;
		isLoadingModels = true;
		const filter = isInference ? workloadType : getModelWorkloadForTraining(workloadType);
		getModels(filter)
			.then((list) => {
				models = list;
				const ids = list.map((m) => m.id);
				const opts = $defaultOptions;
				const defaultId = getDefaultModelForWorkload(
					opts,
					filter as "t2v" | "i2v" | "t2i",
				);
				modelId = ids.includes(defaultId) ? defaultId : list[0]?.id ?? "";
				if (workloadType === "dmd_t2v") {
					const mid = ids.includes(defaultId) ? defaultId : list[0]?.id ?? "";
					realScoreModelPath = mid;
					fakeScoreModelPath = mid;
				}
			})
			.catch((e) => console.error("Failed to load models:", e))
			.finally(() => (isLoadingModels = false));
	});

	$effect(() => {
		if (isOpen && !isInference) {
			getDatasets()
				.then(setReadyDatasets)
				.catch(() => (readyDatasets = []));
		} else {
			readyDatasets = [];
		}
	});

	function setReadyDatasets(list: Awaited<ReturnType<typeof getDatasets>>) {
		readyDatasets = list;
	}

	async function handleImageChange(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) {
			imagePath = "";
			imageFileName = "";
			return;
		}
		isUploadingImage = true;
		imageFileName = file.name;
		try {
			const { path } = await uploadImage(file);
			imagePath = path;
		} catch (err) {
			imagePath = "";
			imageFileName = "";
		} finally {
			isUploadingImage = false;
		}
	}

	function clearImage() {
		imagePath = "";
		imageFileName = "";
		if (imageInputEl) imageInputEl.value = "";
	}

	async function handleSubmit(e: SubmitEvent) {
		e.preventDefault();
		if (isInference && workloadType === "i2v" && !imagePath) return;
		const effectiveDataPath = selectedDatasetId
			? readyDatasets.find((d) => d.id === selectedDatasetId)?.name ?? ""
			: "";
		if (!isInference && !selectedDatasetId) return;
		const effectiveJobType: JobType =
			workloadType === "lora_t2v" ? "lora" : jobType;
		isSubmitting = true;
		try {
			const basePayload = {
				model_id: modelId,
				prompt,
				workload_type: workloadType,
				job_type: effectiveJobType,
						...(isInference
					? {}
					: {
							data_path: effectiveDataPath.trim(),
							max_train_steps: maxTrainSteps,
							train_batch_size: trainBatchSize,
							learning_rate: learningRate,
							num_latent_t: numLatentT,
							validation_dataset_file: selectedValidationDatasetId
								? (readyDatasets.find(
										(d) => d.id === selectedValidationDatasetId,
									)?.name ?? "") || undefined
								: undefined,
							lora_rank: loraRank,
							...(isLtx2Model
								? { ltx2_first_frame_conditioning_p: ltx2FirstFrameConditioningP }
								: {}),
							...(workloadType === "dmd_t2v"
								? {
										dmd_use_vsa: dmdUseVsa,
										dmd_vsa_sparsity: dmdVsaSparsity,
										dmd_denoising_steps: dmdDenoisingSteps,
										min_timestep_ratio: minTimestepRatio,
										max_timestep_ratio: maxTimestepRatio,
										real_score_guidance_scale: realScoreGuidanceScale,
										generator_update_interval: generatorUpdateInterval,
										real_score_model_path: realScoreModelPath || modelId,
										fake_score_model_path: fakeScoreModelPath || modelId,
									}
								: {}),
						}),
			};
			const inferencePayload = isInference
				? {
						...(workloadType === "i2v" && imagePath ? { image_path: imagePath } : {}),
						negative_prompt: negativePrompt,
						num_inference_steps: numInferenceSteps,
						num_frames: numFrames,
						height,
						width,
						guidance_scale: guidanceScale,
						guidance_rescale: guidanceRescale,
						fps,
						seed,
						num_gpus: numGpus,
						dit_cpu_offload: ditCpuOffload,
						text_encoder_cpu_offload: textEncoderCpuOffload,
						vae_cpu_offload: vaeCpuOffload,
						image_encoder_cpu_offload: imageEncoderCpuOffload,
						use_fsdp_inference: useFsdpInference,
						enable_torch_compile: enableTorchCompile,
						vsa_sparsity: vsaSparsity,
						tp_size: tpSize,
						sp_size: spSize,
					}
				: {};
			await createJob({ ...basePayload, ...inferencePayload });
			onSuccess();
			onClose();
		} catch (err) {
			console.error("Failed to create job:", err);
		} finally {
			isSubmitting = false;
		}
	}

	function handleClose() {
		if (isSubmitting) return;
		onClose();
	}

	function handleBackdropClick() {
		handleClose();
	}

	$effect(() => {
		if (!isOpen) return;
		function onKey(e: KeyboardEvent) {
			if (e.key === "Escape" && !isSubmitting) handleClose();
		}
		document.addEventListener("keydown", onKey);
		return () => document.removeEventListener("keydown", onKey);
	});

	const isLtx2Model = $derived(
		(modelId || "").toLowerCase().includes("ltx2") ||
			(modelId || "").toLowerCase().includes("ltx-2"),
	);

	const workloadLabel = $derived(
		WORKLOAD_OPTIONS[jobType]?.find((o) => o.type === workloadType)?.label ?? "",
	);
</script>

{#if isOpen}
	<div class="modal" data-modal>
		<div class="modalBackdrop" onclick={handleBackdropClick} role="presentation"></div>
		<div class="modalContent modalForm">
			<button
				class="modalClose"
				onclick={handleClose}
				disabled={isSubmitting}
				aria-label="Close"
			>
				×
			</button>
			<div class="card">
				<h2>
					New {jobType.charAt(0).toUpperCase() + jobType.slice(1)} Job
					{workloadLabel ? ` (${workloadLabel})` : ""}
				</h2>
				<form onsubmit={handleSubmit} autocomplete="off">
					<div class="formRow">
						<label for="modal-modelId">Model</label>
						<select
							id="modal-modelId"
							bind:value={modelId}
							required
							disabled={isSubmitting || isLoadingModels}
						>
							<option value="" disabled>
								{isLoadingModels
									? "Loading models…"
									: models.length === 0
										? "No models available for this workload"
										: "Select a model…"}
							</option>
							{#each models as model}
								<option value={model.id}>{model.label} ({model.id})</option>
							{/each}
						</select>
					</div>
					{#if isInference && workloadType === "i2v"}
						<div class="formRow">
							<label for="modal-image">Image</label>
							<input
								bind:this={imageInputEl}
								id="modal-image"
								type="file"
								accept=".png,.jpg,.jpeg,.webp,.bmp"
								onchange={handleImageChange}
								disabled={isSubmitting || isUploadingImage}
								required
							/>
							{#if imageFileName}
								<span class="helperText">
									{isUploadingImage ? "Uploading…" : imageFileName} ·
									<button
										type="button"
										class="clearLink"
										onclick={clearImage}
										disabled={isSubmitting || isUploadingImage}
									>
										Clear
									</button>
								</span>
							{/if}
						</div>
					{/if}
					<div class="formRow">
						<label for="modal-prompt">{isInference ? "Prompt" : "Description"}</label>
						<textarea
							id="modal-prompt"
							bind:value={prompt}
							rows={isInference ? 3 : 2}
							placeholder={
								isInference
									? "A curious raccoon peers through a vibrant field of yellow sunflowers…"
									: "Brief description of this training job…"
							}
							required
							disabled={isSubmitting}
						></textarea>
					</div>
					{#if isInference}
						<div class="formRow">
							<label for="modal-negative-prompt">Negative Prompt</label>
							<textarea
								id="modal-negative-prompt"
								bind:value={negativePrompt}
								rows={2}
								placeholder="Optional: things to avoid in the output…"
								disabled={isSubmitting}
							></textarea>
						</div>
					{/if}

					{#if !isInference}
						<div class="datasetRow">
							<div class="formRow">
								<label for="modal-dataset">Dataset *</label>
								<select
									id="modal-dataset"
									bind:value={selectedDatasetId}
									disabled={isSubmitting}
								>
									<option value="" disabled>
										{readyDatasets.length === 0
											? "No datasets (add in Datasets tab)"
											: "Select a dataset…"}
									</option>
									{#each readyDatasets as d}
										<option value={d.id}>{d.name}</option>
									{/each}
								</select>
							</div>
							<div class="formRow">
								<label for="modal-validation-dataset"
									>Validation Dataset (optional)</label
								>
								<select
									id="modal-validation-dataset"
									bind:value={selectedValidationDatasetId}
									disabled={isSubmitting}
								>
									<option value="">None</option>
									{#each readyDatasets as d}
										<option value={d.id}>{d.name}</option>
									{/each}
								</select>
							</div>
						</div>
						<details class="optionsSection">
							<summary>Options</summary>
							<div class="settingsGrid">
								<div class="formRow">
									<label for="modal-max-train-steps">Max Train Steps</label>
									<Slider
										id="modal-max-train-steps"
										min={100}
										max={50000}
										step={100}
										value={maxTrainSteps}
										onChange={(v) => (maxTrainSteps = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-train-batch-size">Train Batch Size</label>
									<Slider
										id="modal-train-batch-size"
										min={1}
										max={8}
										step={1}
										value={trainBatchSize}
										onChange={(v) => (trainBatchSize = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-learning-rate">Learning Rate</label>
									<input
										id="modal-learning-rate"
										type="number"
										step="1e-6"
										min={1e-6}
										max={1}
										bind:value={learningRate}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-num-latent-t">Num Latent T</label>
									<Slider
										id="modal-num-latent-t"
										min={8}
										max={40}
										step={1}
										value={numLatentT}
										onChange={(v) => (numLatentT = v)}
										disabled={isSubmitting}
									/>
								</div>
								{#if workloadType === "lora_t2v"}
									<div class="formRow">
										<label for="modal-lora-rank">LoRA Rank</label>
										<Slider
											id="modal-lora-rank"
											min={8}
											max={128}
											step={8}
											value={loraRank}
											onChange={(v) => (loraRank = v)}
											disabled={isSubmitting}
										/>
									</div>
								{/if}
								{#if isLtx2Model}
									<div class="formRow">
										<label for="modal-ltx2-first-frame" title="Probability of conditioning on the first frame during LTX-2 training">First Frame Conditioning</label>
										<Slider
											id="modal-ltx2-first-frame"
											min={0}
											max={1}
											step={0.05}
											value={ltx2FirstFrameConditioningP}
											onChange={(v) => (ltx2FirstFrameConditioningP = v)}
											disabled={isSubmitting}
											formatValue={(v) => v.toFixed(2)}
										/>
									</div>
								{/if}
								{#if workloadType === "dmd_t2v"}
									<div class="formRow">
										<label for="modal-dmd-use-vsa" title="Use Video Sparse Attention for DMD">Video Sparse Attention (VSA)</label>
										<Toggle
											id="modal-dmd-use-vsa"
											checked={dmdUseVsa}
											onChange={(v) => (dmdUseVsa = v)}
											disabled={isSubmitting}
										/>
									</div>
									{#if dmdUseVsa}
										<div class="formRow">
											<label for="modal-dmd-vsa-sparsity" title="VSA sparsity (0–1)">VSA Sparsity</label>
											<Slider
												id="modal-dmd-vsa-sparsity"
												min={0}
												max={1}
												step={0.05}
												value={dmdVsaSparsity}
												onChange={(v) => (dmdVsaSparsity = v)}
												disabled={isSubmitting}
												formatValue={(v) => v.toFixed(2)}
											/>
										</div>
									{/if}
									<div class="formRow">
										<label for="modal-dmd-denoising-steps" title="Comma-separated denoising steps, e.g. 1000,757,522">DMD Denoising Steps</label>
										<input
											id="modal-dmd-denoising-steps"
											type="text"
											bind:value={dmdDenoisingSteps}
											placeholder="1000,757,522"
											disabled={isSubmitting}
										/>
									</div>
									<div class="formRow">
										<label for="modal-min-timestep-ratio">Min Timestep Ratio</label>
										<Slider
											id="modal-min-timestep-ratio"
											min={0}
											max={1}
											step={0.01}
											value={minTimestepRatio}
											onChange={(v) => (minTimestepRatio = v)}
											disabled={isSubmitting}
											formatValue={(v) => v.toFixed(2)}
										/>
									</div>
									<div class="formRow">
										<label for="modal-max-timestep-ratio">Max Timestep Ratio</label>
										<Slider
											id="modal-max-timestep-ratio"
											min={0}
											max={1}
											step={0.01}
											value={maxTimestepRatio}
											onChange={(v) => (maxTimestepRatio = v)}
											disabled={isSubmitting}
											formatValue={(v) => v.toFixed(2)}
										/>
									</div>
									<div class="formRow">
										<label for="modal-real-score-guidance-scale">Real Score Guidance Scale</label>
										<Slider
											id="modal-real-score-guidance-scale"
											min={1}
											max={10}
											step={0.1}
											value={realScoreGuidanceScale}
											onChange={(v) => (realScoreGuidanceScale = v)}
											disabled={isSubmitting}
											formatValue={(v) => v.toFixed(1)}
										/>
									</div>
									<div class="formRow">
										<label for="modal-generator-update-interval">Generator Update Interval</label>
										<Slider
											id="modal-generator-update-interval"
											min={1}
											max={20}
											step={1}
											value={generatorUpdateInterval}
											onChange={(v) => (generatorUpdateInterval = v)}
											disabled={isSubmitting}
										/>
									</div>
									<div class="formRow">
										<label for="modal-real-score-model">Real Score Model</label>
										<select
											id="modal-real-score-model"
											bind:value={realScoreModelPath}
											disabled={isSubmitting || isLoadingModels}
										>
											<option value="">Same as main model</option>
											{#each models as model}
												<option value={model.id}>{model.label} ({model.id})</option>
											{/each}
										</select>
									</div>
									<div class="formRow">
										<label for="modal-fake-score-model">Fake Score Model</label>
										<select
											id="modal-fake-score-model"
											bind:value={fakeScoreModelPath}
											disabled={isSubmitting || isLoadingModels}
										>
											<option value="">Same as main model</option>
											{#each models as model}
												<option value={model.id}>{model.label} ({model.id})</option>
											{/each}
										</select>
									</div>
								{/if}
							</div>
						</details>
					{/if}

					{#if isInference}
						<details class="optionsSection">
							<summary>Options</summary>
							<div class="settingsGrid">
								{#if workloadType !== "t2i"}
									<div class="formRow">
										<label for="modal-num-frames">Frames</label>
										<Slider
											id="modal-num-frames"
											min={1}
											max={500}
											step={1}
											value={numFrames}
											onChange={(v) => (numFrames = v)}
											disabled={isSubmitting}
										/>
									</div>
								{/if}
								<div class="formRow">
									<label for="modal-height">Height</label>
									<Slider
										id="modal-height"
										min={64}
										max={1080}
										step={16}
										value={height}
										onChange={(v) => (height = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-width">Width</label>
									<Slider
										id="modal-width"
										min={64}
										max={1920}
										step={16}
										value={width}
										onChange={(v) => (width = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-num-steps">Inference Steps</label>
									<Slider
										id="modal-num-steps"
										min={1}
										max={200}
										step={1}
										value={numInferenceSteps}
										onChange={(v) => (numInferenceSteps = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-vsa-sparsity" title="VSA sparsity (0–1)">VSA Sparsity</label>
									<Slider
										id="modal-vsa-sparsity"
										min={0}
										max={1}
										step={0.05}
										value={vsaSparsity}
										onChange={(v) => (vsaSparsity = v)}
										disabled={isSubmitting}
										formatValue={(v) => v.toFixed(2)}
									/>
								</div>
								<div class="formRow">
									<label for="modal-guidance">Guidance Scale</label>
									<Slider
										id="modal-guidance"
										min={0}
										max={20}
										step={0.1}
										value={guidanceScale}
										onChange={(v) => (guidanceScale = v)}
										disabled={isSubmitting}
										formatValue={(v) => v.toFixed(1)}
									/>
								</div>
								<div class="formRow">
									<label for="modal-guidance-rescale" title="0 = disabled">Guidance Rescale</label>
									<Slider
										id="modal-guidance-rescale"
										min={0}
										max={1}
										step={0.05}
										value={guidanceRescale}
										onChange={(v) => (guidanceRescale = v)}
										disabled={isSubmitting}
										formatValue={(v) => v.toFixed(2)}
									/>
								</div>
								<div class="formRow">
									<label for="modal-tp-size" title="-1 = auto">TP Size</label>
									<Slider
										id="modal-tp-size"
										min={-1}
										max={8}
										step={1}
										value={tpSize}
										onChange={(v) => (tpSize = v)}
										disabled={isSubmitting}
										formatValue={(v) => (v === -1 ? "Auto" : String(v))}
									/>
								</div>
								<div class="formRow">
									<label for="modal-sp-size" title="-1 = auto">SP Size</label>
									<Slider
										id="modal-sp-size"
										min={-1}
										max={8}
										step={1}
										value={spSize}
										onChange={(v) => (spSize = v)}
										disabled={isSubmitting}
										formatValue={(v) => (v === -1 ? "Auto" : String(v))}
									/>
								</div>
								{#if workloadType !== "t2i"}
									<div class="formRow">
										<label for="modal-fps">FPS</label>
										<Slider
											id="modal-fps"
											min={1}
											max={60}
											step={1}
											value={fps}
											onChange={(v) => (fps = v)}
											disabled={isSubmitting}
										/>
									</div>
								{/if}
								<div class="formRow">
									<label for="modal-dit-cpu-offload">DiT CPU Offload</label>
									<Toggle
										id="modal-dit-cpu-offload"
										checked={ditCpuOffload}
										onChange={(v) => (ditCpuOffload = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-text-encoder-cpu-offload">Text Encoder CPU Offload</label>
									<Toggle
										id="modal-text-encoder-cpu-offload"
										checked={textEncoderCpuOffload}
										onChange={(v) => (textEncoderCpuOffload = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-use-fsdp-inference">Use FSDP Inference</label>
									<Toggle
										id="modal-use-fsdp-inference"
										checked={useFsdpInference}
										onChange={(v) => (useFsdpInference = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-vae-cpu-offload">VAE CPU Offload</label>
									<Toggle
										id="modal-vae-cpu-offload"
										checked={vaeCpuOffload}
										onChange={(v) => (vaeCpuOffload = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-image-encoder-cpu-offload">Image Encoder CPU Offload</label>
									<Toggle
										id="modal-image-encoder-cpu-offload"
										checked={imageEncoderCpuOffload}
										onChange={(v) => (imageEncoderCpuOffload = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-enable-torch-compile">Torch Compile</label>
									<Toggle
										id="modal-enable-torch-compile"
										checked={enableTorchCompile}
										onChange={(v) => (enableTorchCompile = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-num-gpus">GPUs</label>
									<Slider
										id="modal-num-gpus"
										min={1}
										max={8}
										step={1}
										value={numGpus}
										onChange={(v) => (numGpus = v)}
										disabled={isSubmitting}
									/>
								</div>
								<div class="formRow">
									<label for="modal-seed">Seed</label>
									<input
										id="modal-seed"
										type="number"
										bind:value={seed}
										min={0}
										disabled={isSubmitting}
									/>
								</div>
							</div>
						</details>
					{/if}

					<button type="submit" class="btn btnPrimary" disabled={isSubmitting}>
						{isSubmitting ? "Creating..." : "Create Job"}
					</button>
				</form>
			</div>
		</div>
	</div>
{/if}

<style>
	.modal {
		position: fixed;
		inset: 0;
		z-index: 1000;
		display: flex;
		align-items: center;
		justify-content: center;
		animation: modal-fade-in 0.2s ease-out;
	}
	.modalBackdrop {
		position: absolute;
		inset: 0;
		background: rgba(0, 0, 0, 0.7);
	}
	.modalContent {
		position: relative;
		z-index: 1;
		max-width: 90vw;
		max-height: 90vh;
		border-radius: var(--radius);
		overflow: hidden;
		background: #000;
	}
	.modalForm {
		background: var(--surface);
		border: 1px solid var(--border);
		padding: 0;
		max-width: 850px;
		width: 90vw;
		max-height: 90vh;
		overflow-y: auto;
	}
	.modalClose {
		position: absolute;
		top: 0.5rem;
		right: 0.75rem;
		background: none;
		border: none;
		color: var(--text);
		font-size: 1.6rem;
		cursor: pointer;
		z-index: 2;
		line-height: 1;
	}
	.modalClose:hover {
		opacity: 0.7;
	}
	.modalClose:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.card {
		padding: 1.5rem;
		margin: 0;
		border: none;
	}
	.card h2 {
		font-size: 1.15rem;
		margin-bottom: 1rem;
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
		padding-left: 3px;
		letter-spacing: 0.04em;
	}
	.datasetRow {
		display: flex;
		gap: 1rem;
		margin-bottom: 0.85rem;
	}
	.datasetRow .formRow {
		flex: 1;
		min-width: 0;
		margin-bottom: 0;
	}
	.helperText {
		font-size: 0.75rem;
		color: var(--text-dim);
		margin-top: 0.2rem;
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
	.optionsSection {
		margin-bottom: 1rem;
	}
	.optionsSection summary {
		cursor: pointer;
		font-size: 0.85rem;
		color: var(--accent-h);
		user-select: none;
		margin-bottom: 0.5rem;
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
	.btnPrimary {
		background: var(--accent);
		color: #fff;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	@keyframes modal-fade-in {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}
</style>
