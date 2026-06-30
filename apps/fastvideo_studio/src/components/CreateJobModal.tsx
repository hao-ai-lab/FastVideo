'use client';

import * as React from 'react';

import { cn } from '@/lib/utils';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { NativeSelect } from '@/components/ui/native-select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { useStore } from '@/hooks/useStore';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import {
  createJob,
  getDatasets,
  getModels,
  uploadImage,
  type CreateJobRequest,
  type Model,
} from '@/lib/api';
import { getDefaultModelForWorkload } from '@/lib/defaultOptions';
import { WORKLOAD_OPTIONS } from '@/lib/jobConfig';
import type { JobType } from '@/lib/types';

export interface CreateJobModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  jobType: JobType;
  workloadType: string;
}

// --- Small presentational helpers (kept local so we don't touch the
// foundation primitives). They mirror the Svelte `Toggle`/`Slider` UX on top
// of the shadcn `Switch`/`Slider`. ---

function FieldRow({
  htmlFor,
  label,
  title,
  className,
  children,
}: {
  htmlFor: string;
  label: string;
  title?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      <Label
        htmlFor={htmlFor}
        title={title}
        className="pl-0.5 text-xs font-normal tracking-wide text-muted-foreground"
      >
        {label}
      </Label>
      {children}
    </div>
  );
}

function SliderRow({
  id,
  label,
  title,
  min,
  max,
  step,
  value,
  onChange,
  disabled,
  formatValue = (v) => String(v),
}: {
  id: string;
  label: string;
  title?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  formatValue?: (v: number) => string;
}) {
  return (
    <FieldRow htmlFor={id} label={label} title={title}>
      <div className="flex items-center gap-2">
        <Slider
          id={id}
          min={min}
          max={max}
          step={step}
          value={[value]}
          onValueChange={(v) => onChange(v[0])}
          disabled={disabled}
          aria-label={label}
          className="min-w-0 flex-1"
        />
        <span
          aria-hidden="true"
          className="min-w-10 shrink-0 text-right text-sm tabular-nums text-muted-foreground"
        >
          {formatValue(value)}
        </span>
      </div>
    </FieldRow>
  );
}

function ToggleRow({
  id,
  label,
  title,
  checked,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  title?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <FieldRow htmlFor={id} label={label} title={title}>
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
      />
    </FieldRow>
  );
}

function getModelWorkloadForTraining(_w: string): string {
  return 't2v';
}

export default function CreateJobModal({
  isOpen,
  onClose,
  onSuccess,
  jobType,
  workloadType,
}: CreateJobModalProps) {
  const { options } = useStore(defaultOptionsStore);

  const isInference = jobType === 'inference';
  const inferenceWorkload = isInference
    ? workloadType
    : getModelWorkloadForTraining(workloadType);

  const [models, setModels] = React.useState<Model[]>([]);
  const [modelId, setModelId] = React.useState('');
  const [prompt, setPrompt] = React.useState('');
  const [imagePath, setImagePath] = React.useState('');
  const [imageFileName, setImageFileName] = React.useState('');
  const [isUploadingImage, setIsUploadingImage] = React.useState(false);
  const [negativePrompt, setNegativePrompt] = React.useState('');
  const [numInferenceSteps, setNumInferenceSteps] = React.useState(50);
  const [numFrames, setNumFrames] = React.useState(81);
  const [height, setHeight] = React.useState(480);
  const [width, setWidth] = React.useState(832);
  const [guidanceScale, setGuidanceScale] = React.useState(5);
  const [guidanceRescale, setGuidanceRescale] = React.useState(0);
  const [fps, setFps] = React.useState(24);
  const [seed, setSeed] = React.useState(1024);
  const [numGpus, setNumGpus] = React.useState(1);
  const [ditCpuOffload, setDitCpuOffload] = React.useState(false);
  const [textEncoderCpuOffload, setTextEncoderCpuOffload] =
    React.useState(false);
  const [vaeCpuOffload, setVaeCpuOffload] = React.useState(false);
  const [imageEncoderCpuOffload, setImageEncoderCpuOffload] =
    React.useState(false);
  const [useFsdpInference, setUseFsdpInference] = React.useState(false);
  const [enableTorchCompile, setEnableTorchCompile] = React.useState(false);
  const [vsaSparsity, setVsaSparsity] = React.useState(0);
  const [tpSize, setTpSize] = React.useState(-1);
  const [spSize, setSpSize] = React.useState(-1);
  const [selectedDatasetId, setSelectedDatasetId] = React.useState('');
  const [readyDatasets, setReadyDatasets] = React.useState<
    Awaited<ReturnType<typeof getDatasets>>
  >([]);
  const [maxTrainSteps, setMaxTrainSteps] = React.useState(1000);
  const [trainBatchSize, setTrainBatchSize] = React.useState(1);
  const [learningRate, setLearningRate] = React.useState(5e-5);
  const [numLatentT, setNumLatentT] = React.useState(20);
  const [selectedValidationDatasetId, setSelectedValidationDatasetId] =
    React.useState('');
  const [loraRank, setLoraRank] = React.useState(32);
  const [ltx2FirstFrameConditioningP, setLtx2FirstFrameConditioningP] =
    React.useState(0.1);
  const [dmdUseVsa, setDmdUseVsa] = React.useState(false);
  const [dmdVsaSparsity, setDmdVsaSparsity] = React.useState(0.8);
  const [dmdDenoisingSteps, setDmdDenoisingSteps] =
    React.useState('1000,757,522');
  const [minTimestepRatio, setMinTimestepRatio] = React.useState(0.02);
  const [maxTimestepRatio, setMaxTimestepRatio] = React.useState(0.98);
  const [realScoreGuidanceScale, setRealScoreGuidanceScale] =
    React.useState(3.5);
  const [generatorUpdateInterval, setGeneratorUpdateInterval] =
    React.useState(5);
  const [realScoreModelPath, setRealScoreModelPath] = React.useState('');
  const [fakeScoreModelPath, setFakeScoreModelPath] = React.useState('');
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isLoadingModels, setIsLoadingModels] = React.useState(false);
  const imageInputRef = React.useRef<HTMLInputElement>(null);

  const isLtx2Model =
    (modelId || '').toLowerCase().includes('ltx2') ||
    (modelId || '').toLowerCase().includes('ltx-2');

  // Seed field values from the persisted default options each time the modal
  // OPENS. A naive port of the Svelte `$effect` would re-seed on every
  // `$defaultOptions` change; we deliberately seed only on the open transition
  // so a late `initDefaultOptions()` settings refresh can't clobber the user's
  // in-progress edits or desync the (already validated) model selection.
  const justOpenedRef = React.useRef(false);
  React.useEffect(() => {
    const justOpened = isOpen && !justOpenedRef.current;
    justOpenedRef.current = isOpen;
    if (!justOpened) return;
    const opts = options;
    setNumInferenceSteps(opts.numInferenceSteps);
    setNumFrames(workloadType === 't2i' ? 1 : opts.numFrames);
    setHeight(opts.height);
    setWidth(opts.width);
    setGuidanceScale(opts.guidanceScale);
    setGuidanceRescale(opts.guidanceRescale);
    setFps(opts.fps);
    setSeed(opts.seed);
    setNumGpus(opts.numGpus);
    setDitCpuOffload(opts.ditCpuOffload);
    setTextEncoderCpuOffload(opts.textEncoderCpuOffload);
    setVaeCpuOffload(opts.vaeCpuOffload);
    setImageEncoderCpuOffload(opts.imageEncoderCpuOffload);
    setUseFsdpInference(opts.useFsdpInference);
    setEnableTorchCompile(opts.enableTorchCompile);
    setVsaSparsity(opts.vsaSparsity);
    setTpSize(opts.tpSize);
    setSpSize(opts.spSize);
    setModelId(
      getDefaultModelForWorkload(
        opts,
        inferenceWorkload as 't2v' | 'i2v' | 't2i',
      ),
    );
    setImagePath('');
    setImageFileName('');
    setSelectedDatasetId('');
    setSelectedValidationDatasetId('');
    if (workloadType === 'dmd_t2v') {
      setDmdUseVsa(false);
      setDmdVsaSparsity(0.8);
      setDmdDenoisingSteps('1000,757,522');
      setMinTimestepRatio(0.02);
      setMaxTimestepRatio(0.98);
      setRealScoreGuidanceScale(3.5);
      setGeneratorUpdateInterval(5);
      setRealScoreModelPath('');
      setFakeScoreModelPath('');
    }
  }, [isOpen, workloadType, inferenceWorkload, options]);

  // Load the models available for this workload.
  React.useEffect(() => {
    if (!isOpen) return;
    setIsLoadingModels(true);
    const filter = isInference
      ? workloadType
      : getModelWorkloadForTraining(workloadType);
    getModels(filter)
      .then((list) => {
        setModels(list);
        const ids = list.map((m) => m.id);
        const opts = defaultOptionsStore.get().options;
        const defaultId = getDefaultModelForWorkload(
          opts,
          filter as 't2v' | 'i2v' | 't2i',
        );
        const chosen = ids.includes(defaultId) ? defaultId : (list[0]?.id ?? '');
        setModelId(chosen);
        if (workloadType === 'dmd_t2v') {
          setRealScoreModelPath(chosen);
          setFakeScoreModelPath(chosen);
        }
      })
      .catch((e) => console.error('Failed to load models:', e))
      .finally(() => setIsLoadingModels(false));
  }, [isOpen, isInference, workloadType]);

  // Training jobs need a dataset; load the ready datasets when relevant.
  React.useEffect(() => {
    if (isOpen && !isInference) {
      getDatasets()
        .then(setReadyDatasets)
        .catch(() => setReadyDatasets([]));
    } else {
      setReadyDatasets([]);
    }
  }, [isOpen, isInference]);

  async function handleImageChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) {
      setImagePath('');
      setImageFileName('');
      return;
    }
    setIsUploadingImage(true);
    setImageFileName(file.name);
    try {
      const { path } = await uploadImage(file);
      setImagePath(path);
    } catch {
      setImagePath('');
      setImageFileName('');
    } finally {
      setIsUploadingImage(false);
    }
  }

  function clearImage() {
    setImagePath('');
    setImageFileName('');
    if (imageInputRef.current) imageInputRef.current.value = '';
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (isInference && workloadType === 'i2v' && !imagePath) return;
    const effectiveDataPath = selectedDatasetId
      ? (readyDatasets.find((d) => d.id === selectedDatasetId)?.name ?? '')
      : '';
    if (!isInference && !selectedDatasetId) return;
    // `lora_t2v` jobs are persisted with a dedicated backend job_type that the
    // front-end JobType enum does not model; cast to keep payload parity.
    const effectiveJobType = (
      workloadType === 'lora_t2v' ? 'lora' : jobType
    ) as JobType;
    setIsSubmitting(true);
    try {
      const payload: CreateJobRequest = {
        model_id: modelId,
        prompt,
        workload_type: workloadType,
        job_type: effectiveJobType,
        ...(isInference
          ? {
              ...(workloadType === 'i2v' && imagePath
                ? { image_path: imagePath }
                : {}),
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
          : {
              data_path: effectiveDataPath.trim(),
              max_train_steps: maxTrainSteps,
              train_batch_size: trainBatchSize,
              learning_rate: learningRate,
              num_latent_t: numLatentT,
              validation_dataset_file: selectedValidationDatasetId
                ? (readyDatasets.find(
                    (d) => d.id === selectedValidationDatasetId,
                  )?.name ?? '') || undefined
                : undefined,
              lora_rank: loraRank,
              ...(isLtx2Model
                ? {
                    ltx2_first_frame_conditioning_p:
                      ltx2FirstFrameConditioningP,
                  }
                : {}),
              ...(workloadType === 'dmd_t2v'
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
      await createJob(payload);
      onSuccess();
      onClose();
    } catch (err) {
      console.error('Failed to create job:', err);
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleClose() {
    if (isSubmitting) return;
    onClose();
  }

  const workloadLabel =
    WORKLOAD_OPTIONS[jobType]?.find((o) => o.type === workloadType)?.label ?? '';
  const title = `New ${jobType.charAt(0).toUpperCase() + jobType.slice(1)} Job${
    workloadLabel ? ` (${workloadLabel})` : ''
  }`;

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) handleClose();
      }}
    >
      <DialogContent
        className="max-h-[90vh] w-[90vw] max-w-[850px] overflow-y-auto"
        onEscapeKeyDown={(e) => {
          if (isSubmitting) e.preventDefault();
        }}
        onInteractOutside={(e) => {
          if (isSubmitting) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        <form
          onSubmit={handleSubmit}
          autoComplete="off"
          className="flex flex-col gap-3.5"
        >
          <FieldRow htmlFor="modal-modelId" label="Model">
            <NativeSelect
              id="modal-modelId"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              required
              disabled={isSubmitting || isLoadingModels}
            >
              <option value="" disabled>
                {isLoadingModels
                  ? 'Loading models…'
                  : models.length === 0
                    ? 'No models available for this workload'
                    : 'Select a model…'}
              </option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label} ({model.id})
                </option>
              ))}
            </NativeSelect>
          </FieldRow>

          {isInference && workloadType === 'i2v' && (
            <FieldRow htmlFor="modal-image" label="Image">
              <Input
                ref={imageInputRef}
                id="modal-image"
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.bmp"
                onChange={handleImageChange}
                disabled={isSubmitting || isUploadingImage}
                required
                className="h-auto py-2 file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-secondary file:px-2 file:py-1 file:text-sm file:text-secondary-foreground"
              />
              {imageFileName && (
                <span className="mt-0.5 text-xs text-muted-foreground">
                  {isUploadingImage ? 'Uploading…' : imageFileName} ·{' '}
                  <button
                    type="button"
                    onClick={clearImage}
                    disabled={isSubmitting || isUploadingImage}
                    className="text-indigo-400 underline-offset-2 hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Clear
                  </button>
                </span>
              )}
            </FieldRow>
          )}

          <FieldRow
            htmlFor="modal-prompt"
            label={isInference ? 'Prompt' : 'Description'}
          >
            <Textarea
              id="modal-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={isInference ? 3 : 2}
              placeholder={
                isInference
                  ? 'A curious raccoon peers through a vibrant field of yellow sunflowers…'
                  : 'Brief description of this training job…'
              }
              required
              disabled={isSubmitting}
            />
          </FieldRow>

          {isInference && (
            <FieldRow htmlFor="modal-negative-prompt" label="Negative Prompt">
              <Textarea
                id="modal-negative-prompt"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
                placeholder="Optional: things to avoid in the output…"
                disabled={isSubmitting}
              />
            </FieldRow>
          )}

          {!isInference && (
            <>
              <div className="flex gap-4">
                <FieldRow
                  htmlFor="modal-dataset"
                  label="Dataset *"
                  className="min-w-0 flex-1"
                >
                  <NativeSelect
                    id="modal-dataset"
                    value={selectedDatasetId}
                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                    disabled={isSubmitting}
                  >
                    <option value="" disabled>
                      {readyDatasets.length === 0
                        ? 'No datasets (add in Datasets tab)'
                        : 'Select a dataset…'}
                    </option>
                    {readyDatasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                  </NativeSelect>
                </FieldRow>
                <FieldRow
                  htmlFor="modal-validation-dataset"
                  label="Validation Dataset (optional)"
                  className="min-w-0 flex-1"
                >
                  <NativeSelect
                    id="modal-validation-dataset"
                    value={selectedValidationDatasetId}
                    onChange={(e) =>
                      setSelectedValidationDatasetId(e.target.value)
                    }
                    disabled={isSubmitting}
                  >
                    <option value="">None</option>
                    {readyDatasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                  </NativeSelect>
                </FieldRow>
              </div>

              <details>
                <summary className="mb-2 cursor-pointer select-none text-sm text-indigo-400">
                  Options
                </summary>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-x-3 gap-y-2">
                  <SliderRow
                    id="modal-max-train-steps"
                    label="Max Train Steps"
                    min={100}
                    max={50000}
                    step={100}
                    value={maxTrainSteps}
                    onChange={setMaxTrainSteps}
                    disabled={isSubmitting}
                  />
                  <SliderRow
                    id="modal-train-batch-size"
                    label="Train Batch Size"
                    min={1}
                    max={8}
                    step={1}
                    value={trainBatchSize}
                    onChange={setTrainBatchSize}
                    disabled={isSubmitting}
                  />
                  <FieldRow
                    htmlFor="modal-learning-rate"
                    label="Learning Rate"
                  >
                    <Input
                      id="modal-learning-rate"
                      type="number"
                      step="1e-6"
                      min={1e-6}
                      max={1}
                      value={learningRate}
                      onChange={(e) => {
                        const v = e.target.valueAsNumber;
                        if (!Number.isNaN(v)) setLearningRate(v);
                      }}
                      disabled={isSubmitting}
                    />
                  </FieldRow>
                  <SliderRow
                    id="modal-num-latent-t"
                    label="Num Latent T"
                    min={8}
                    max={40}
                    step={1}
                    value={numLatentT}
                    onChange={setNumLatentT}
                    disabled={isSubmitting}
                  />
                  {workloadType === 'lora_t2v' && (
                    <SliderRow
                      id="modal-lora-rank"
                      label="LoRA Rank"
                      min={8}
                      max={128}
                      step={8}
                      value={loraRank}
                      onChange={setLoraRank}
                      disabled={isSubmitting}
                    />
                  )}
                  {isLtx2Model && (
                    <SliderRow
                      id="modal-ltx2-first-frame"
                      label="First Frame Conditioning"
                      title="Probability of conditioning on the first frame during LTX-2 training"
                      min={0}
                      max={1}
                      step={0.05}
                      value={ltx2FirstFrameConditioningP}
                      onChange={setLtx2FirstFrameConditioningP}
                      disabled={isSubmitting}
                      formatValue={(v) => v.toFixed(2)}
                    />
                  )}
                  {workloadType === 'dmd_t2v' && (
                    <>
                      <ToggleRow
                        id="modal-dmd-use-vsa"
                        label="Video Sparse Attention (VSA)"
                        title="Use Video Sparse Attention for DMD"
                        checked={dmdUseVsa}
                        onChange={setDmdUseVsa}
                        disabled={isSubmitting}
                      />
                      {dmdUseVsa && (
                        <SliderRow
                          id="modal-dmd-vsa-sparsity"
                          label="VSA Sparsity"
                          title="VSA sparsity (0–1)"
                          min={0}
                          max={1}
                          step={0.05}
                          value={dmdVsaSparsity}
                          onChange={setDmdVsaSparsity}
                          disabled={isSubmitting}
                          formatValue={(v) => v.toFixed(2)}
                        />
                      )}
                      <FieldRow
                        htmlFor="modal-dmd-denoising-steps"
                        label="DMD Denoising Steps"
                        title="Comma-separated denoising steps, e.g. 1000,757,522"
                      >
                        <Input
                          id="modal-dmd-denoising-steps"
                          type="text"
                          value={dmdDenoisingSteps}
                          onChange={(e) => setDmdDenoisingSteps(e.target.value)}
                          placeholder="1000,757,522"
                          disabled={isSubmitting}
                        />
                      </FieldRow>
                      <SliderRow
                        id="modal-min-timestep-ratio"
                        label="Min Timestep Ratio"
                        min={0}
                        max={1}
                        step={0.01}
                        value={minTimestepRatio}
                        onChange={setMinTimestepRatio}
                        disabled={isSubmitting}
                        formatValue={(v) => v.toFixed(2)}
                      />
                      <SliderRow
                        id="modal-max-timestep-ratio"
                        label="Max Timestep Ratio"
                        min={0}
                        max={1}
                        step={0.01}
                        value={maxTimestepRatio}
                        onChange={setMaxTimestepRatio}
                        disabled={isSubmitting}
                        formatValue={(v) => v.toFixed(2)}
                      />
                      <SliderRow
                        id="modal-real-score-guidance-scale"
                        label="Real Score Guidance Scale"
                        min={1}
                        max={10}
                        step={0.1}
                        value={realScoreGuidanceScale}
                        onChange={setRealScoreGuidanceScale}
                        disabled={isSubmitting}
                        formatValue={(v) => v.toFixed(1)}
                      />
                      <SliderRow
                        id="modal-generator-update-interval"
                        label="Generator Update Interval"
                        min={1}
                        max={20}
                        step={1}
                        value={generatorUpdateInterval}
                        onChange={setGeneratorUpdateInterval}
                        disabled={isSubmitting}
                      />
                      <FieldRow
                        htmlFor="modal-real-score-model"
                        label="Real Score Model"
                      >
                        <NativeSelect
                          id="modal-real-score-model"
                          value={realScoreModelPath}
                          onChange={(e) =>
                            setRealScoreModelPath(e.target.value)
                          }
                          disabled={isSubmitting || isLoadingModels}
                        >
                          <option value="">Same as main model</option>
                          {models.map((model) => (
                            <option key={model.id} value={model.id}>
                              {model.label} ({model.id})
                            </option>
                          ))}
                        </NativeSelect>
                      </FieldRow>
                      <FieldRow
                        htmlFor="modal-fake-score-model"
                        label="Fake Score Model"
                      >
                        <NativeSelect
                          id="modal-fake-score-model"
                          value={fakeScoreModelPath}
                          onChange={(e) =>
                            setFakeScoreModelPath(e.target.value)
                          }
                          disabled={isSubmitting || isLoadingModels}
                        >
                          <option value="">Same as main model</option>
                          {models.map((model) => (
                            <option key={model.id} value={model.id}>
                              {model.label} ({model.id})
                            </option>
                          ))}
                        </NativeSelect>
                      </FieldRow>
                    </>
                  )}
                </div>
              </details>
            </>
          )}

          {isInference && (
            <details>
              <summary className="mb-2 cursor-pointer select-none text-sm text-indigo-400">
                Options
              </summary>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-x-3 gap-y-2">
                {workloadType !== 't2i' && (
                  <SliderRow
                    id="modal-num-frames"
                    label="Frames"
                    min={1}
                    max={500}
                    step={1}
                    value={numFrames}
                    onChange={setNumFrames}
                    disabled={isSubmitting}
                  />
                )}
                <SliderRow
                  id="modal-height"
                  label="Height"
                  min={64}
                  max={1080}
                  step={16}
                  value={height}
                  onChange={setHeight}
                  disabled={isSubmitting}
                />
                <SliderRow
                  id="modal-width"
                  label="Width"
                  min={64}
                  max={1920}
                  step={16}
                  value={width}
                  onChange={setWidth}
                  disabled={isSubmitting}
                />
                <SliderRow
                  id="modal-num-steps"
                  label="Inference Steps"
                  min={1}
                  max={200}
                  step={1}
                  value={numInferenceSteps}
                  onChange={setNumInferenceSteps}
                  disabled={isSubmitting}
                />
                <SliderRow
                  id="modal-vsa-sparsity"
                  label="VSA Sparsity"
                  title="VSA sparsity (0–1)"
                  min={0}
                  max={1}
                  step={0.05}
                  value={vsaSparsity}
                  onChange={setVsaSparsity}
                  disabled={isSubmitting}
                  formatValue={(v) => v.toFixed(2)}
                />
                <SliderRow
                  id="modal-guidance"
                  label="Guidance Scale"
                  min={0}
                  max={20}
                  step={0.1}
                  value={guidanceScale}
                  onChange={setGuidanceScale}
                  disabled={isSubmitting}
                  formatValue={(v) => v.toFixed(1)}
                />
                <SliderRow
                  id="modal-guidance-rescale"
                  label="Guidance Rescale"
                  title="0 = disabled"
                  min={0}
                  max={1}
                  step={0.05}
                  value={guidanceRescale}
                  onChange={setGuidanceRescale}
                  disabled={isSubmitting}
                  formatValue={(v) => v.toFixed(2)}
                />
                <SliderRow
                  id="modal-tp-size"
                  label="TP Size"
                  title="-1 = auto"
                  min={-1}
                  max={8}
                  step={1}
                  value={tpSize}
                  onChange={setTpSize}
                  disabled={isSubmitting}
                  formatValue={(v) => (v === -1 ? 'Auto' : String(v))}
                />
                <SliderRow
                  id="modal-sp-size"
                  label="SP Size"
                  title="-1 = auto"
                  min={-1}
                  max={8}
                  step={1}
                  value={spSize}
                  onChange={setSpSize}
                  disabled={isSubmitting}
                  formatValue={(v) => (v === -1 ? 'Auto' : String(v))}
                />
                {workloadType !== 't2i' && (
                  <SliderRow
                    id="modal-fps"
                    label="FPS"
                    min={1}
                    max={60}
                    step={1}
                    value={fps}
                    onChange={setFps}
                    disabled={isSubmitting}
                  />
                )}
                <ToggleRow
                  id="modal-dit-cpu-offload"
                  label="DiT CPU Offload"
                  checked={ditCpuOffload}
                  onChange={setDitCpuOffload}
                  disabled={isSubmitting}
                />
                <ToggleRow
                  id="modal-text-encoder-cpu-offload"
                  label="Text Encoder CPU Offload"
                  checked={textEncoderCpuOffload}
                  onChange={setTextEncoderCpuOffload}
                  disabled={isSubmitting}
                />
                <ToggleRow
                  id="modal-use-fsdp-inference"
                  label="Use FSDP Inference"
                  checked={useFsdpInference}
                  onChange={setUseFsdpInference}
                  disabled={isSubmitting}
                />
                <ToggleRow
                  id="modal-vae-cpu-offload"
                  label="VAE CPU Offload"
                  checked={vaeCpuOffload}
                  onChange={setVaeCpuOffload}
                  disabled={isSubmitting}
                />
                <ToggleRow
                  id="modal-image-encoder-cpu-offload"
                  label="Image Encoder CPU Offload"
                  checked={imageEncoderCpuOffload}
                  onChange={setImageEncoderCpuOffload}
                  disabled={isSubmitting}
                />
                <ToggleRow
                  id="modal-enable-torch-compile"
                  label="Torch Compile"
                  checked={enableTorchCompile}
                  onChange={setEnableTorchCompile}
                  disabled={isSubmitting}
                />
                <SliderRow
                  id="modal-num-gpus"
                  label="GPUs"
                  min={1}
                  max={8}
                  step={1}
                  value={numGpus}
                  onChange={setNumGpus}
                  disabled={isSubmitting}
                />
                <FieldRow htmlFor="modal-seed" label="Seed">
                  <Input
                    id="modal-seed"
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(e) => {
                      const v = e.target.valueAsNumber;
                      if (!Number.isNaN(v)) setSeed(v);
                    }}
                    disabled={isSubmitting}
                  />
                </FieldRow>
              </div>
            </details>
          )}

          <div>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Creating...' : 'Create Job'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
