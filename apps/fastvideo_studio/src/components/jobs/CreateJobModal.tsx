'use client';

import * as React from 'react';

import {
  FieldRow,
  NumberRow,
  SliderRow,
  ToggleRow,
} from '@/components/form-rows';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { NativeSelect } from '@/components/ui/native-select';
import { Textarea } from '@/components/ui/textarea';
import { useStore } from '@/hooks/useStore';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import {
  createJob,
  updateJob,
  getDatasets,
  getModels,
  uploadImage,
  uploadMedia,
  type CreateJobRequest,
  type Model,
} from '@/lib/api';
import { getDefaultModelForWorkload } from '@/lib/defaultOptions';
import { WORKLOAD_OPTIONS } from '@/lib/jobConfig';
import type { JobType } from '@/lib/types';
import {
  H3_MAX_REFERENCES,
  labelReferences,
  referencePromptSeed,
  validateReferences,
  type H3Reference,
} from '@/lib/h3References';
import {
  EMPTY_H3_PROMPT_FIELDS,
  H3_PROMPT_SECTIONS,
  H3_SECTION_HINTS,
  H3_SECTION_LABELS,
  isEmptyPromptFields,
  parseH3Prompt,
  serializeH3Prompt,
  type H3PromptFields,
} from '@/lib/h3Prompt';
import { jobToFormFields, type JobLike } from '@/lib/jobToFields';

export interface CreateJobModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  jobType: JobType;
  workloadType: string;
  /** When set, the modal edits this pending job instead of creating a new one. */
  editingJob?: JobLike | null;
  /** Show the configuration without allowing changes (started/finished jobs). */
  readOnly?: boolean;
}

export default function CreateJobModal({
  isOpen,
  onClose,
  onSuccess,
  jobType,
  workloadType,
  editingJob,
  readOnly = false,
}: CreateJobModalProps) {
  const { options } = useStore(defaultOptionsStore);

  const isInference = jobType === 'inference';
  // Training jobs always pick from the t2v model catalogue.
  const inferenceWorkload = isInference ? workloadType : 't2v';

  const [models, setModels] = React.useState<Model[]>([]);
  const [modelId, setModelId] = React.useState('');
  const [name, setName] = React.useState('');
  const [prompt, setPrompt] = React.useState('');
  const [imagePath, setImagePath] = React.useState('');
  const [lastImagePath, setLastImagePath] = React.useState('');
  const [references, setReferences] = React.useState<H3Reference[]>([]);
  const [isUploadingReference, setIsUploadingReference] = React.useState(false);
  const [referenceError, setReferenceError] = React.useState<string | null>(null);
  const [promptFields, setPromptFields] = React.useState<H3PromptFields>(
    EMPTY_H3_PROMPT_FIELDS,
  );
  const [useGuidedPrompt, setUseGuidedPrompt] = React.useState(true);
  const [lastImageFileName, setLastImageFileName] = React.useState('');
  const [isUploadingLastImage, setIsUploadingLastImage] = React.useState(false);
  const [lastImageUploadError, setLastImageUploadError] = React.useState<
    string | null
  >(null);
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
  const [ditLayerwiseOffload, setDitLayerwiseOffload] = React.useState(false);
  const [textEncoderCpuOffload, setTextEncoderCpuOffload] =
    React.useState(false);
  const [vaeCpuOffload, setVaeCpuOffload] = React.useState(false);
  const [imageEncoderCpuOffload, setImageEncoderCpuOffload] =
    React.useState(false);
  const [useFsdpInference, setUseFsdpInference] = React.useState(false);

  // H3 is the only registered model with an end frame or references.
  const supportsLastImage = modelId.toLowerCase().includes('minimax-h3');
  const usingReferences = supportsLastImage && references.length > 0;

  // JobCard re-renders on every job-list poll, so `editingJob` is a fresh
  // object each time. Effects must depend on these, never on the object.
  const editingJobId = editingJob?.id ?? null;
  const editingJobModelId = editingJob?.model_id ?? null;

  // Layerwise offload and FSDP compete for the DiT weights and FastVideoArgs
  // silently picks a winner (fastvideo_args.py:859); resolve it visibly here.
  // dit_cpu_offload is deliberately not interlocked -- it is a modifier, not a
  // competing strategy.
  const handleDitLayerwiseOffloadChange = React.useCallback((next: boolean) => {
    setDitLayerwiseOffload(next);
    if (next) {
      setUseFsdpInference(false);
    }
  }, []);

  const handleUseFsdpInferenceChange = React.useCallback((next: boolean) => {
    setUseFsdpInference(next);
    if (next) {
      setDitLayerwiseOffload(false);
    }
  }, []);

  const handleNumGpusChange = React.useCallback((next: number) => {
    setNumGpus(next);
    if (next > 1) {
      // Dropping back to one GPU leaves FSDP alone: single-GPU FSDP is a
      // valid way to reach its CPU offload (docs/inference/offloading.md).
      setUseFsdpInference(true);
      setDitLayerwiseOffload(false);
    }
  }, []);
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
  const [dmdUseVsa, setDmdUseVsa] = React.useState(false);
  const [dmdVsaSparsity, setDmdVsaSparsity] = React.useState(0.8);
  const [dmdDenoisingSteps, setDmdDenoisingSteps] =
    React.useState('1000,757,522');
  const [realScoreGuidanceScale, setRealScoreGuidanceScale] =
    React.useState(3.5);
  const [generatorUpdateInterval, setGeneratorUpdateInterval] =
    React.useState(5);
  const [realScoreModelPath, setRealScoreModelPath] = React.useState('');
  const [fakeScoreModelPath, setFakeScoreModelPath] = React.useState('');
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isLoadingModels, setIsLoadingModels] = React.useState(false);
  const [isLoadingDatasets, setIsLoadingDatasets] = React.useState(false);
  const [modelLoadError, setModelLoadError] = React.useState<string | null>(
    null,
  );
  const [datasetLoadError, setDatasetLoadError] = React.useState<string | null>(
    null,
  );
  const [imageUploadError, setImageUploadError] = React.useState<string | null>(
    null,
  );
  const [submitError, setSubmitError] = React.useState<string | null>(null);
  const imageInputRef = React.useRef<HTMLInputElement>(null);

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
    if (editingJob) {
      // Must not fall through to the defaults below: a partially-seeded
      // form silently edits values the user never saw.
      const f = jobToFormFields(editingJob);
      setModelId(f.modelId);
      setName(f.name);
      setPrompt(f.prompt);
      setNegativePrompt(f.negativePrompt);
      setImagePath(f.imagePath);
      setImageFileName(f.imagePath.split('/').pop() ?? '');
      setLastImagePath(f.lastImagePath);
      setLastImageFileName(f.lastImagePath.split('/').pop() ?? '');
      setReferences(f.references);
      setPromptFields(f.promptFields ?? EMPTY_H3_PROMPT_FIELDS);
      setUseGuidedPrompt(f.promptFields !== null);
      setNumInferenceSteps(f.numInferenceSteps);
      setNumFrames(f.numFrames);
      setHeight(f.height);
      setWidth(f.width);
      setGuidanceScale(f.guidanceScale);
      setGuidanceRescale(f.guidanceRescale);
      setFps(f.fps);
      setSeed(f.seed);
      setNumGpus(f.numGpus);
      setDitCpuOffload(f.ditCpuOffload);
      setDitLayerwiseOffload(f.ditLayerwiseOffload);
      setTextEncoderCpuOffload(f.textEncoderCpuOffload);
      setVaeCpuOffload(f.vaeCpuOffload);
      setImageEncoderCpuOffload(f.imageEncoderCpuOffload);
      setUseFsdpInference(f.useFsdpInference);
      setEnableTorchCompile(f.enableTorchCompile);
      setVsaSparsity(f.vsaSparsity);
      setTpSize(f.tpSize);
      setSpSize(f.spSize);
      setReferenceError(null);
      setModelLoadError(null);
      setImageUploadError(null);
      setLastImageUploadError(null);
      setSubmitError(null);
      return;
    }
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
    setDitLayerwiseOffload(opts.ditLayerwiseOffload ?? false);
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
    setName('');
    setImagePath('');
    setImageFileName('');
    setLastImagePath('');
    setLastImageFileName('');
    setLastImageUploadError(null);
    setReferences([]);
    setReferenceError(null);
    setPromptFields(EMPTY_H3_PROMPT_FIELDS);
    setUseGuidedPrompt(true);
    setSelectedDatasetId('');
    setSelectedValidationDatasetId('');
    setModelLoadError(null);
    setDatasetLoadError(null);
    setImageUploadError(null);
    setSubmitError(null);
    if (workloadType === 'dmd_t2v') {
      setDmdUseVsa(false);
      setDmdVsaSparsity(0.8);
      setDmdDenoisingSteps('1000,757,522');
      setRealScoreGuidanceScale(3.5);
      setGeneratorUpdateInterval(5);
      setRealScoreModelPath('');
      setFakeScoreModelPath('');
    }
  }, [isOpen, workloadType, inferenceWorkload, options, editingJobId]);

  // Load the models available for this workload.
  React.useEffect(() => {
    if (!isOpen) return;
    // Ignore a superseded response so a slow fetch for a previous workload
    // can't overwrite the current workload's model list/selection.
    let stale = false;
    setIsLoadingModels(true);
    setModelLoadError(null);
    getModels(inferenceWorkload)
      .then((list) => {
        if (stale) return;
        setModels(list);
        const ids = list.map((m) => m.id);
        const opts = defaultOptionsStore.get().options;
        const defaultId = getDefaultModelForWorkload(
          opts,
          inferenceWorkload as 't2v' | 'i2v' | 't2i',
        );
        // When editing, the job's own model wins over the workload default --
        // this resolves after the seeding effect, so choosing a default here
        // would silently swap the model out from under the user.
        const editedId = editingJobModelId;
        const chosen =
          editedId && ids.includes(editedId)
            ? editedId
            : ids.includes(defaultId)
              ? defaultId
              : (list[0]?.id ?? '');
        setModelId(chosen);
        if (workloadType === 'dmd_t2v') {
          setRealScoreModelPath(chosen);
          setFakeScoreModelPath(chosen);
        }
      })
      .catch((e) => {
        if (stale) return;
        console.error('Failed to load models:', e);
        setModels([]);
        setModelId('');
        setModelLoadError(
          'Models could not be loaded. Check the Studio API and reopen this form to try again.',
        );
      })
      .finally(() => {
        if (!stale) setIsLoadingModels(false);
      });
    return () => {
      stale = true;
    };
  }, [isOpen, inferenceWorkload, workloadType, editingJobModelId]);

  // Training jobs need a dataset; load the ready datasets when relevant.
  React.useEffect(() => {
    if (isOpen && !isInference) {
      setIsLoadingDatasets(true);
      setDatasetLoadError(null);
      getDatasets()
        .then(setReadyDatasets)
        .catch((error) => {
          console.error('Failed to load datasets:', error);
          setReadyDatasets([]);
          setDatasetLoadError(
            'Datasets could not be loaded. Check the Studio API and reopen this form to try again.',
          );
        })
        .finally(() => setIsLoadingDatasets(false));
    } else {
      setReadyDatasets([]);
      setIsLoadingDatasets(false);
      setDatasetLoadError(null);
    }
  }, [isOpen, isInference]);

  async function handleImageChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) {
      setImagePath('');
      setImageFileName('');
      setImageUploadError(null);
      return;
    }
    setIsUploadingImage(true);
    setImageFileName(file.name);
    setImageUploadError(null);
    try {
      const { path } = await uploadImage(file);
      setImagePath(path);
    } catch (error) {
      console.error('Failed to upload image:', error);
      setImagePath('');
      setImageFileName('');
      setImageUploadError(
        error instanceof Error
          ? `${error.message}. Choose the image again to retry.`
          : 'The image could not be uploaded. Choose it again to retry.',
      );
    } finally {
      setIsUploadingImage(false);
    }
  }

  async function handleLastImageChange(
    e: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = e.target.files?.[0];
    if (!file) {
      setLastImagePath('');
      setLastImageFileName('');
      setLastImageUploadError(null);
      return;
    }
    setIsUploadingLastImage(true);
    setLastImageFileName(file.name);
    setLastImageUploadError(null);
    try {
      const { path } = await uploadImage(file);
      setLastImagePath(path);
    } catch (error) {
      console.error('Failed to upload end image:', error);
      setLastImagePath('');
      setLastImageFileName('');
      setLastImageUploadError(
        error instanceof Error
          ? `${error.message}. Choose the image again to retry.`
          : 'The image could not be uploaded. Choose it again to retry.',
      );
    } finally {
      setIsUploadingLastImage(false);
    }
  }

  async function handleAddReference(
    e: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = e.target.files?.[0];
    e.target.value = '';               // allow re-picking the same file
    if (!file) return;
    setIsUploadingReference(true);
    setReferenceError(null);
    try {
      const { path, media_type } = await uploadMedia(file);
      const next: H3Reference[] = [
        ...references,
        {
          id: `${Date.now()}-${file.name}`,
          source: path,
          media_type,
          fileName: file.name,
        },
      ];
      setReferences(next);
      setReferenceError(validateReferences(next));
    } catch (error) {
      console.error('Failed to upload reference:', error);
      setReferenceError(
        error instanceof Error ? error.message : 'The file could not be uploaded.',
      );
    } finally {
      setIsUploadingReference(false);
    }
  }

  function removeReference(id: string) {
    const next = references.filter((r) => r.id !== id);
    setReferences(next);
    setReferenceError(validateReferences(next));
  }

  function seedPromptFields() {
    setPromptFields({
      ...EMPTY_H3_PROMPT_FIELDS,
      ...referencePromptSeed(references),
    });
    setUseGuidedPrompt(true);
  }

  function setPromptField(section: string, value: string) {
    setPromptFields((prev) => ({ ...prev, [section]: value }));
  }

  // Switching between the guided fields and the raw editor keeps whatever was
  // typed: serialize on the way out, parse back on the way in.
  function toggleGuidedPrompt() {
    if (useGuidedPrompt) {
      if (!isEmptyPromptFields(promptFields)) {
        setPrompt(serializeH3Prompt(promptFields));
      }
      setUseGuidedPrompt(false);
    } else {
      const parsed = parseH3Prompt(prompt);
      if (parsed) setPromptFields(parsed);
      setUseGuidedPrompt(true);
    }
  }

  function clearLastImage() {
    setLastImagePath('');
    setLastImageFileName('');
    setLastImageUploadError(null);
  }

  function clearImage() {
    setImagePath('');
    setImageFileName('');
    setImageUploadError(null);
    if (imageInputRef.current) imageInputRef.current.value = '';
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (isInference && workloadType === 'i2v' && !imagePath && !usingReferences)
      return;
    if (usingReferences && validateReferences(references)) return;
    if (
      usingReferences &&
      useGuidedPrompt &&
      isEmptyPromptFields(promptFields) &&
      !prompt.trim()
    )
      return;
    // Send the dataset id; the backend resolves it to the on-disk media dir.
    const effectiveDataPath = selectedDatasetId ?? '';
    if (!isInference && !selectedDatasetId) return;
    // `lora_t2v` jobs are persisted with a dedicated backend job_type that the
    // front-end JobType enum does not model; cast to keep payload parity.
    const effectiveJobType = (
      workloadType === 'lora_t2v' ? 'lora' : jobType
    ) as JobType;
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const payload: CreateJobRequest = {
        model_id: modelId,
        name: name.trim(),
        prompt:
          usingReferences && useGuidedPrompt && !isEmptyPromptFields(promptFields)
            ? serializeH3Prompt(promptFields)
            : prompt,
        workload_type: workloadType,
        job_type: effectiveJobType,
        ...(isInference
          ? {
              // Ref2VA and the FL2VA keyframes are mutually exclusive:
              // _prepare_ref2va rejects image_path/last_image_path outright
              // when references are present.
              ...(workloadType === 'i2v' && !usingReferences && imagePath
                ? { image_path: imagePath }
                : {}),
              ...(workloadType === 'i2v' &&
              supportsLastImage &&
              !usingReferences &&
              lastImagePath
                ? { last_image_path: lastImagePath }
                : {}),
              ...(workloadType === 'i2v' && supportsLastImage && references.length
                ? {
                    references: references.map((r) => ({
                      source: r.source,
                      media_type: r.media_type,
                    })),
                  }
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
              dit_layerwise_offload: ditLayerwiseOffload,
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
              validation_dataset_file: selectedValidationDatasetId || undefined,
              lora_rank: loraRank,
              ...(workloadType === 'dmd_t2v'
                ? {
                    dmd_use_vsa: dmdUseVsa,
                    dmd_vsa_sparsity: dmdVsaSparsity,
                    dmd_denoising_steps: dmdDenoisingSteps,
                    real_score_guidance_scale: realScoreGuidanceScale,
                    generator_update_interval: generatorUpdateInterval,
                    real_score_model_path: realScoreModelPath || modelId,
                    fake_score_model_path: fakeScoreModelPath || modelId,
                  }
                : {}),
            }),
      };
      if (editingJob) {
        await updateJob(
          editingJob.id,
          payload as unknown as Record<string, unknown>,
        );
      } else {
        await createJob(payload);
      }
      onSuccess();
      onClose();
    } catch (err) {
      console.error('Failed to create job:', err);
      setSubmitError(
        err instanceof Error
          ? `${err.message}. Check the form and Studio API, then try again.`
          : 'The job could not be created. Check the form and Studio API, then try again.',
      );
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
  const title = `${readOnly ? 'View' : editingJob ? 'Edit' : 'New'} ${
    jobType.charAt(0).toUpperCase() + jobType.slice(1)
  } Job${workloadLabel ? ` (${workloadLabel})` : ''}`;

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
          {/* disabled cascades to every control inside; display:contents
              keeps the parent's flex layout. */}
          <fieldset
            disabled={readOnly}
            style={{ display: 'contents' }}
            className="contents"
          >
          <FieldRow htmlFor="modal-name" label="Name (optional)">
            <Input
              id="modal-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Shown on the job card and used for the output filename"
              disabled={isSubmitting}
            />
          </FieldRow>

          <FieldRow htmlFor="modal-modelId" label="Model">
            <NativeSelect
              id="modal-modelId"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              required
              aria-describedby={
                modelLoadError ? 'modal-model-error' : undefined
              }
              aria-invalid={modelLoadError ? true : undefined}
              disabled={isSubmitting || isLoadingModels || !!modelLoadError}
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
            {modelLoadError && (
              <p
                id="modal-model-error"
                role="alert"
                className="text-sm text-destructive"
              >
                {modelLoadError}
              </p>
            )}
          </FieldRow>

          {isInference && workloadType === 'i2v' && (
            <FieldRow htmlFor="modal-image" label="Image">
              <Input
                ref={imageInputRef}
                id="modal-image"
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.bmp"
                onChange={handleImageChange}
                disabled={isSubmitting || isUploadingImage || usingReferences}
                aria-describedby={
                  imageUploadError ? 'modal-image-error' : undefined
                }
                aria-invalid={imageUploadError ? true : undefined}
                required={!usingReferences}
                className="h-auto py-2 file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-secondary file:px-2 file:py-1 file:text-sm file:text-secondary-foreground"
              />
              {imageFileName && (
                <span className="mt-0.5 text-xs text-muted-foreground">
                  {isUploadingImage ? 'Uploading…' : imageFileName} ·{' '}
                  <button
                    type="button"
                    onClick={clearImage}
                    disabled={isSubmitting || isUploadingImage}
                    className="text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Clear
                  </button>
                </span>
              )}
              {imageUploadError && (
                <p
                  id="modal-image-error"
                  role="alert"
                  className="text-sm text-destructive"
                >
                  {imageUploadError}
                </p>
              )}
            </FieldRow>
          )}

          {isInference && workloadType === 'i2v' && supportsLastImage && (
            <FieldRow htmlFor="modal-last-image" label="End Frame (optional)">
              <Input
                id="modal-last-image"
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.bmp"
                onChange={handleLastImageChange}
                disabled={isSubmitting || isUploadingLastImage}
                aria-describedby={
                  lastImageUploadError ? 'modal-last-image-error' : undefined
                }
                aria-invalid={lastImageUploadError ? true : undefined}
                className="h-auto py-2 file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-secondary file:px-2 file:py-1 file:text-sm file:text-secondary-foreground"
              />
              {lastImageFileName && (
                <span className="mt-0.5 text-xs text-muted-foreground">
                  {isUploadingLastImage ? 'Uploading…' : lastImageFileName} ·{' '}
                  <button
                    type="button"
                    onClick={clearLastImage}
                    disabled={isSubmitting || isUploadingLastImage}
                    className="text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Clear
                  </button>
                </span>
              )}
              {lastImageUploadError && (
                <p
                  id="modal-last-image-error"
                  role="alert"
                  className="text-sm text-destructive"
                >
                  {lastImageUploadError}
                </p>
              )}
            </FieldRow>
          )}

          {isInference && workloadType === 'i2v' && supportsLastImage && (
            <FieldRow htmlFor="modal-reference" label="References (Ref2VA)">
              <Input
                id="modal-reference"
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.bmp,.mp4,.mov,.mkv,.webm,.avi,.wav,.mp3,.flac,.m4a,.ogg"
                onChange={handleAddReference}
                disabled={
                  isSubmitting ||
                  isUploadingReference ||
                  references.length >= H3_MAX_REFERENCES
                }
                className="h-auto py-2 file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-secondary file:px-2 file:py-1 file:text-sm file:text-secondary-foreground"
              />
              {isUploadingReference && (
                <span className="mt-0.5 text-xs text-muted-foreground">
                  Uploading…
                </span>
              )}
              {references.length > 0 && (
                <ul className="mt-1 flex list-none flex-col gap-1 p-0">
                  {references.map((reference, index) => (
                    <li
                      key={reference.id}
                      className="flex items-center gap-2 text-xs text-muted-foreground"
                    >
                      <code className="font-mono text-accent-blue">
                        {labelReferences(references)[index]}
                      </code>
                      <span className="truncate">{reference.fileName}</span>
                      <button
                        type="button"
                        onClick={() => removeReference(reference.id)}
                        disabled={isSubmitting}
                        className="ml-auto text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {references.length > 0 && (
                <button
                  type="button"
                  onClick={seedPromptFields}
                  disabled={isSubmitting}
                  className="mt-1 self-start text-xs text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Fill prompt sections from references
                </button>
              )}
              {referenceError && (
                <p role="alert" className="mt-0.5 text-xs text-destructive">
                  {referenceError}
                </p>
              )}
              <span className="mt-0.5 text-xs text-muted-foreground">
                Ref2VA replaces the keyframes: up to 9 images, 3 videos, 3 audio
                (12 total). Audio needs at least one image or video.
              </span>
            </FieldRow>
          )}

          {usingReferences && useGuidedPrompt ? (
            /* Six-section format from the model's reference prompt guide. */
            <>
              {H3_PROMPT_SECTIONS.map((section) => (
                <FieldRow
                  key={section}
                  htmlFor={`modal-prompt-${section}`}
                  label={H3_SECTION_LABELS[section]}
                >
                  <Textarea
                    id={`modal-prompt-${section}`}
                    value={promptFields[section]}
                    onChange={(e) => setPromptField(section, e.target.value)}
                    rows={section === 'detailed_description' ? 5 : 2}
                    placeholder={H3_SECTION_HINTS[section]}
                    disabled={isSubmitting}
                  />
                </FieldRow>
              ))}
              <button
                type="button"
                onClick={toggleGuidedPrompt}
                disabled={isSubmitting}
                className="self-start text-xs text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
              >
                Edit as raw prompt
              </button>
            </>
          ) : (
            <>
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
                  required={!(usingReferences && useGuidedPrompt)}
                  disabled={isSubmitting}
                />
              </FieldRow>
              {usingReferences && (
                <button
                  type="button"
                  onClick={toggleGuidedPrompt}
                  disabled={isSubmitting}
                  className="self-start text-xs text-accent-blue underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Edit as prompt sections
                </button>
              )}
            </>
          )}

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
                    aria-describedby={
                      datasetLoadError ? 'modal-dataset-error' : undefined
                    }
                    aria-invalid={datasetLoadError ? true : undefined}
                    disabled={
                      isSubmitting || isLoadingDatasets || !!datasetLoadError
                    }
                  >
                    <option value="" disabled>
                      {isLoadingDatasets
                        ? 'Loading datasets…'
                        : datasetLoadError
                          ? 'Datasets unavailable'
                          : readyDatasets.length === 0
                            ? 'No datasets (add in Datasets tab)'
                            : 'Select a dataset…'}
                    </option>
                    {readyDatasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                  </NativeSelect>
                  {datasetLoadError && (
                    <p
                      id="modal-dataset-error"
                      role="alert"
                      className="text-sm text-destructive"
                    >
                      {datasetLoadError}
                    </p>
                  )}
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
                    disabled={
                      isSubmitting || isLoadingDatasets || !!datasetLoadError
                    }
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
                <summary className="mb-2 cursor-pointer select-none text-sm font-medium text-accent-blue">
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
                  <NumberRow
                    id="modal-learning-rate"
                    label="Learning Rate"
                    step="1e-6"
                    min={1e-6}
                    max={1}
                    value={learningRate}
                    onChange={setLearningRate}
                    disabled={isSubmitting}
                  />
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
                          format={(v) => v.toFixed(2)}
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
                        id="modal-real-score-guidance-scale"
                        label="Real Score Guidance Scale"
                        min={1}
                        max={10}
                        step={0.1}
                        value={realScoreGuidanceScale}
                        onChange={setRealScoreGuidanceScale}
                        disabled={isSubmitting}
                        format={(v) => v.toFixed(1)}
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
                      {(
                        [
                          {
                            id: 'modal-real-score-model',
                            label: 'Real Score Model',
                            value: realScoreModelPath,
                            onChange: setRealScoreModelPath,
                          },
                          {
                            id: 'modal-fake-score-model',
                            label: 'Fake Score Model',
                            value: fakeScoreModelPath,
                            onChange: setFakeScoreModelPath,
                          },
                        ] as const
                      ).map((select) => (
                        <FieldRow
                          key={select.id}
                          htmlFor={select.id}
                          label={select.label}
                        >
                          <NativeSelect
                            id={select.id}
                            value={select.value}
                            onChange={(e) => select.onChange(e.target.value)}
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
                      ))}
                    </>
                  )}
                </div>
              </details>
            </>
          )}

          {isInference && (
            <details>
              <summary className="mb-2 cursor-pointer select-none text-sm font-medium text-accent-blue">
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
                  format={(v) => v.toFixed(2)}
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
                  format={(v) => v.toFixed(1)}
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
                  format={(v) => v.toFixed(2)}
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
                  format={(v) => (v === -1 ? 'Auto' : String(v))}
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
                  format={(v) => (v === -1 ? 'Auto' : String(v))}
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
                  id="modal-dit-layerwise-offload"
                  label="DiT Layerwise Offload"
                  checked={ditLayerwiseOffload}
                  onChange={handleDitLayerwiseOffloadChange}
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
                  onChange={handleUseFsdpInferenceChange}
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
                  onChange={handleNumGpusChange}
                  disabled={isSubmitting}
                />
                <NumberRow
                  id="modal-seed"
                  label="Seed"
                  min={0}
                  value={seed}
                  onChange={setSeed}
                  disabled={isSubmitting}
                />
              </div>
            </details>
          )}

          </fieldset>

          <div className="flex flex-col items-start gap-2">
            {submitError && (
              <p role="alert" className="text-sm text-destructive">
                {submitError}
              </p>
            )}
            <Button
              type="submit"
              hidden={readOnly}
              disabled={
                readOnly ||
                isSubmitting ||
                isUploadingImage ||
                !!modelLoadError ||
                !!datasetLoadError
              }
            >
              {isSubmitting
                ? editingJob
                  ? 'Saving…'
                  : 'Creating…'
                : editingJob
                  ? 'Save Changes'
                  : 'Create Job'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
