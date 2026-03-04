'use client';

import { createJob, getModels, uploadImage, type Model } from "@/lib/api";
import { getDefaultModelForWorkload } from "@/lib/defaultOptions";
import { useDefaultOptions } from "@/contexts/DefaultOptionsContext";
import { useEffect, useRef, useState } from "react";
import type { WorkloadType } from "./CreateJobButton";
import modalStyles from "./styles/Modal.module.css";
import formStyles from "./styles/Form.module.css";
import cardStyles from "./styles/Card.module.css";
import buttonStyles from "./styles/Button.module.css";
import Toggle from "./Toggle";
import Slider from "./Slider";

interface CreateJobModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  workloadType?: WorkloadType;
}

export default function CreateJobModal({
  isOpen,
  onClose,
  onSuccess,
  workloadType = "t2v",
}: CreateJobModalProps) {
  const { options: defaultOptions } = useDefaultOptions();
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [imagePath, setImagePath] = useState("");
  const [imageFileName, setImageFileName] = useState("");
  const [isUploadingImage, setIsUploadingImage] = useState(false);
  const [negativePrompt, setNegativePrompt] = useState("");
  const [numInferenceSteps, setNumInferenceSteps] = useState(defaultOptions.numInferenceSteps);
  const [numFrames, setNumFrames] = useState(defaultOptions.numFrames);
  const [height, setHeight] = useState(defaultOptions.height);
  const [width, setWidth] = useState(defaultOptions.width);
  const [guidanceScale, setGuidanceScale] = useState(defaultOptions.guidanceScale);
  const [guidanceRescale, setGuidanceRescale] = useState(defaultOptions.guidanceRescale);
  const [fps, setFps] = useState(defaultOptions.fps);
  const [seed, setSeed] = useState(defaultOptions.seed);
  const [numGpus, setNumGpus] = useState(defaultOptions.numGpus);
  const [ditCpuOffload, setDitCpuOffload] = useState<boolean>(defaultOptions.ditCpuOffload);
  const [textEncoderCpuOffload, setTextEncoderCpuOffload] = useState<boolean>(defaultOptions.textEncoderCpuOffload);
  const [vaeCpuOffload, setVaeCpuOffload] = useState<boolean>(defaultOptions.vaeCpuOffload);
  const [imageEncoderCpuOffload, setImageEncoderCpuOffload] = useState<boolean>(defaultOptions.imageEncoderCpuOffload);
  const [useFsdpInference, setUseFsdpInference] = useState<boolean>(defaultOptions.useFsdpInference);
  const [enableTorchCompile, setEnableTorchCompile] = useState<boolean>(defaultOptions.enableTorchCompile);
  const [vsaSparsity, setVsaSparsity] = useState<number>(defaultOptions.vsaSparsity);
  const [tpSize, setTpSize] = useState<number>(defaultOptions.tpSize);
  const [spSize, setSpSize] = useState<number>(defaultOptions.spSize);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const imageInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isSubmitting) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen, isSubmitting, onClose]);

  useEffect(() => {
    if (isOpen) {
      setIsLoadingModels(true);
      getModels(workloadType)
        .then((fetchedModels) => {
          setModels(fetchedModels);
          const ids = fetchedModels.map((m) => m.id);
          const defaultId = getDefaultModelForWorkload(defaultOptions, workloadType);
          setModelId(
            ids.includes(defaultId) ? defaultId : fetchedModels[0]?.id ?? ""
          );
        })
        .catch((error) => {
          console.error("Failed to load models:", error);
        })
        .finally(() => {
          setIsLoadingModels(false);
        });
    } else {
      setModels([]);
    }
  }, [
    isOpen,
    workloadType,
    defaultOptions.defaultModelIdT2v,
    defaultOptions.defaultModelIdI2v,
    defaultOptions.defaultModelIdT2i,
  ]);

  useEffect(() => {
    if (isOpen) {
      setModelId(getDefaultModelForWorkload(defaultOptions, workloadType));
      setImagePath("");
      setImageFileName("");
      setNumInferenceSteps(defaultOptions.numInferenceSteps);
      setNumFrames(
        workloadType === "t2i" ? 1 : defaultOptions.numFrames
      );
      setHeight(defaultOptions.height);
      setWidth(defaultOptions.width);
      setGuidanceScale(defaultOptions.guidanceScale);
      setGuidanceRescale(defaultOptions.guidanceRescale);
      setFps(defaultOptions.fps);
      setSeed(defaultOptions.seed);
      setNumGpus(defaultOptions.numGpus);
      setDitCpuOffload(defaultOptions.ditCpuOffload);
      setTextEncoderCpuOffload(defaultOptions.textEncoderCpuOffload);
      setVaeCpuOffload(defaultOptions.vaeCpuOffload);
      setImageEncoderCpuOffload(defaultOptions.imageEncoderCpuOffload);
      setUseFsdpInference(defaultOptions.useFsdpInference);
      setEnableTorchCompile(defaultOptions.enableTorchCompile);
      setVsaSparsity(defaultOptions.vsaSparsity);
      setTpSize(defaultOptions.tpSize);
      setSpSize(defaultOptions.spSize);
    }
  }, [isOpen, defaultOptions, workloadType]);

  const handleImageChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      setImagePath("");
      setImageFileName("");
      return;
    }
    setIsUploadingImage(true);
    setImageFileName(file.name);
    try {
      const { path } = await uploadImage(file);
      setImagePath(path);
    } catch (err) {
      console.error("Image upload failed:", err);
      setImagePath("");
      setImageFileName("");
    } finally {
      setIsUploadingImage(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (workloadType === "i2v" && !imagePath) {
      return;
    }
    setIsSubmitting(true);
    try {
      await createJob({
        model_id: modelId,
        prompt,
        workload_type: workloadType,
        ...(workloadType === "i2v" && imagePath
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
      });
      setModelId(getDefaultModelForWorkload(defaultOptions, workloadType));
      setPrompt("");
      setImagePath("");
      setImageFileName("");
      setNegativePrompt("");
      setNumInferenceSteps(defaultOptions.numInferenceSteps);
      setNumFrames(
        workloadType === "t2i" ? 1 : defaultOptions.numFrames
      );
      setHeight(defaultOptions.height);
      setWidth(defaultOptions.width);
      setGuidanceScale(defaultOptions.guidanceScale);
      setGuidanceRescale(defaultOptions.guidanceRescale);
      setFps(defaultOptions.fps);
      setSeed(defaultOptions.seed);
      setNumGpus(defaultOptions.numGpus);
      setDitCpuOffload(defaultOptions.ditCpuOffload);
      setTextEncoderCpuOffload(defaultOptions.textEncoderCpuOffload);
      setVaeCpuOffload(defaultOptions.vaeCpuOffload);
      setImageEncoderCpuOffload(defaultOptions.imageEncoderCpuOffload);
      setUseFsdpInference(defaultOptions.useFsdpInference);
      setEnableTorchCompile(defaultOptions.enableTorchCompile);
      setVsaSparsity(defaultOptions.vsaSparsity);
      setTpSize(defaultOptions.tpSize);
      setSpSize(defaultOptions.spSize);
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create job:", error);
      // You could add error handling/toast here
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      setModelId(getDefaultModelForWorkload(defaultOptions, workloadType));
      setPrompt("");
      setImagePath("");
      setImageFileName("");
      setNegativePrompt("");
      setNumInferenceSteps(defaultOptions.numInferenceSteps);
      setNumFrames(defaultOptions.numFrames);
      setHeight(defaultOptions.height);
      setWidth(defaultOptions.width);
      setGuidanceScale(defaultOptions.guidanceScale);
      setGuidanceRescale(defaultOptions.guidanceRescale);
      setFps(defaultOptions.fps);
      setSeed(defaultOptions.seed);
      setNumGpus(defaultOptions.numGpus);
      setDitCpuOffload(defaultOptions.ditCpuOffload);
      setTextEncoderCpuOffload(defaultOptions.textEncoderCpuOffload);
      setVaeCpuOffload(defaultOptions.vaeCpuOffload);
      setImageEncoderCpuOffload(defaultOptions.imageEncoderCpuOffload);
      setUseFsdpInference(defaultOptions.useFsdpInference);
      setEnableTorchCompile(defaultOptions.enableTorchCompile);
      setVsaSparsity(defaultOptions.vsaSparsity);
      setTpSize(defaultOptions.tpSize);
      setSpSize(defaultOptions.spSize);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className={modalStyles.modal} data-modal>
      <div className={modalStyles.modalBackdrop} onClick={handleClose} />
      <div className={`${modalStyles.modalContent} ${modalStyles.modalForm}`}>
        <button
          className={modalStyles.modalClose}
          onClick={handleClose}
          disabled={isSubmitting}
          aria-label="Close"
        >
          ×
        </button>
        <div className={cardStyles.card} style={{ margin: 0, border: 'none' }}>
          <h2>
            New Job
            {workloadType === "t2v" && " (T2V)"}
            {workloadType === "i2v" && " (I2V)"}
            {workloadType === "t2i" && " (T2I)"}
          </h2>
          <form onSubmit={handleSubmit} autoComplete="off">
            <div className={formStyles.formRow}>
              <label htmlFor="modal-modelId">Model</label>
              <select
                name="modelId"
                id="modal-modelId"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
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
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}  ({model.id})
                  </option>
                ))}
              </select>
            </div>
            {workloadType === "i2v" && (
              <div className={formStyles.formRow}>
                <label htmlFor="modal-image">Image</label>
                <input
                  ref={imageInputRef}
                  id="modal-image"
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp,.bmp"
                  onChange={handleImageChange}
                  disabled={isSubmitting || isUploadingImage}
                  required
                />
                {imageFileName && (
                  <span className={formStyles.helperText}>
                    {isUploadingImage ? "Uploading…" : imageFileName}
                    {" · "}
                    <button
                      type="button"
                      className={formStyles.clearLink}
                      onClick={() => {
                        setImagePath("");
                        setImageFileName("");
                        if (imageInputRef.current) {
                          imageInputRef.current.value = "";
                        }
                      }}
                      disabled={isSubmitting || isUploadingImage}
                    >
                      Clear
                    </button>
                  </span>
                )}
              </div>
            )}
            <div className={formStyles.formRow}>
              <label htmlFor="modal-prompt">Prompt</label>
              <textarea
                name="prompt"
                id="modal-prompt"
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A curious raccoon peers through a vibrant field of yellow sunflowers…"
                required
                disabled={isSubmitting}
              />
            </div>
            <div className={formStyles.formRow}>
              <label htmlFor="modal-negative-prompt">Negative Prompt</label>
              <textarea
                name="negativePrompt"
                id="modal-negative-prompt"
                rows={2}
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="Optional: things to avoid in the output…"
                disabled={isSubmitting}
              />
            </div>

            {/* Advanced settings (collapsed by default) */}
            <details className={formStyles.advancedSettings}>
              <summary>Advanced Settings</summary>
              <div className={formStyles.settingsGrid}>
                {workloadType !== "t2i" && (
                  <div className={formStyles.formRow}>
                    <label htmlFor="modal-num-frames">Frames</label>
                    <Slider
                      id="modal-num-frames"
                      min={1}
                      max={500}
                      step={1}
                      value={numFrames}
                      onChange={setNumFrames}
                      disabled={isSubmitting}
                    />
                  </div>
                )}
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-height">Height</label>
                  <Slider
                    id="modal-height"
                    min={64}
                    max={1080}
                    step={16}
                    value={height}
                    onChange={setHeight}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-width">Width</label>
                  <Slider
                    id="modal-width"
                    min={64}
                    max={1920}
                    step={16}
                    value={width}
                    onChange={setWidth}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-num-steps">Inference Steps</label>
                  <Slider
                    id="modal-num-steps"
                    min={1}
                    max={200}
                    step={1}
                    value={numInferenceSteps}
                    onChange={setNumInferenceSteps}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label
                    htmlFor="modal-vsa-sparsity"
                    title="Video Sparse Attention sparsity (0–1, higher = sparser)"
                  >
                    VSA Sparsity
                  </label>
                  <Slider
                    id="modal-vsa-sparsity"
                    min={0}
                    max={1}
                    step={0.05}
                    value={vsaSparsity}
                    onChange={setVsaSparsity}
                    disabled={isSubmitting}
                    formatValue={(v) => v.toFixed(2)}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-guidance">Guidance Scale</label>
                  <Slider
                    id="modal-guidance"
                    min={0}
                    max={20}
                    step={0.1}
                    value={guidanceScale}
                    onChange={setGuidanceScale}
                    disabled={isSubmitting}
                    formatValue={(v) => v.toFixed(1)}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label
                    htmlFor="modal-guidance-rescale"
                    title="Guidance rescale factor (0 = disabled)"
                  >
                    Guidance Rescale
                  </label>
                  <Slider
                    id="modal-guidance-rescale"
                    min={0}
                    max={1}
                    step={0.05}
                    value={guidanceRescale}
                    onChange={setGuidanceRescale}
                    disabled={isSubmitting}
                    formatValue={(v) => v.toFixed(2)}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label
                    htmlFor="modal-tp-size"
                    title="Tensor parallelism size (-1 = auto)"
                  >
                    TP Size
                  </label>
                  <Slider
                    id="modal-tp-size"
                    min={-1}
                    max={8}
                    step={1}
                    value={tpSize}
                    onChange={setTpSize}
                    disabled={isSubmitting}
                    formatValue={(v) => (v === -1 ? "Auto" : String(v))}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label
                    htmlFor="modal-sp-size"
                    title="Sequence parallelism size (-1 = auto)"
                  >
                    SP Size
                  </label>
                  <Slider
                    id="modal-sp-size"
                    min={-1}
                    max={8}
                    step={1}
                    value={spSize}
                    onChange={setSpSize}
                    disabled={isSubmitting}
                    formatValue={(v) => (v === -1 ? "Auto" : String(v))}
                  />
                </div>
                {workloadType !== "t2i" && (
                  <div className={formStyles.formRow}>
                    <label htmlFor="modal-fps">FPS</label>
                    <Slider
                      id="modal-fps"
                      min={1}
                      max={60}
                      step={1}
                      value={fps}
                      onChange={setFps}
                      disabled={isSubmitting}
                    />
                  </div>
                )}
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-dit-cpu-offload">DiT CPU Offload</label>
                  <Toggle
                    id="modal-dit-cpu-offload"
                    checked={ditCpuOffload}
                    onChange={setDitCpuOffload}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-text-encoder-cpu-offload">
                    Text Encoder CPU Offload
                  </label>
                  <Toggle
                    id="modal-text-encoder-cpu-offload"
                    checked={textEncoderCpuOffload}
                    onChange={setTextEncoderCpuOffload}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-use-fsdp-inference">Use FSDP Inference</label>
                  <Toggle
                    id="modal-use-fsdp-inference"
                    checked={useFsdpInference}
                    onChange={setUseFsdpInference}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-vae-cpu-offload">VAE CPU Offload</label>
                  <Toggle
                    id="modal-vae-cpu-offload"
                    checked={vaeCpuOffload}
                    onChange={setVaeCpuOffload}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-image-encoder-cpu-offload">
                    Image Encoder CPU Offload
                  </label>
                  <Toggle
                    id="modal-image-encoder-cpu-offload"
                    checked={imageEncoderCpuOffload}
                    onChange={setImageEncoderCpuOffload}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-enable-torch-compile">Torch Compile</label>
                  <Toggle
                    id="modal-enable-torch-compile"
                    checked={enableTorchCompile}
                    onChange={setEnableTorchCompile}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-num-gpus">GPUs</label>
                  <Slider
                    id="modal-num-gpus"
                    min={1}
                    max={8}
                    step={1}
                    value={numGpus}
                    onChange={setNumGpus}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-seed">Seed</label>
                  <input
                    id="modal-seed"
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value, 10))}
                    min={0}
                    disabled={isSubmitting}
                  />
                </div>
              </div>
            </details>

            <button type="submit" className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`} disabled={isSubmitting}>
              {isSubmitting ? "Creating..." : "Create Job"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
