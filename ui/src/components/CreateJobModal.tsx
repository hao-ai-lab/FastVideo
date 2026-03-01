'use client';

import { createJob, getModels, type Model } from "@/lib/api";
import { useEffect, useState } from "react";
import modalStyles from "./styles/Modal.module.css";
import formStyles from "./styles/Form.module.css";
import cardStyles from "./styles/Card.module.css";
import buttonStyles from "./styles/Button.module.css";

interface CreateJobModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function CreateJobModal({ isOpen, onClose, onSuccess }: CreateJobModalProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [numInferenceSteps, setNumInferenceSteps] = useState(50);
  const [numFrames, setNumFrames] = useState(81);
  const [height, setHeight] = useState(480);
  const [width, setWidth] = useState(832);
  const [guidanceScale, setGuidanceScale] = useState(5.0);
  const [seed, setSeed] = useState(1024);
  const [numGpus, setNumGpus] = useState(1);
  const [ditCpuOffload, setDitCpuOffload] = useState<boolean>(false);
  const [textEncoderCpuOffload, setTextEncoderCpuOffload] = useState<boolean>(false);
  const [useFsdpInference, setUseFsdpInference] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

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
    if (isOpen && models.length === 0) {
      setIsLoadingModels(true);
      getModels()
        .then(setModels)
        .catch((error) => {
          console.error("Failed to load models:", error);
        })
        .finally(() => {
          setIsLoadingModels(false);
        });
    }
  }, [isOpen, models.length]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await createJob({
        model_id: modelId,
        prompt,
        num_inference_steps: numInferenceSteps,
        num_frames: numFrames,
        height,
        width,
        guidance_scale: guidanceScale,
        seed,
        num_gpus: numGpus,
        dit_cpu_offload: ditCpuOffload,
        text_encoder_cpu_offload: textEncoderCpuOffload,
        use_fsdp_inference: useFsdpInference,
      });
      setModelId("");
      setPrompt("");
      setNumInferenceSteps(50);
      setNumFrames(81);
      setHeight(480);
      setWidth(832);
      setGuidanceScale(5.0);
      setSeed(1024);
      setNumGpus(1);
      setDitCpuOffload(false);
      setTextEncoderCpuOffload(false);
      setUseFsdpInference(false);
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
      setModelId("");
      setPrompt("");
      setNumInferenceSteps(50);
      setNumFrames(81);
      setHeight(480);
      setWidth(832);
      setGuidanceScale(5.0);
      setSeed(1024);
      setNumGpus(1);
      setDitCpuOffload(false);
      setTextEncoderCpuOffload(false);
      setUseFsdpInference(false);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className={modalStyles.modal}>
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
          <h2>New Job</h2>
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
                  {isLoadingModels ? "Loading models…" : "Select a model…"}
                </option>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}  ({model.id})
                  </option>
                ))}
              </select>
            </div>
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

            {/* Advanced settings (collapsed by default) */}
            <details className={formStyles.advancedSettings}>
              <summary>Advanced Settings</summary>
              <div className={formStyles.settingsGrid}>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-num-steps">Inference Steps</label>
                  <input
                    id="modal-num-steps"
                    type="number"
                    value={numInferenceSteps}
                    onChange={(e) => setNumInferenceSteps(parseInt(e.target.value, 10))}
                    min={1}
                    max={200}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-num-frames">Frames</label>
                  <input
                    id="modal-num-frames"
                    type="number"
                    value={numFrames}
                    onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
                    min={1}
                    max={500}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-height">Height</label>
                  <input
                    id="modal-height"
                    type="number"
                    value={height}
                    onChange={(e) => setHeight(parseInt(e.target.value, 10))}
                    min={64}
                    step={16}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-width">Width</label>
                  <input
                    id="modal-width"
                    type="number"
                    value={width}
                    onChange={(e) => setWidth(parseInt(e.target.value, 10))}
                    min={64}
                    step={16}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-guidance">Guidance Scale</label>
                  <input
                    id="modal-guidance"
                    type="number"
                    value={guidanceScale}
                    onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                    min={0}
                    step={0.1}
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
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-num-gpus">GPUs</label>
                  <input
                    id="modal-num-gpus"
                    type="number"
                    value={numGpus}
                    onChange={(e) => setNumGpus(parseInt(e.target.value, 10))}
                    min={1}
                    max={8}
                    disabled={isSubmitting}
                  />
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-dit-cpu-offload">DiT CPU Offload</label>
                  <select
                    id="modal-dit-cpu-offload"
                    value={ditCpuOffload ? 'enabled' : 'disabled'}
                    onChange={(e) => setDitCpuOffload(e.target.value === 'enabled')}
                    disabled={isSubmitting}
                  >
                    <option value="disabled">Disabled</option>
                    <option value="enabled">Enabled</option>
                  </select>
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-text-encoder-cpu-offload">Text Encoder CPU Offload</label>
                  <select
                    id="modal-text-encoder-cpu-offload"
                    value={textEncoderCpuOffload ? 'enabled' : 'disabled'}
                    onChange={(e) => setTextEncoderCpuOffload(e.target.value === 'enabled')}
                    disabled={isSubmitting}
                  >
                    <option value="disabled">Disabled</option>
                    <option value="enabled">Enabled</option>
                  </select>
                </div>
                <div className={formStyles.formRow}>
                  <label htmlFor="modal-use-fsdp-inference">Use FSDP Inference</label>
                  <select
                    id="modal-use-fsdp-inference"
                    value={useFsdpInference ? 'enabled' : 'disabled'}
                    onChange={(e) => setUseFsdpInference(e.target.value === 'enabled')}
                    disabled={isSubmitting}
                  >
                    <option value="disabled">Disabled</option>
                    <option value="enabled">Enabled</option>
                  </select>
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
