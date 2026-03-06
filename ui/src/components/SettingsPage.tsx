'use client';

import { useEffect, useState } from "react";
import { useDefaultOptions } from "@/contexts/DefaultOptionsContext";
import { useHeaderActions } from "@/contexts/ActiveTabContext";
import { getModels, type Model } from "@/lib/api";
import type { WorkloadType } from "@/lib/defaultOptions";
import cardStyles from "@styles/Card.module.css";
import formStyles from "@styles/Form.module.css";
import layoutStyles from "@/app/Layout.module.css";
import buttonStyles from "@styles/Button.module.css";
import Toggle from "@/components/Toggle";
import Slider from "@/components/Slider";

export default function SettingsPage() {
  useHeaderActions([]);
  const { options, updateOption, resetToDefaults } = useDefaultOptions();
  const [modelsT2v, setModelsT2v] = useState<Model[]>([]);
  const [modelsI2v, setModelsI2v] = useState<Model[]>([]);
  const [modelsT2i, setModelsT2i] = useState<Model[]>([]);

  useEffect(() => {
    Promise.all([
      getModels("t2v"),
      getModels("i2v"),
      getModels("t2i"),
    ])
      .then(([t2v, i2v, t2i]) => {
        setModelsT2v(t2v);
        setModelsI2v(i2v);
        setModelsT2i(t2i);
      })
      .catch((error) => console.error("Failed to load models:", error));
  }, []);

  const DefaultModelSelect = ({
    workloadType,
    label,
    models,
    value,
    onUpdate,
  }: {
    workloadType: WorkloadType;
    label: string;
    models: Model[];
    value: string;
    onUpdate: (modelId: string) => void;
  }) => (
    <div className={formStyles.formRow}>
      <label htmlFor={`settings-default-model-${workloadType}`}>{label}</label>
      <select
        id={`settings-default-model-${workloadType}`}
        value={value}
        onChange={(e) => onUpdate(e.target.value)}
      >
        <option value="">None (select when creating job)</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label} ({model.id})
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <main className={layoutStyles.main}>
      <section className={cardStyles.card}>
        <h2>Behavior</h2>
        <div className={formStyles.row}>
          <label htmlFor="settings-auto-start-job">
            Auto Start Job on Create
          </label>
          <Toggle
            id="settings-auto-start-job"
            checked={options.autoStartJob}
            onChange={(v) => updateOption("autoStartJob", v)}
          />
        </div>
        <hr style={{ margin: "1rem 0", border: "none", borderTop: "1px solid var(--border)" }} />
        <h2>Paths</h2>
        <div className={formStyles.row}>
          <label htmlFor="settings-dataset-upload-path">
            Dataset Upload Path
          </label>
          <input
            id="settings-dataset-upload-path"
            type="text"
            value={options.datasetUploadPath ?? ""}
            onChange={(e) =>
              updateOption("datasetUploadPath", e.target.value)
            }
            placeholder="outputs/ui_data/uploads/datasets"
            style={{ fontFamily: "monospace", fontSize: "0.9rem" }}
          />
        </div>
        <hr style={{ margin: "1rem 0", border: "none", borderTop: "1px solid var(--border)" }} />
        <div className={layoutStyles.sectionHeader}>
          <h2>Default Options</h2>
          <button
            type="button"
            className={`${buttonStyles.btn} ${buttonStyles.btnSmall}`}
            onClick={resetToDefaults}
          >
            Reset to Defaults
          </button>
        </div>
        <p
          style={{
            color: "var(--text-dim)",
            fontSize: "0.9rem",
            marginBottom: "1rem",
          }}
        >
          These values are used as defaults when creating new jobs. Adjust them
          below to match your typical workflow.
        </p>
        <div className={formStyles.settingsGrid}>
          <DefaultModelSelect
            workloadType="t2v"
            label="Default Model (T2V)"
            models={modelsT2v}
            value={options.defaultModelIdT2v}
            onUpdate={(v) => updateOption("defaultModelIdT2v", v)}
          />
          <DefaultModelSelect
            workloadType="i2v"
            label="Default Model (I2V)"
            models={modelsI2v}
            value={options.defaultModelIdI2v}
            onUpdate={(v) => updateOption("defaultModelIdI2v", v)}
          />
          <DefaultModelSelect
            workloadType="t2i"
            label="Default Model (T2I)"
            models={modelsT2i}
            value={options.defaultModelIdT2i}
            onUpdate={(v) => updateOption("defaultModelIdT2i", v)}
          />
          <div className={formStyles.formRow}>
            <label htmlFor="settings-num-frames">Frames</label>
            <Slider
              id="settings-num-frames"
              min={1}
              max={500}
              step={1}
              value={options.numFrames}
              onChange={(v) => updateOption("numFrames", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-height">Height</label>
            <Slider
              id="settings-height"
              min={64}
              max={1080}
              step={16}
              value={options.height}
              onChange={(v) => updateOption("height", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-width">Width</label>
            <Slider
              id="settings-width"
              min={64}
              max={1920}
              step={16}
              value={options.width}
              onChange={(v) => updateOption("width", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-num-steps">Inference Steps</label>
            <Slider
              id="settings-num-steps"
              min={1}
              max={200}
              step={1}
              value={options.numInferenceSteps}
              onChange={(v) => updateOption("numInferenceSteps", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-vsa-sparsity"
              title="Video Sparse Attention sparsity (0–1, higher = sparser)"
            >
              VSA Sparsity
            </label>
            <Slider
              id="settings-vsa-sparsity"
              min={0}
              max={1}
              step={0.05}
              value={options.vsaSparsity}
              onChange={(v) => updateOption("vsaSparsity", v)}
              formatValue={(v) => v.toFixed(2)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-guidance">Guidance Scale</label>
            <Slider
              id="settings-guidance"
              min={0}
              max={20}
              step={0.1}
              value={options.guidanceScale}
              onChange={(v) => updateOption("guidanceScale", v)}
              formatValue={(v) => v.toFixed(1)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-guidance-rescale"
              title="Guidance rescale factor (0 = disabled)"
            >
              Guidance Rescale
            </label>
            <Slider
              id="settings-guidance-rescale"
              min={0}
              max={1}
              step={0.05}
              value={options.guidanceRescale ?? 0}
              onChange={(v) => updateOption("guidanceRescale", v)}
              formatValue={(v) => v.toFixed(2)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-tp-size"
              title="Tensor parallelism size (-1 = auto)"
            >
              TP Size
            </label>
            <Slider
              id="settings-tp-size"
              min={-1}
              max={8}
              step={1}
              value={options.tpSize}
              onChange={(v) => updateOption("tpSize", v)}
              formatValue={(v) => (v === -1 ? "Auto" : String(v))}
            />
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-sp-size"
              title="Sequence parallelism size (-1 = auto)"
            >
              SP Size
            </label>
            <Slider
              id="settings-sp-size"
              min={-1}
              max={8}
              step={1}
              value={options.spSize}
              onChange={(v) => updateOption("spSize", v)}
              formatValue={(v) => (v === -1 ? "Auto" : String(v))}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-fps">FPS</label>
            <Slider
              id="settings-fps"
              min={1}
              max={60}
              step={1}
              value={options.fps ?? 24}
              onChange={(v) => updateOption("fps", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-dit-cpu-offload">DiT CPU Offload</label>
            <Toggle
              id="settings-dit-cpu-offload"
              checked={options.ditCpuOffload}
              onChange={(v) => updateOption("ditCpuOffload", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-text-encoder-cpu-offload">
              Text Encoder CPU Offload
            </label>
            <Toggle
              id="settings-text-encoder-cpu-offload"
              checked={options.textEncoderCpuOffload}
              onChange={(v) => updateOption("textEncoderCpuOffload", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-use-fsdp-inference">
              Use FSDP Inference
            </label>
            <Toggle
              id="settings-use-fsdp-inference"
              checked={options.useFsdpInference}
              onChange={(v) => updateOption("useFsdpInference", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-vae-cpu-offload">VAE CPU Offload</label>
            <Toggle
              id="settings-vae-cpu-offload"
              checked={options.vaeCpuOffload}
              onChange={(v) => updateOption("vaeCpuOffload", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-image-encoder-cpu-offload">
              Image Encoder CPU Offload
            </label>
            <Toggle
              id="settings-image-encoder-cpu-offload"
              checked={options.imageEncoderCpuOffload}
              onChange={(v) => updateOption("imageEncoderCpuOffload", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-enable-torch-compile">
              Torch Compile
            </label>
            <Toggle
              id="settings-enable-torch-compile"
              checked={options.enableTorchCompile}
              onChange={(v) => updateOption("enableTorchCompile", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-num-gpus">GPUs</label>
            <Slider
              id="settings-num-gpus"
              min={1}
              max={8}
              step={1}
              value={options.numGpus}
              onChange={(v) => updateOption("numGpus", v)}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-seed">Seed</label>
            <input
              id="settings-seed"
              type="number"
              value={options.seed}
              onChange={(e) =>
                updateOption("seed", parseInt(e.target.value, 10))
              }
              min={0}
            />
          </div>
        </div>
      </section>
    </main>
  );
}
