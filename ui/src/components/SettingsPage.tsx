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
            <label htmlFor="settings-num-steps">Inference Steps</label>
            <input
              id="settings-num-steps"
              type="number"
              value={options.numInferenceSteps}
              onChange={(e) =>
                updateOption("numInferenceSteps", parseInt(e.target.value, 10))
              }
              min={1}
              max={200}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-num-frames">Frames</label>
            <input
              id="settings-num-frames"
              type="number"
              value={options.numFrames}
              onChange={(e) =>
                updateOption("numFrames", parseInt(e.target.value, 10))
              }
              min={1}
              max={500}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-height">Height</label>
            <input
              id="settings-height"
              type="number"
              value={options.height}
              onChange={(e) =>
                updateOption("height", parseInt(e.target.value, 10))
              }
              min={64}
              step={16}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-width">Width</label>
            <input
              id="settings-width"
              type="number"
              value={options.width}
              onChange={(e) =>
                updateOption("width", parseInt(e.target.value, 10))
              }
              min={64}
              step={16}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-guidance">Guidance Scale</label>
            <input
              id="settings-guidance"
              type="number"
              value={options.guidanceScale}
              onChange={(e) =>
                updateOption("guidanceScale", parseFloat(e.target.value))
              }
              min={0}
              step={0.1}
            />
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-guidance-rescale"
              title="Guidance rescale factor (0 = disabled)"
            >
              Guidance Rescale
            </label>
            <input
              id="settings-guidance-rescale"
              type="number"
              value={options.guidanceRescale ?? 0}
              onChange={(e) =>
                updateOption("guidanceRescale", parseFloat(e.target.value))
              }
              min={0}
              max={1}
              step={0.05}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-fps">FPS</label>
            <input
              id="settings-fps"
              type="number"
              value={options.fps ?? 24}
              onChange={(e) =>
                updateOption("fps", parseInt(e.target.value, 10))
              }
              min={1}
              max={60}
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
          <div className={formStyles.formRow}>
            <label htmlFor="settings-num-gpus">GPUs</label>
            <input
              id="settings-num-gpus"
              type="number"
              value={options.numGpus}
              onChange={(e) =>
                updateOption("numGpus", parseInt(e.target.value, 10))
              }
              min={1}
              max={8}
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-dit-cpu-offload">DiT CPU Offload</label>
            <select
              id="settings-dit-cpu-offload"
              value={options.ditCpuOffload ? "enabled" : "disabled"}
              onChange={(e) =>
                updateOption("ditCpuOffload", e.target.value === "enabled")
              }
            >
              <option value="disabled">Disabled</option>
              <option value="enabled">Enabled</option>
            </select>
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-text-encoder-cpu-offload">
              Text Encoder CPU Offload
            </label>
            <select
              id="settings-text-encoder-cpu-offload"
              value={options.textEncoderCpuOffload ? "enabled" : "disabled"}
              onChange={(e) =>
                updateOption(
                  "textEncoderCpuOffload",
                  e.target.value === "enabled"
                )
              }
            >
              <option value="disabled">Disabled</option>
              <option value="enabled">Enabled</option>
            </select>
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-use-fsdp-inference">
              Use FSDP Inference
            </label>
            <select
              id="settings-use-fsdp-inference"
              value={options.useFsdpInference ? "enabled" : "disabled"}
              onChange={(e) =>
                updateOption("useFsdpInference", e.target.value === "enabled")
              }
            >
              <option value="disabled">Disabled</option>
              <option value="enabled">Enabled</option>
            </select>
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-vae-cpu-offload">VAE CPU Offload</label>
            <select
              id="settings-vae-cpu-offload"
              value={options.vaeCpuOffload ? "enabled" : "disabled"}
              onChange={(e) =>
                updateOption("vaeCpuOffload", e.target.value === "enabled")
              }
            >
              <option value="disabled">Disabled</option>
              <option value="enabled">Enabled</option>
            </select>
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-image-encoder-cpu-offload">
              Image Encoder CPU Offload
            </label>
            <select
              id="settings-image-encoder-cpu-offload"
              value={options.imageEncoderCpuOffload ? "enabled" : "disabled"}
              onChange={(e) =>
                updateOption(
                  "imageEncoderCpuOffload",
                  e.target.value === "enabled"
                )
              }
            >
              <option value="disabled">Disabled</option>
              <option value="enabled">Enabled</option>
            </select>
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="settings-enable-torch-compile">
              Torch Compile
            </label>
            <select
              id="settings-enable-torch-compile"
              value={options.enableTorchCompile ? "enabled" : "disabled"}
              onChange={(e) =>
                updateOption(
                  "enableTorchCompile",
                  e.target.value === "enabled"
                )
              }
            >
              <option value="disabled">Disabled</option>
              <option value="enabled">Enabled</option>
            </select>
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-vsa-sparsity"
              title="Video Sparse Attention sparsity (0–1, higher = sparser)"
            >
              VSA Sparsity
            </label>
            <input
              id="settings-vsa-sparsity"
              type="number"
              value={options.vsaSparsity}
              onChange={(e) =>
                updateOption("vsaSparsity", parseFloat(e.target.value))
              }
              min={0}
              max={1}
              step={0.05}
            />
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-tp-size"
              title="Tensor parallelism size (-1 = auto)"
            >
              TP Size
            </label>
            <input
              id="settings-tp-size"
              type="number"
              value={options.tpSize}
              onChange={(e) =>
                updateOption("tpSize", parseInt(e.target.value, 10) || -1)
              }
              min={-1}
              max={8}
            />
          </div>
          <div className={formStyles.formRow}>
            <label
              htmlFor="settings-sp-size"
              title="Sequence parallelism size (-1 = auto)"
            >
              SP Size
            </label>
            <input
              id="settings-sp-size"
              type="number"
              value={options.spSize}
              onChange={(e) =>
                updateOption("spSize", parseInt(e.target.value, 10) || -1)
              }
              min={-1}
              max={8}
            />
          </div>
        </div>
      </section>
    </main>
  );
}
