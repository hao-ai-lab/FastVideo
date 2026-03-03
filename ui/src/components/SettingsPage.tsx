'use client';

import { useEffect, useState } from "react";
import { useDefaultOptions } from "@/contexts/DefaultOptionsContext";
import { useHeaderActions } from "@/contexts/ActiveTabContext";
import { getModels, type Model } from "@/lib/api";
import cardStyles from "@styles/Card.module.css";
import formStyles from "@styles/Form.module.css";
import layoutStyles from "@/app/Layout.module.css";
import buttonStyles from "@styles/Button.module.css";

export default function SettingsPage() {
  useHeaderActions([]);
  const { options, updateOption, resetToDefaults } = useDefaultOptions();
  const [models, setModels] = useState<Model[]>([]);

  useEffect(() => {
    getModels()
      .then(setModels)
      .catch((error) => console.error("Failed to load models:", error));
  }, []);

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
          <div className={formStyles.formRow}>
            <label htmlFor="settings-default-model">Default Model</label>
            <select
              id="settings-default-model"
              value={options.defaultModelId}
              onChange={(e) =>
                updateOption("defaultModelId", e.target.value)
              }
            >
              <option value="">None (select when creating job)</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label} ({model.id})
                </option>
              ))}
            </select>
          </div>
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
        </div>
      </section>
    </main>
  );
}
