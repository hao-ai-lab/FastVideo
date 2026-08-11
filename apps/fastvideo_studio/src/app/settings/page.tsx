'use client';

import * as React from 'react';

import {
  FieldRow,
  NumberRow,
  SliderRow,
  ToggleRow,
} from '@/components/form-rows';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { NativeSelect } from '@/components/ui/native-select';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { useStore } from '@/hooks/useStore';
import { getModels, type Model } from '@/lib/api';
import {
  defaultOptionsStore,
  resetToDefaults,
  updateOption,
} from '@/stores/defaultOptions';

function TextSettingRow({
  id,
  label,
  value,
  placeholder,
  onCommit,
}: {
  id: string;
  label: string;
  value: string;
  placeholder?: string;
  onCommit: (v: string) => void;
}) {
  // Buffer keystrokes locally and persist on blur/Enter so each character
  // doesn't fire a settings PUT (or a localStorage write).
  const [draft, setDraft] = React.useState<string | null>(null);
  return (
    <FieldRow htmlFor={id} label={label}>
      <Input
        id={id}
        type="text"
        className="font-mono text-sm"
        value={draft ?? value}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          if (draft !== null && draft !== value) onCommit(draft);
          setDraft(null);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
        }}
        placeholder={placeholder}
      />
    </FieldRow>
  );
}

function SelectRow({
  id,
  label,
  value,
  onChange,
  children,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <FieldRow htmlFor={id} label={label}>
      <NativeSelect
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {children}
      </NativeSelect>
    </FieldRow>
  );
}

export default function SettingsPage() {
  const { options } = useStore(defaultOptionsStore);

  const [models, setModels] = React.useState<
    Record<'t2v' | 'i2v' | 't2i', Model[]>
  >({ t2v: [], i2v: [], t2i: [] });

  React.useEffect(() => {
    Promise.all([getModels('t2v'), getModels('i2v'), getModels('t2i')])
      .then(([t2v, i2v, t2i]) => setModels({ t2v, i2v, t2i }))
      .catch((e) => console.error('Failed to load models:', e));
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-[850px] flex-col gap-6 px-4 pb-12 pt-6">
      <Card>
        <CardContent className="space-y-4 p-6">
          <h2 className="text-lg font-semibold">Behavior</h2>
          <div className="flex items-center justify-between gap-4">
            <Label
              htmlFor="settings-auto-start-job"
              className="pl-0.5 text-xs font-normal tracking-wide text-muted-foreground"
            >
              Auto Start Job on Create
            </Label>
            <Switch
              id="settings-auto-start-job"
              checked={options.autoStartJob}
              onCheckedChange={(v) => updateOption('autoStartJob', v)}
            />
          </div>

          <Separator />

          <h2 className="text-lg font-semibold">Paths</h2>
          <div className="space-y-4">
            <TextSettingRow
              id="settings-api-server-base-url"
              label="API Server Base URL"
              value={options.apiServerBaseUrl ?? ''}
              onCommit={(v) => updateOption('apiServerBaseUrl', v)}
              placeholder="http://localhost:8189/api"
            />
            <TextSettingRow
              id="settings-dataset-upload-path"
              label="Dataset Upload Path"
              value={options.datasetUploadPath ?? ''}
              onCommit={(v) => updateOption('datasetUploadPath', v)}
              placeholder="outputs/ui_data/uploads/datasets"
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Default Options</h2>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={resetToDefaults}
            >
              Reset to Defaults
            </Button>
          </div>
          <p className="text-sm text-muted-foreground">
            These values are used as defaults when creating new jobs.
          </p>

          <div className="grid gap-x-3 gap-y-2 [grid-template-columns:repeat(auto-fill,minmax(160px,1fr))]">
            <SelectRow
              id="settings-default-model-t2v"
              label="Default Model (T2V)"
              value={options.defaultModelIdT2v}
              onChange={(v) => updateOption('defaultModelIdT2v', v)}
            >
              <option value="">None (select when creating job)</option>
              {models.t2v.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label} ({model.id})
                </option>
              ))}
            </SelectRow>
            <SelectRow
              id="settings-default-model-i2v"
              label="Default Model (I2V)"
              value={options.defaultModelIdI2v}
              onChange={(v) => updateOption('defaultModelIdI2v', v)}
            >
              <option value="">None</option>
              {models.i2v.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label} ({model.id})
                </option>
              ))}
            </SelectRow>
            <SelectRow
              id="settings-default-model-t2i"
              label="Default Model (T2I)"
              value={options.defaultModelIdT2i}
              onChange={(v) => updateOption('defaultModelIdT2i', v)}
            >
              <option value="">None</option>
              {models.t2i.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label} ({model.id})
                </option>
              ))}
            </SelectRow>

            <SliderRow
              id="settings-num-frames"
              label="Frames"
              min={1}
              max={500}
              step={1}
              value={options.numFrames}
              onChange={(v) => updateOption('numFrames', v)}
            />
            <SliderRow
              id="settings-height"
              label="Height"
              min={64}
              max={1080}
              step={16}
              value={options.height}
              onChange={(v) => updateOption('height', v)}
            />
            <SliderRow
              id="settings-width"
              label="Width"
              min={64}
              max={1920}
              step={16}
              value={options.width}
              onChange={(v) => updateOption('width', v)}
            />
            <SliderRow
              id="settings-num-steps"
              label="Inference Steps"
              min={1}
              max={200}
              step={1}
              value={options.numInferenceSteps}
              onChange={(v) => updateOption('numInferenceSteps', v)}
            />
            <SliderRow
              id="settings-vsa-sparsity"
              label="VSA Sparsity"
              title="VSA sparsity (0–1)"
              min={0}
              max={1}
              step={0.05}
              value={options.vsaSparsity}
              onChange={(v) => updateOption('vsaSparsity', v)}
              format={(v) => v.toFixed(2)}
            />
            <SliderRow
              id="settings-guidance"
              label="Guidance Scale"
              min={0}
              max={20}
              step={0.1}
              value={options.guidanceScale}
              onChange={(v) => updateOption('guidanceScale', v)}
              format={(v) => v.toFixed(1)}
            />
            <SliderRow
              id="settings-guidance-rescale"
              label="Guidance Rescale"
              title="0 = disabled"
              min={0}
              max={1}
              step={0.05}
              value={options.guidanceRescale ?? 0}
              onChange={(v) => updateOption('guidanceRescale', v)}
              format={(v) => v.toFixed(2)}
            />
            <SliderRow
              id="settings-tp-size"
              label="TP Size"
              title="-1 = auto"
              min={-1}
              max={8}
              step={1}
              value={options.tpSize}
              onChange={(v) => updateOption('tpSize', v)}
              format={(v) => (v === -1 ? 'Auto' : String(v))}
            />
            <SliderRow
              id="settings-sp-size"
              label="SP Size"
              title="-1 = auto"
              min={-1}
              max={8}
              step={1}
              value={options.spSize}
              onChange={(v) => updateOption('spSize', v)}
              format={(v) => (v === -1 ? 'Auto' : String(v))}
            />
            <SliderRow
              id="settings-fps"
              label="FPS"
              min={1}
              max={60}
              step={1}
              value={options.fps ?? 24}
              onChange={(v) => updateOption('fps', v)}
            />

            <ToggleRow
              id="settings-dit-cpu-offload"
              label="DiT CPU Offload"
              checked={options.ditCpuOffload}
              onChange={(v) => updateOption('ditCpuOffload', v)}
            />
            <ToggleRow
              id="settings-text-encoder-cpu-offload"
              label="Text Encoder CPU Offload"
              checked={options.textEncoderCpuOffload}
              onChange={(v) => updateOption('textEncoderCpuOffload', v)}
            />
            <ToggleRow
              id="settings-use-fsdp-inference"
              label="Use FSDP Inference"
              checked={options.useFsdpInference}
              onChange={(v) => updateOption('useFsdpInference', v)}
            />
            <ToggleRow
              id="settings-vae-cpu-offload"
              label="VAE CPU Offload"
              checked={options.vaeCpuOffload}
              onChange={(v) => updateOption('vaeCpuOffload', v)}
            />
            <ToggleRow
              id="settings-image-encoder-cpu-offload"
              label="Image Encoder CPU Offload"
              checked={options.imageEncoderCpuOffload}
              onChange={(v) => updateOption('imageEncoderCpuOffload', v)}
            />
            <ToggleRow
              id="settings-enable-torch-compile"
              label="Torch Compile"
              checked={options.enableTorchCompile}
              onChange={(v) => updateOption('enableTorchCompile', v)}
            />

            <SliderRow
              id="settings-num-gpus"
              label="GPUs"
              min={1}
              max={8}
              step={1}
              value={options.numGpus}
              onChange={(v) => updateOption('numGpus', v)}
            />

            <NumberRow
              id="settings-seed"
              label="Seed"
              min={0}
              value={options.seed}
              onChange={(v) => updateOption('seed', v)}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
