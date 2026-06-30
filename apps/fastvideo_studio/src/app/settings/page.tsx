'use client';

import * as React from 'react';

import { useHeaderActions } from '@/components/HeaderActionsContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { NativeSelect } from '@/components/ui/native-select';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { useStore } from '@/hooks/useStore';
import { getModels, type Model } from '@/lib/api';
import {
  defaultOptionsStore,
  resetToDefaults,
  updateOption,
} from '@/stores/defaultOptions';

const labelClass = 'text-xs font-normal text-muted-foreground';

function ToggleRow({
  id,
  label,
  checked,
  onChange,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id} className={labelClass}>
        {label}
      </Label>
      <Switch id={id} checked={checked} onCheckedChange={onChange} />
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
  format,
}: {
  id: string;
  label: string;
  title?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id} title={title} className={labelClass}>
        {label}
      </Label>
      <div className="flex items-center gap-2">
        <Slider
          id={id}
          min={min}
          max={max}
          step={step}
          value={[value]}
          onValueChange={(vals) => onChange(vals[0])}
          className="min-w-0 flex-1"
        />
        <span
          aria-hidden="true"
          className="min-w-10 shrink-0 text-right text-xs text-muted-foreground"
        >
          {format ? format(value) : String(value)}
        </span>
      </div>
    </div>
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
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id} className={labelClass}>
        {label}
      </Label>
      <NativeSelect
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {children}
      </NativeSelect>
    </div>
  );
}

export default function SettingsPage() {
  const { options } = useStore(defaultOptionsStore);
  const { setActions } = useHeaderActions();

  const [modelsT2v, setModelsT2v] = React.useState<Model[]>([]);
  const [modelsI2v, setModelsI2v] = React.useState<Model[]>([]);
  const [modelsT2i, setModelsT2i] = React.useState<Model[]>([]);

  React.useEffect(() => {
    setActions(null);
    return () => setActions(null);
  }, [setActions]);

  React.useEffect(() => {
    Promise.all([getModels('t2v'), getModels('i2v'), getModels('t2i')])
      .then(([t2v, i2v, t2i]) => {
        setModelsT2v(t2v);
        setModelsI2v(i2v);
        setModelsT2i(t2i);
      })
      .catch((e) => console.error('Failed to load models:', e));
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-[850px] flex-col gap-6 px-4 pb-12 pt-6">
      <Card>
        <CardContent className="space-y-4 p-6">
          <h2 className="text-lg font-semibold">Behavior</h2>
          <div className="flex items-center justify-between gap-4">
            <Label htmlFor="settings-auto-start-job" className={labelClass}>
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
            <div className="flex flex-col gap-1.5">
              <Label
                htmlFor="settings-api-server-base-url"
                className={labelClass}
              >
                API Server Base URL
              </Label>
              <Input
                id="settings-api-server-base-url"
                type="text"
                className="font-mono text-sm"
                value={options.apiServerBaseUrl ?? ''}
                onChange={(e) =>
                  updateOption('apiServerBaseUrl', e.target.value)
                }
                placeholder="http://localhost:8189/api"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label
                htmlFor="settings-dataset-upload-path"
                className={labelClass}
              >
                Dataset Upload Path
              </Label>
              <Input
                id="settings-dataset-upload-path"
                type="text"
                className="font-mono text-sm"
                value={options.datasetUploadPath ?? ''}
                onChange={(e) =>
                  updateOption('datasetUploadPath', e.target.value)
                }
                placeholder="outputs/ui_data/uploads/datasets"
              />
            </div>
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
              {modelsT2v.map((model) => (
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
              {modelsI2v.map((model) => (
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
              {modelsT2i.map((model) => (
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

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="settings-seed" className={labelClass}>
                Seed
              </Label>
              <Input
                id="settings-seed"
                type="number"
                min={0}
                value={options.seed}
                onChange={(e) => {
                  const n = parseInt(e.target.value, 10);
                  updateOption('seed', Number.isNaN(n) ? 0 : n);
                }}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
