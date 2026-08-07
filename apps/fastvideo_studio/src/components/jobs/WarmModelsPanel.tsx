'use client';

import * as React from 'react';
import { toast } from 'sonner';

import { Badge, type BadgeProps } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { NativeSelect } from '@/components/ui/native-select';
import { useStore } from '@/hooks/useStore';
import {
  getModels,
  listGenerators,
  preloadGenerator,
  unloadGenerator,
  type GeneratorInfo,
  type GeneratorRequest,
  type Model,
} from '@/lib/api';
import { getDefaultModelForWorkload } from '@/lib/defaultOptions';
import { defaultOptionsStore } from '@/stores/defaultOptions';

// Mirrors the backend's model_label(): readable label from an HF-style path.
function modelLabel(modelId: string): string {
  return (modelId.split('/').pop() ?? modelId).replace(/[-_]/g, ' ');
}

// Compact summary of only the non-default engine bits, e.g. "8 GPU · compile".
function configSummary(gen: GeneratorInfo): string {
  const parts: string[] = [];
  if (gen.num_gpus !== 1) parts.push(`${gen.num_gpus} GPU`);
  if (gen.sp_size !== -1) parts.push(`SP ${gen.sp_size}`);
  if (gen.tp_size !== -1) parts.push(`TP ${gen.tp_size}`);
  if (gen.dit_cpu_offload) parts.push('DiT offload');
  if (gen.text_encoder_cpu_offload) parts.push('TE offload');
  if (gen.vae_cpu_offload) parts.push('VAE offload');
  if (gen.image_encoder_cpu_offload) parts.push('image enc offload');
  if (gen.use_fsdp_inference) parts.push('FSDP');
  if (gen.enable_torch_compile) parts.push('compile');
  if (gen.vsa_sparsity > 0) parts.push(`VSA ${gen.vsa_sparsity.toFixed(2)}`);
  return parts.join(' · ');
}

// The identifying engine subset of a listed generator (the unload payload).
function toRequest(gen: GeneratorInfo): GeneratorRequest {
  return {
    model_id: gen.model_id,
    workload_type: gen.workload_type,
    num_gpus: gen.num_gpus,
    dit_cpu_offload: gen.dit_cpu_offload,
    text_encoder_cpu_offload: gen.text_encoder_cpu_offload,
    vae_cpu_offload: gen.vae_cpu_offload,
    image_encoder_cpu_offload: gen.image_encoder_cpu_offload,
    use_fsdp_inference: gen.use_fsdp_inference,
    enable_torch_compile: gen.enable_torch_compile,
    vsa_sparsity: gen.vsa_sparsity,
    tp_size: gen.tp_size,
    sp_size: gen.sp_size,
  };
}

const STATE_VARIANTS: Record<GeneratorInfo['state'], BadgeProps['variant']> = {
  ready: 'success',
  loading: 'warning',
  failed: 'destructive',
};

/**
 * Utility strip for keeping inference models resident in GPU memory: lists
 * the warm (or loading/failed) generators, preloads the selected model using
 * the persisted default job options, and unloads warm models on demand.
 */
export default function WarmModelsPanel() {
  const { options } = useStore(defaultOptionsStore);

  const [generators, setGenerators] = React.useState<GeneratorInfo[]>([]);
  const [models, setModels] = React.useState<Model[]>([]);
  const [modelId, setModelId] = React.useState('');
  const [isBusy, setIsBusy] = React.useState(false);

  const fetchGenerators = React.useCallback(async () => {
    try {
      setGenerators(await listGenerators());
    } catch (e) {
      console.error('Failed to fetch generators:', e);
    }
  }, []);

  React.useEffect(() => {
    fetchGenerators();
  }, [fetchGenerators]);

  // Poll every 5s while any preload is in flight; stop otherwise.
  const anyLoading = generators.some((g) => g.state === 'loading');
  React.useEffect(() => {
    if (!anyLoading) return;
    const interval = setInterval(fetchGenerators, 5000);
    return () => clearInterval(interval);
  }, [anyLoading, fetchGenerators]);

  // Same model catalogue (and default selection) as the create-job modal.
  React.useEffect(() => {
    getModels('t2v')
      .then((list) => {
        setModels(list);
        const defaultId = getDefaultModelForWorkload(
          defaultOptionsStore.get().options,
          't2v',
        );
        setModelId(
          list.some((m) => m.id === defaultId)
            ? defaultId
            : (list[0]?.id ?? ''),
        );
      })
      .catch((e) => console.error('Failed to load models:', e));
  }, []);

  async function handlePreload() {
    if (!modelId || isBusy) return;
    setIsBusy(true);
    try {
      await preloadGenerator({
        model_id: modelId,
        workload_type: 't2v',
        num_gpus: options.numGpus,
        dit_cpu_offload: options.ditCpuOffload,
        text_encoder_cpu_offload: options.textEncoderCpuOffload,
        vae_cpu_offload: options.vaeCpuOffload,
        image_encoder_cpu_offload: options.imageEncoderCpuOffload,
        use_fsdp_inference: options.useFsdpInference,
        enable_torch_compile: options.enableTorchCompile,
        vsa_sparsity: options.vsaSparsity,
        tp_size: options.tpSize,
        sp_size: options.spSize,
      });
      await fetchGenerators();
    } catch (err) {
      console.error('Failed to preload model:', err);
      toast.error('Model was not preloaded', {
        description:
          err instanceof Error
            ? err.message
            : 'Check the Studio API, then retry.',
      });
    } finally {
      setIsBusy(false);
    }
  }

  async function handleUnload(gen: GeneratorInfo) {
    if (isBusy) return;
    setIsBusy(true);
    try {
      await unloadGenerator(toRequest(gen));
      await fetchGenerators();
    } catch (err) {
      console.error('Failed to unload model:', err);
      toast.error('Model was not unloaded', {
        description:
          err instanceof Error
            ? err.message
            : 'Check the Studio API, then retry.',
      });
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <section
      aria-label="Warm models"
      className="mx-auto w-full max-w-[850px] px-10 pt-6"
    >
      <div className="flex flex-col gap-3 rounded-lg border border-border bg-background p-4">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="mr-auto text-sm font-semibold text-foreground">
            Warm Models
          </h2>
          <label htmlFor="warm-model-select" className="sr-only">
            Model to preload
          </label>
          <NativeSelect
            id="warm-model-select"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            disabled={isBusy || models.length === 0}
            className="h-9 w-auto max-w-64 rounded-lg"
          >
            <option value="" disabled>
              {models.length === 0 ? 'Loading models…' : 'Select a model…'}
            </option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </NativeSelect>
          <Button
            size="sm"
            onClick={handlePreload}
            disabled={isBusy || !modelId}
          >
            Preload model
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Keeps the model resident in GPU memory so jobs skip the load wait
          (uses your default job options).
        </p>
        {generators.length > 0 && (
          <ul className="flex flex-col gap-2">
            {generators.map((gen) => (
              <li
                key={JSON.stringify(toRequest(gen))}
                className="flex min-h-8 flex-wrap items-center gap-2"
              >
                <Badge
                  variant={STATE_VARIANTS[gen.state]}
                  className={
                    gen.state === 'loading' ? 'animate-pulse' : undefined
                  }
                  title={
                    gen.state === 'failed' ? (gen.error ?? undefined) : undefined
                  }
                >
                  {gen.state}
                </Badge>
                <span className="text-sm font-medium text-foreground">
                  {modelLabel(gen.model_id)}
                </span>
                <span className="text-xs text-muted-foreground">
                  {configSummary(gen)}
                </span>
                {gen.state === 'ready' && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="ml-auto"
                    onClick={() => handleUnload(gen)}
                    disabled={isBusy}
                  >
                    Unload
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
