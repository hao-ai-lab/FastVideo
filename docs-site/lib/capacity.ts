import { useEffect, useState } from "react";
// @ts-expect-error The shared ES module has no TypeScript declarations.
import { assessCapacity } from "../../docs/js/cookbook-memory.mjs";
import weightsData from "../../docs/assets/cookbook-weights.json";

export type MemoryStatus = "fits" | "tight" | "exceeds" | "unsupported" | "unknown";

/** The subset of an `assessCapacity` result the UI shows. */
export interface MemoryAssessment {
  status: MemoryStatus;
  reason?: string;
  requiredGiB?: number;
  capacityGiB?: number;
  basis?: string;
  parts?: Record<string, number>;
}

export interface WeightsModel {
  parts: Record<string, number>;
  alternates?: Record<string, number>;
  sources?: string[];
}

const WEIGHTS = weightsData as { models: Record<string, WeightsModel> };

/** Measured checkpoint sizes for recipes that have no full capacity model yet. */
export function weightsFor(modelId: string | undefined): WeightsModel | undefined {
  return modelId ? WEIGHTS.models[modelId] : undefined;
}

/**
 * Compare resident checkpoint weights with one device. This answers whether the
 * weights load, not whether a request peaks below the device limit: activations,
 * VSA workspace, and decode buffers are not modeled.
 */
export function assessWeightsRecipe({
  weights,
  config,
  hardware,
  budgetGb,
}: {
  weights: WeightsModel;
  config: Record<string, unknown>;
  hardware: { memory_gb?: number; usable_budget_gb?: number; memory_unit?: string };
  budgetGb?: number;
}): MemoryAssessment | undefined {
  const generator = (config.generator ?? {}) as {
    engine?: { offload?: Record<string, boolean> };
    pipeline?: {
      components?: Record<string, string | null>;
      experimental?: Record<string, unknown>;
    };
  };
  const parts = { ...weights.parts };
  const components = generator.pipeline?.components ?? {};
  const experimental = generator.pipeline?.experimental ?? {};
  if (components.text_encoder_weights && weights.alternates?.text_encoder_quantized !== undefined) {
    parts.text_encoder = weights.alternates.text_encoder_quantized;
  }
  const quant = (config.generator as { engine?: { quantization?: { transformer_quant?: string | null } } })?.engine
    ?.quantization?.transformer_quant;
  if (quant && weights.alternates) {
    const key = quant === "NVFP4" ? "transformer_nvfp4" : quant === "FP8" ? "transformer_fp8" : undefined;
    if (key && weights.alternates[key] !== undefined) parts.transformer = weights.alternates[key];
  }
  const decoder = experimental.h3_vae_nvfp4_decoder;
  if (decoder && weights.alternates) {
    const key = String(decoder).includes("fast-8L") ? "vae_nvfp4_fast" : "vae_nvfp4_quality";
    if (weights.alternates[key] !== undefined) parts.vae = weights.alternates[key];
  }
  const offload = generator.engine?.offload ?? {};
  if (offload.text_encoder) delete parts.text_encoder;
  const resident = Object.values(parts).reduce((sum, value) => sum + value, 0);
  const capacityGb = budgetGb ?? hardware.usable_budget_gb ?? hardware.memory_gb;
  if (!capacityGb) return undefined;
  const capacityGiB = hardware.memory_unit === "GiB" ? capacityGb : (capacityGb * 1e9) / GIB;
  const label: Record<string, string> = {
    transformer: "transformer",
    text_encoder: "text encoder",
    vae: "VAE",
    audio_vae: "audio VAE",
  };
  const split = Object.entries(parts)
    .map(([role, value]) => `${label[role] ?? role} ${value.toFixed(1)} GiB`)
    .join(", ");
  const status: MemoryStatus = resident > capacityGiB ? "exceeds" : resident > capacityGiB * 0.85 ? "tight" : "fits";
  const offloaded = offload.text_encoder ? " The text encoder is offloaded." : "";
  return {
    status,
    basis: "weights",
    requiredGiB: resident,
    capacityGiB,
    parts,
    reason: `Weights only: ${split}.${offloaded} Activations, VSA workspace, and decode buffers are not modeled.`,
  };
}

interface CapacityModel {
  id: string;
  name?: string;
  examples?: { deployment_id?: string }[];
  reference_runs?: unknown[];
  [key: string]: unknown;
}

interface CapacityState {
  status: "loading" | "ready" | "unavailable";
  model?: CapacityModel;
}

const CAPACITY_INDEX = `${import.meta.env.BASE_URL.replace(/\/$/, "")}/assets/cookbook-config/capacity/index.json`;
const GIB = 1024 ** 3;

let indexPromise: Promise<{ models: { id: string; catalog_url: string }[] }> | null = null;

function loadIndex() {
  indexPromise ??= fetch(CAPACITY_INDEX)
    .then((response) => {
      if (!response.ok) throw new Error(`Capacity index: HTTP ${response.status}`);
      return response.json() as Promise<{ models: { id: string; catalog_url: string }[] }>;
    })
    .catch((error) => {
      indexPromise = null;
      throw error;
    });
  return indexPromise;
}

/**
 * Load the capacity model for one cookbook entry. Most recipes do not have one;
 * the hook stays `unavailable` and the UI says so instead of guessing.
 */
export function useCapacityModel(modelId: string | undefined): CapacityState {
  const [state, setState] = useState<CapacityState>({ status: "loading" });
  useEffect(() => {
    if (!modelId) {
      setState({ status: "unavailable" });
      return;
    }
    let live = true;
    setState({ status: "loading" });
    loadIndex()
      .then(async (index) => {
        const entry = index.models.find((model) => model.id === modelId);
        if (!entry) throw new Error("No capacity model ships for this checkpoint.");
        const url = new URL(entry.catalog_url, new URL(CAPACITY_INDEX, window.location.origin));
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Capacity model: HTTP ${response.status}`);
        const model = (await response.json()) as CapacityModel;
        if (live) setState({ status: "ready", model });
      })
      .catch(() => {
        if (live) setState({ status: "unavailable" });
      });
    return () => {
      live = false;
    };
  }, [modelId]);
  return state;
}

/**
 * Run the analytical estimate for the edited configuration. Returns `undefined`
 * when no capacity model covers the checkpoint, so the caller can say so.
 */
export function assessRecipe({
  model,
  example,
  hardware,
  config,
  budgetGb,
  hostGb,
}: {
  model: CapacityModel;
  example: { workload?: string; env?: Record<string, string> };
  hardware?: {
    memory_gb?: number;
    memory_unit?: string;
    memory_type?: string;
    usable_budget_gb?: number;
    host_ram_gb?: number;
    id?: string;
  };
  config?: Record<string, unknown>;
  budgetGb?: number;
  hostGb?: number;
}): MemoryAssessment | undefined {
  if (!config || !hardware?.memory_gb) return undefined;
  try {
    const result = assessCapacity({
      model,
      example,
      hardware,
      config,
      memoryBudgetGiB: budgetGb ? (hardware.memory_unit === "GiB" ? budgetGb : (budgetGb * 1e9) / GIB) : undefined,
      hostMemoryGb: hostGb,
    }) as MemoryAssessment;
    return {
      status: (result.status ?? "unknown") as MemoryStatus,
      reason: result.reason,
      requiredGiB: result.requiredGiB,
      capacityGiB: result.capacityGiB,
      basis: result.basis,
    };
  } catch {
    return undefined;
  }
}
