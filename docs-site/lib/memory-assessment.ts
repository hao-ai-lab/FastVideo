// @ts-expect-error The shared ES module has no TypeScript declarations.
import { assessCapacity, normalizeHardware } from "../../docs/js/cookbook-memory.mjs";
import weightsData from "../../docs/assets/cookbook-weights.json" with { type: "json" };

export type MemoryStatus = "fits" | "tight" | "exceeds" | "unsupported" | "unknown";

/** The subset of an `assessCapacity` result the UI shows. */
export interface MemoryAssessment {
  status: MemoryStatus;
  reason?: string;
  requiredGiB?: number;
  capacityGiB?: number;
  basis?: string;
  parts?: Record<string, number>;
  checkpointGiB?: number;
  notes?: string[];
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

export interface AssessmentHardware {
  id?: string;
  backend?: string;
  name?: string;
  memory_gb?: number;
  memory_unit?: string;
  memory_type?: string;
  usable_budget_gb?: number;
  host_ram_gb?: number;
}

const GIB = 1024 ** 3;

export function capacityHardware(hardware: AssessmentHardware) {
  return {
    ...hardware,
    memory: {
      type: hardware.memory_type,
      nominal: { value: hardware.memory_gb, unit: hardware.memory_unit ?? "GB" },
      ...(hardware.usable_budget_gb
        ? { usable: { value: hardware.usable_budget_gb, unit: hardware.memory_unit ?? "GB" } }
        : {}),
    },
    host_memory_gb: hardware.host_ram_gb,
  };
}

/** Checkpoint payload is not peak residency: offload, sharding and packing change placement. */
export function assessWeightsRecipe({
  weights,
  config,
  hardware,
  budgetGb,
}: {
  weights: WeightsModel;
  config: Record<string, unknown>;
  hardware: AssessmentHardware;
  budgetGb?: number;
}): MemoryAssessment {
  const generator = (config.generator ?? {}) as {
    engine?: { quantization?: { transformer_quant?: string | null }; offload?: Record<string, boolean> };
    pipeline?: { components?: Record<string, string | null>; experimental?: Record<string, unknown> };
  };
  const device = normalizeHardware(capacityHardware(hardware), {
    memoryBudgetGiB: budgetGb ? (hardware.memory_unit === "GiB" ? budgetGb : (budgetGb * 1e9) / GIB) : undefined,
  });
  if (!device)
    return {
      status: "unknown",
      reason:
        "Select dedicated GPU memory or unified memory with a valid budget. Host-only system RAM is not GPU capacity.",
    };
  const parts = { ...weights.parts };
  const missing: string[] = [];
  const experimental = generator.pipeline?.experimental ?? {};
  if (generator.pipeline?.components?.text_encoder_weights) {
    delete parts.text_encoder;
    missing.push("replacement text encoder");
  }
  if (generator.engine?.quantization?.transformer_quant) {
    delete parts.transformer;
    missing.push("transformer at the selected quantization and layer profile");
  }
  if (experimental.video_decode_backend === "taeh3" || experimental.h3_vae_nvfp4_decoder) {
    delete parts.vae;
    missing.push(
      experimental.video_decode_backend === "taeh3"
        ? "TAEH3 decoder"
        : "VAE with the NVFP4 decoder and retained encoder",
    );
  }
  const checkpointGiB = Object.values(parts).reduce((sum, value) => sum + value, 0);
  return {
    status: "unknown",
    basis: "checkpoint",
    checkpointGiB,
    capacityGiB: device.capacityGiB,
    parts,
    reason: missing.length
      ? `Partial checkpoint data. Missing sizes for ${missing.join("; ")}. Runtime peak memory is not modeled.`
      : "Checkpoint sizes are known, but runtime peak memory is not modeled. CPU offload and GPU count change residency, not checkpoint file sizes.",
    notes: [
      "Activations, temporary buffers, loading peaks and per-rank placement are not included. Frame count and resolution affect these unmodeled allocations.",
      ...(hardware.memory_type === "unified"
        ? ["Unified memory is shared by the CPU and GPU. Offloading to the CPU does not create a second memory pool."]
        : []),
      ...(device.budgetAssumed
        ? ["The budget uses 75% of nominal unified memory because no usable memory budget was supplied."]
        : []),
    ],
  };
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
  model: Record<string, unknown>;
  example: { workload?: string; env?: Record<string, string> };
  hardware?: AssessmentHardware;
  config?: Record<string, unknown>;
  budgetGb?: number;
  hostGb?: number;
}): MemoryAssessment | undefined {
  if (!config || !hardware?.memory_gb) return undefined;
  try {
    const result = assessCapacity({
      model,
      example,
      hardware: capacityHardware(hardware),
      config,
      memoryBudgetGiB: budgetGb ? (hardware.memory_unit === "GiB" ? budgetGb : (budgetGb * 1e9) / GIB) : undefined,
      hostMemoryGb: hostGb,
    }) as MemoryAssessment;
    return {
      status: (result.status ?? "unknown") as MemoryStatus,
      reason: result.reason,
      requiredGiB: result.requiredGiB ?? (result as MemoryAssessment & { peakGiB?: number }).peakGiB,
      capacityGiB: result.capacityGiB,
      basis: result.basis,
      notes: result.notes,
    };
  } catch (error) {
    return { status: "unknown", reason: error instanceof Error ? error.message : "Memory assessment failed." };
  }
}
