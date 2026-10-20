import { useEffect, useState } from "react";

export type MemoryType = "dedicated" | "unified" | "system";
export type MemoryUnit = "GB" | "GiB";

/** One capacity variant from the shared hardware catalog. */
export interface HardwareRecord {
  id: string;
  name: string;
  memory_type: MemoryType;
  memory_gb: number;
  memory_unit: MemoryUnit;
  backend: string;
  supported_dtypes?: string[];
  sources?: string[];
  notes?: string;
  ui_id?: string;
  aliases?: string[];
  vendor?: string;
  kind?: string;
  family?: string;
  variant?: string;
  form_factor?: string;
  gpu_count?: number;
  host_ram_gb?: number;
  usable_budget_gb?: number;
}

export interface CustomHardware {
  name: string;
  backend: string;
  memory_type: MemoryType;
  memory_gb: number;
  memory_unit: MemoryUnit;
  usable_budget_gb?: number;
  host_ram_gb?: number;
}

export interface HardwareSelection {
  id: string;
  custom?: CustomHardware;
}

export interface HardwareCatalogState {
  status: "loading" | "ready" | "unavailable";
  records: HardwareRecord[];
  error?: string;
  reload: () => void;
}

const HARDWARE_PATH = "/assets/cookbook-config/capacity/hardware.json";
const CUSTOM_PARAMS = ["hwName", "hwBackend", "hwType", "hwMemory", "hwUnit", "hwBudget", "hwRam"] as const;

const MEMORY_TYPES: MemoryType[] = ["dedicated", "unified", "system"];

function text(value: unknown, max = 200): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim().slice(0, max) : undefined;
}

function positive(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value <= 4096 ? value : undefined;
}

function numberText(value: string | null): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 && parsed <= 4096 ? parsed : undefined;
}

function memoryType(value: unknown): MemoryType | undefined {
  return MEMORY_TYPES.includes(value as MemoryType) ? (value as MemoryType) : undefined;
}

function memoryUnit(value: unknown): MemoryUnit {
  return value === "GiB" ? "GiB" : "GB";
}

function backend(value: unknown, fallback = "cuda"): string {
  if (value === "mlx" || value === "metal") return "mlx";
  if (value === "cuda" || value === "rocm") return value;
  return fallback;
}

/** Catalog records keep backends the cookbook cannot run, so they never pose as CUDA targets. */
function catalogBackend(value: unknown): string {
  if (value === "sycl" || value === "unknown") return value;
  return backend(value, "unknown");
}

/** Accepts the generated `{ version, devices: [] }` asset and the older mapping forms. */
function recordsFrom(payload: unknown): HardwareRecord[] {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return [];
  const container = payload as Record<string, unknown>;
  let entries: Array<[string, unknown]> = [];
  if (Array.isArray(container.devices)) {
    entries = container.devices.map((entry) => {
      const item = entry && typeof entry === "object" ? (entry as Record<string, unknown>) : {};
      return [text(item.id, 120) ?? "", entry];
    });
  } else {
    const source =
      container.hardware && typeof container.hardware === "object" && !Array.isArray(container.hardware)
        ? (container.hardware as Record<string, unknown>)
        : container;
    entries = Object.entries(source);
  }
  const records: HardwareRecord[] = [];
  for (const [fallbackId, value] of entries) {
    if (!value || typeof value !== "object" || Array.isArray(value)) continue;
    const item = value as Record<string, unknown>;
    const id = text(item.id, 120) ?? text(fallbackId, 120);
    const name = text(item.name, 120);
    const memory =
      item.memory && typeof item.memory === "object" ? (item.memory as Record<string, unknown>) : undefined;
    const nominal =
      memory?.nominal && typeof memory.nominal === "object" ? (memory.nominal as Record<string, unknown>) : undefined;
    const capacity = positive(nominal?.value) ?? positive(item.memory_gb);
    const type = memoryType(memory?.type) ?? memoryType(item.memory_type);
    if (!id || !name || !capacity || !type) continue;
    const usable =
      memory?.usable && typeof memory.usable === "object" ? (memory.usable as Record<string, unknown>) : undefined;
    const host =
      item.host_memory && typeof item.host_memory === "object"
        ? (item.host_memory as Record<string, unknown>)
        : undefined;
    const aliases = Array.isArray(item.aliases)
      ? item.aliases.map((alias) => text(alias, 80)).filter((alias): alias is string => !!alias)
      : undefined;
    const sources = Array.isArray(item.sources)
      ? item.sources
          .map((source) =>
            typeof source === "string"
              ? text(source, 500)
              : source && typeof source === "object"
                ? text((source as Record<string, unknown>).url, 500)
                : undefined,
          )
          .filter((url): url is string => !!url)
      : undefined;
    records.push({
      id,
      name,
      memory_type: type,
      memory_gb: capacity,
      memory_unit: memoryUnit(nominal?.unit ?? item.memory_unit),
      backend: catalogBackend(item.backend),
      supported_dtypes: Array.isArray(item.supported_dtypes)
        ? item.supported_dtypes.filter((dtype): dtype is string => typeof dtype === "string")
        : undefined,
      sources,
      notes: text(item.notes, 600),
      ui_id: text(item.ui_id, 40),
      aliases,
      vendor: text(item.vendor, 80),
      kind: text(item.kind, 40),
      family: text(item.family, 80),
      variant: text(item.variant, 80),
      form_factor: text(item.form_factor, 40),
      gpu_count: positive(item.gpu_count),
      host_ram_gb: positive(host?.value) ?? positive(item.host_memory_gb) ?? positive(item.host_ram_gb),
      usable_budget_gb: positive(usable?.value) ?? positive(item.usable_memory_gb) ?? positive(item.usable_budget_gb),
    });
  }
  return records.sort((a, b) => a.name.localeCompare(b.name));
}

/** Loads only the shared lookup data. It does not estimate fit or inspect the browser's hardware. */
export function useHardwareCatalog(): HardwareCatalogState {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<Omit<HardwareCatalogState, "reload">>({
    status: "loading",
    records: [],
  });
  useEffect(() => {
    if (typeof window === "undefined") return;
    const controller = new AbortController();
    const url = new URL(`${import.meta.env.BASE_URL.replace(/\/$/, "")}${HARDWARE_PATH}`, window.location.origin);
    setState({ status: "loading", records: [] });
    fetch(url.href, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return recordsFrom(await response.json());
      })
      .then((records) => {
        if (controller.signal.aborted) return;
        setState(
          records.length
            ? { status: "ready", records }
            : { status: "unavailable", records: [], error: "The hardware catalog is empty." },
        );
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({
            status: "unavailable",
            records: [],
            error: error instanceof Error ? error.message : "The hardware catalog could not be loaded.",
          });
      });
    return () => controller.abort();
  }, [attempt]);
  return { ...state, reload: () => setAttempt((current) => current + 1) };
}

export function readHardwareSelection(search: URLSearchParams): HardwareSelection {
  const requested = search.get("hardware")?.trim() ?? "";
  if (requested.startsWith("custom:")) {
    return {
      id: "custom",
      custom: {
        name: requested.slice(7).slice(0, 80),
        backend: "cuda",
        memory_type: "dedicated",
        memory_gb: 0,
        memory_unit: "GB",
      },
    };
  }
  if (requested !== "custom") return { id: requested };
  return {
    id: "custom",
    custom: {
      name: text(search.get("hwName"), 80) ?? "",
      backend: backend(search.get("hwBackend")),
      memory_type: memoryType(search.get("hwType")) ?? "dedicated",
      memory_gb: numberText(search.get("hwMemory")) ?? 0,
      memory_unit: memoryUnit(search.get("hwUnit")),
      usable_budget_gb: numberText(search.get("hwBudget")),
      host_ram_gb: numberText(search.get("hwRam")),
    },
  };
}

export function writeHardwareSelection(target: URL | URLSearchParams, selection: HardwareSelection) {
  const search = target instanceof URL ? target.searchParams : target;
  for (const key of CUSTOM_PARAMS) search.delete(key);
  if (!selection.id) {
    search.delete("hardware");
    return;
  }
  search.set("hardware", selection.id);
  if (selection.id !== "custom" || !selection.custom) return;
  const custom = selection.custom;
  if (custom.name) search.set("hwName", custom.name.slice(0, 80));
  search.set("hwBackend", custom.backend);
  search.set("hwType", custom.memory_type);
  if (custom.memory_gb > 0) search.set("hwMemory", String(custom.memory_gb));
  search.set("hwUnit", custom.memory_unit);
  if (custom.usable_budget_gb) search.set("hwBudget", String(custom.usable_budget_gb));
  if (custom.host_ram_gb) search.set("hwRam", String(custom.host_ram_gb));
}

export function resolveHardware(selection: HardwareSelection, records: HardwareRecord[]): HardwareRecord | undefined {
  if (selection.id === "custom") {
    const custom = selection.custom;
    if (!custom?.name.trim() || !custom.memory_gb) return undefined;
    return {
      id: "custom",
      name: custom.name.trim(),
      memory_type: custom.memory_type,
      memory_gb: custom.memory_gb,
      memory_unit: custom.memory_unit,
      backend: custom.backend,
      usable_budget_gb:
        custom.usable_budget_gb && custom.usable_budget_gb <= custom.memory_gb ? custom.usable_budget_gb : undefined,
      host_ram_gb: custom.host_ram_gb,
    };
  }
  return (
    records.find((record) => record.id === selection.id) ??
    records.find((record) => record.aliases?.includes(selection.id))
  );
}

/** A query-only selector shared by the hardware picker and tests. */
export function searchHardware(records: HardwareRecord[], query: string): HardwareRecord[] {
  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
  if (!tokens.length) return records;
  return records.filter((record) => {
    const haystack = [
      record.name,
      record.id,
      record.ui_id,
      record.vendor,
      record.kind,
      record.family,
      record.variant,
      record.form_factor,
      record.gpu_count ? `${record.gpu_count}gpu` : undefined,
      record.backend,
      record.memory_type,
      `${record.memory_gb}gb`,
      `${record.memory_gb}gib`,
      ...(record.aliases ?? []),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return tokens.every((token) => haystack.includes(token));
  });
}

export function memoryText(record: HardwareRecord, includeUnit = true): string {
  const value = `${record.memory_gb} ${record.memory_unit}`;
  if (!includeUnit) return value;
  if (record.memory_type === "dedicated") return `${value} VRAM`;
  if (record.memory_type === "unified") return `${value} unified memory`;
  return `${value} system memory`;
}

export function memoryContext(record: HardwareRecord): string {
  if (record.memory_type === "dedicated") return "Dedicated GPU memory. It is per GPU and is not shared with host RAM.";
  if (record.memory_type === "unified")
    return "Unified memory is shared by the CPU and GPU; the full amount is not available to one process.";
  return "System memory is host RAM used by the CPU and any offload.";
}

export function backendLabel(value: string): string {
  if (value === "mlx") return "Apple MLX";
  if (value === "rocm") return "ROCm";
  if (value === "sycl") return "Intel SYCL";
  if (value === "unknown") return "Unknown backend";
  return "CUDA";
}

/** GPU count never multiplies a catalog capacity into a pool. */
export function gpuScopeNote(record: HardwareRecord | undefined, gpuCount: number): string | undefined {
  if (!record || gpuCount <= 1) return undefined;
  if (record.memory_type === "unified")
    return `${gpuCount} GPUs share unified memory. This catalog entry describes one system; its ${record.memory_gb} ${record.memory_unit} is not multiplied.`;
  if (record.memory_type === "system")
    return `${gpuCount} GPUs use host system memory. This entry describes one host; the total is not multiplied.`;
  return `${gpuCount} GPUs × ${record.memory_gb} ${record.memory_unit} VRAM. Memory is per device, not a combined pool.`;
}

export function gpuCountFrom(config: Record<string, unknown> | undefined): number {
  const generator = config?.generator;
  if (!generator || typeof generator !== "object" || Array.isArray(generator)) return 1;
  const root = generator as Record<string, unknown>;
  const engine =
    root.engine && typeof root.engine === "object" && !Array.isArray(root.engine)
      ? (root.engine as Record<string, unknown>)
      : undefined;
  for (const value of [root.num_gpus, engine?.num_gpus]) {
    if (typeof value === "number" && Number.isInteger(value) && value > 0) return value;
  }
  return 1;
}
