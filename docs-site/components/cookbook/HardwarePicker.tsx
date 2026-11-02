import { useEffect, useId, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  backendLabel,
  gpuScopeNote,
  memoryText,
  resolveHardware,
  searchHardware,
  type CustomHardware,
  type HardwareCatalogState,
  type HardwareRecord,
  type HardwareSelection,
  type MemoryType,
  type MemoryUnit,
} from "@/lib/hardware";

interface Draft {
  name: string;
  backend: string;
  memory_type: MemoryType;
  memory: string;
  memory_unit: MemoryUnit;
  budget: string;
  ram: string;
}

function draftFrom(custom: CustomHardware | undefined, backend: string): Draft {
  return {
    name: custom?.name ?? "",
    backend: custom?.backend ?? (backend === "mlx" ? "mlx" : "cuda"),
    memory_type: custom?.memory_type ?? (backend === "mlx" ? "unified" : "dedicated"),
    memory: custom?.memory_gb ? String(custom.memory_gb) : "",
    memory_unit: custom?.memory_unit ?? "GB",
    budget: custom?.usable_budget_gb ? String(custom.usable_budget_gb) : "",
    ram: custom?.host_ram_gb ? String(custom.host_ram_gb) : "",
  };
}

function amount(value: string): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 && parsed <= 4096 ? parsed : undefined;
}

export default function HardwarePicker({
  backend,
  catalog,
  status,
  error,
  value,
  onChange,
  gpuCount = 1,
  reload,
}: {
  backend: string;
  catalog: HardwareRecord[];
  status: HardwareCatalogState["status"];
  error?: string;
  value: HardwareSelection;
  onChange: (selection: HardwareSelection) => void;
  gpuCount?: number;
  reload?: () => void;
}) {
  const id = useId();
  const panelId = `${id}-lookup`;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState<Draft>(() => draftFrom(value.custom, backend));
  const record = resolveHardware(value, catalog);
  const unknown = status === "ready" && !!value.id && value.id !== "custom" && !record;
  const loading = status === "loading" && !!value.id && value.id !== "custom";
  const candidates = catalog.filter((item) => item.backend === (backend === "mlx" ? "mlx" : "cuda"));
  const matches = searchHardware(candidates, query).slice(0, 20);
  const draftMemory = amount(draft.memory);
  const draftBudget = amount(draft.budget);
  const budgetTooLarge = draftMemory !== undefined && draftBudget !== undefined && draftBudget > draftMemory;
  const validDraft = !!draft.name.trim() && !!draftMemory && !budgetTooLarge;
  const scope = gpuScopeNote(record, gpuCount);

  useEffect(() => {
    if (value.custom) setDraft(draftFrom(value.custom, backend));
  }, [backend, value.custom]);

  useEffect(() => {
    if (status === "unavailable") setAdding(true);
  }, [status]);

  useEffect(() => {
    if (open && status === "ready" && !adding) searchRef.current?.focus();
  }, [adding, open, status]);

  function close() {
    setOpen(false);
    triggerRef.current?.focus();
  }

  function choose(recordId: string) {
    onChange({ id: recordId });
    setOpen(false);
    setQuery("");
  }

  function save(event: React.FormEvent) {
    event.preventDefault();
    const memory = amount(draft.memory);
    if (!validDraft || !memory) return;
    onChange({
      id: "custom",
      custom: {
        name: draft.name.trim(),
        backend: draft.backend,
        memory_type: draft.memory_type,
        memory_gb: memory,
        memory_unit: draft.memory_unit,
        usable_budget_gb: amount(draft.budget),
        host_ram_gb: draft.memory_type === "unified" ? undefined : amount(draft.ram),
      },
    });
    setOpen(false);
    setAdding(false);
  }

  return (
    <div
      className="hardware-picker"
      onKeyDown={(event) => {
        if (event.key === "Escape" && open) {
          event.stopPropagation();
          close();
        }
      }}
    >
      {record ? (
        <div className="hardware-selected">
          <div>
            <span className="hardware-selected-name">{record.name}</span>
            <span className="hardware-selected-meta">
              {memoryText(record)} · {backendLabel(record.backend)}
              {record.form_factor ? ` · ${record.form_factor}` : ""}
            </span>
            {record.usable_budget_gb ? (
              <span className="hardware-selected-meta">
                Usable memory: {record.usable_budget_gb} {record.memory_unit}
              </span>
            ) : null}
            {record.host_ram_gb && record.memory_type !== "unified" ? (
              <span className="hardware-selected-meta">
                Host RAM: {record.host_ram_gb} {record.memory_unit}
              </span>
            ) : null}
          </div>
          <button type="button" className="hardware-clear" onClick={() => onChange({ id: "" })}>
            Clear hardware
          </button>
        </div>
      ) : value.id === "custom" ? (
        <p className="hardware-current" role="status">
          Add this device's memory to finish selecting it.
        </p>
      ) : loading ? (
        <p className="hardware-current" role="status">
          Loading hardware catalog…
        </p>
      ) : unknown ? (
        <p className="hardware-current" role="status">
          The saved device ({value.id}) is not in this catalog.
        </p>
      ) : (
        <p className="hardware-current" role="status">
          No hardware selected.
        </p>
      )}
      {record ? (
        <p className="hardware-address-note">
          {record.memory_type === "unified"
            ? "Shared with the CPU; not dedicated VRAM."
            : record.memory_type === "dedicated"
              ? "Per device, not a combined pool."
              : "Host memory, not GPU VRAM."}
        </p>
      ) : null}
      {scope ? <p className="hardware-scope">{scope}</p> : null}
      <div className="hardware-picker-actions">
        {!record && value.id ? (
          <button type="button" className="hardware-clear" onClick={() => onChange({ id: "" })}>
            Clear hardware
          </button>
        ) : null}
        <button
          ref={triggerRef}
          type="button"
          className="hardware-add"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => {
            setOpen(!open);
            setAdding(false);
          }}
        >
          {record ? "Change hardware" : backend === "mlx" ? "+ Find or add a Mac" : "+ Find or add a GPU"}
        </button>
      </div>
      {record || open ? (
        <p className="hardware-picker-caption">Used for the setup link and, on single-unit systems, the GPU count.</p>
      ) : null}
      {open ? (
        <div className="hardware-lookup" id={panelId} onKeyDown={(event) => event.key === "Escape" && close()}>
          {status === "loading" ? <p className="hardware-lookup-status">Loading hardware catalog…</p> : null}
          {status === "unavailable" ? (
            <div className="hardware-lookup-status">
              <p>Hardware catalog unavailable. Add your device manually.</p>
              {error ? <span>{error}</span> : null}
              {reload ? (
                <button type="button" className="hardware-retry" onClick={reload}>
                  Retry
                </button>
              ) : null}
            </div>
          ) : null}
          {status === "ready" ? (
            <>
              <Input
                ref={searchRef}
                aria-label="Search hardware"
                type="search"
                placeholder="Search by name, alias, or memory"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <div className="hardware-result-list" role="radiogroup" aria-label="Matching devices">
                {matches.map((item) => (
                  <label key={item.id} className="hardware-result" data-selected={value.id === item.id}>
                    <input
                      type="radio"
                      name={`${id}-hardware`}
                      checked={value.id === item.id}
                      onChange={() => choose(item.id)}
                    />
                    <span>
                      <strong>{item.name}</strong>
                      <small>
                        {memoryText(item)} · {backendLabel(item.backend)}
                      </small>
                    </span>
                  </label>
                ))}
              </div>
              {!matches.length ? <p className="hardware-lookup-status">No matching devices.</p> : null}
            </>
          ) : null}
          <button
            type="button"
            className="hardware-manual-toggle"
            aria-expanded={adding}
            onClick={() => setAdding(!adding)}
          >
            {adding ? "Hide manual entry" : "Can't find it? Add a device"}
          </button>
          {adding ? (
            <form className="hardware-form" onSubmit={save}>
              <p className="hardware-form-title">Add a device</p>
              <div className="hardware-form-grid">
                <label className="hardware-form-field hardware-form-wide">
                  <span>Device name</span>
                  <Input
                    value={draft.name}
                    maxLength={80}
                    placeholder={backend === "mlx" ? "e.g. M4 Max" : "e.g. RTX 4090"}
                    onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                  />
                </label>
                <label className="hardware-form-field">
                  <span>Backend</span>
                  <select
                    value={draft.backend}
                    onChange={(event) => setDraft({ ...draft, backend: event.target.value })}
                  >
                    <option value="cuda">CUDA</option>
                    <option value="mlx">Apple MLX</option>
                  </select>
                </label>
                <label className="hardware-form-field">
                  <span>Memory type</span>
                  <select
                    value={draft.memory_type}
                    onChange={(event) => setDraft({ ...draft, memory_type: event.target.value as MemoryType })}
                  >
                    <option value="dedicated">GPU memory (VRAM)</option>
                    <option value="unified">Unified memory</option>
                    <option value="system">System memory (RAM)</option>
                  </select>
                </label>
                <label className="hardware-form-field">
                  <span>Total memory</span>
                  <Input
                    type="number"
                    min="1"
                    step="any"
                    inputMode="decimal"
                    value={draft.memory}
                    onChange={(event) => setDraft({ ...draft, memory: event.target.value })}
                  />
                </label>
                <label className="hardware-form-field">
                  <span>Memory unit</span>
                  <select
                    value={draft.memory_unit}
                    onChange={(event) => setDraft({ ...draft, memory_unit: event.target.value as MemoryUnit })}
                  >
                    <option value="GB">GB</option>
                    <option value="GiB">GiB</option>
                  </select>
                </label>
                <label className="hardware-form-field">
                  <span>Usable memory (optional)</span>
                  <Input
                    type="number"
                    min="1"
                    step="any"
                    inputMode="decimal"
                    placeholder="Your own limit"
                    value={draft.budget}
                    onChange={(event) => setDraft({ ...draft, budget: event.target.value })}
                  />
                </label>
                {draft.memory_type !== "unified" ? (
                  <label className="hardware-form-field">
                    <span>Host RAM (optional)</span>
                    <Input
                      type="number"
                      min="1"
                      step="any"
                      inputMode="decimal"
                      value={draft.ram}
                      onChange={(event) => setDraft({ ...draft, ram: event.target.value })}
                    />
                  </label>
                ) : null}
              </div>
              {budgetTooLarge ? <p className="hardware-form-error">Usable memory cannot exceed total memory.</p> : null}
              <p className="hardware-form-note">
                Total memory is the device's capacity. Usable memory is the part FastVideo may use, such as a MIG slice
                or a limit you set. Unified memory is shared with the CPU.
              </p>
              <div className="hardware-form-actions">
                <button type="submit" disabled={!validDraft}>
                  Save device
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setAdding(false);
                    setDraft(draftFrom(value.custom, backend));
                  }}
                >
                  Cancel
                </button>
              </div>
            </form>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
