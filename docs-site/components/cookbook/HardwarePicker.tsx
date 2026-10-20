import { useEffect, useId, useState } from "react";
import { Input } from "@/components/ui/input";

type Purpose = "target" | "filter";

export default function HardwarePicker({
  backend,
  options,
  value,
  onChange,
  purpose = "target",
}: {
  backend: string;
  options: string[][];
  value: string;
  onChange: (value: string) => void;
  purpose?: Purpose;
}) {
  const [search, setSearch] = useState("");
  const [adding, setAdding] = useState(false);
  const [custom, setCustom] = useState("");
  useEffect(() => {
    const requested = new URL(window.location.href).searchParams.get("hardware");
    setCustom(requested?.startsWith("custom:") ? requested.slice(7) : "");
  }, []);
  const id = useId();
  const known = options.filter((option) => option.join(" ").toLowerCase().includes(search.toLowerCase()));
  const selected = value.startsWith("custom:") ? value.slice(7) : options.find(([key]) => key === value)?.[1];
  const savedUnknown = !!value && !selected;
  function add() {
    const name = search.trim();
    if (!name) return;
    setCustom(name);
    onChange(`custom:${name}`);
    setSearch("");
    setAdding(false);
  }
  function clear() {
    setCustom("");
    setSearch("");
    onChange("");
  }
  return (
    <div className="hardware-picker">
      <fieldset className="hardware-matrix">
        <legend className="sr-only">Hardware {purpose}</legend>
        {known.map(([key, title, detail]) => (
          <label key={key} data-selected={value === key}>
            <input
              type="radio"
              name={`hardware-${id}`}
              checked={value === key}
              onChange={() => onChange(key)}
              value={key}
            />
            <span>
              <strong>{title}</strong>
              <small>{detail}</small>
            </span>
          </label>
        ))}
        {custom && (
          <label data-selected={value === `custom:${custom}`}>
            <input
              type="radio"
              name={`hardware-${id}`}
              checked={value === `custom:${custom}`}
              onChange={() => onChange(`custom:${custom}`)}
            />
            <span>
              <strong>{custom}</strong>
              <small>Custom device</small>
            </span>
          </label>
        )}
      </fieldset>
      <div className="hardware-picker-actions">
        <button type="button" className="hardware-add" aria-expanded={adding} onClick={() => setAdding(!adding)}>
          + Find or add {backend === "mlx" ? "a Mac" : "a GPU"}
        </button>
        {(selected || savedUnknown) && (
          <button type="button" className="hardware-clear" onClick={clear}>
            Clear hardware {purpose}
          </button>
        )}
      </div>
      {!selected && (
        <p className="hardware-current" role="status">
          {savedUnknown ? "Saved hardware target is not in this list." : `No hardware ${purpose} selected.`}
        </p>
      )}
      {adding && (
        <form
          className="hardware-search"
          onSubmit={(event) => {
            event.preventDefault();
            add();
          }}
        >
          <label htmlFor={id}>Device name</label>
          <Input
            id={id}
            autoFocus
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={backend === "mlx" ? "e.g. M4 Max" : "e.g. RTX 4090"}
            maxLength={80}
          />
          <p>
            {purpose === "filter"
              ? "Saved in this URL and used to filter by runtime. Fit checks appear only when an estimate covers the device."
              : "Saved in this URL as your setup target. Device choice does not change generated configuration."}
          </p>
          <button type="submit" disabled={!search.trim()}>
            Add device
          </button>
        </form>
      )}
    </div>
  );
}
