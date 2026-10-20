import { useEffect, useId, useState } from "react";
import { Input } from "@/components/ui/input";

export default function HardwarePicker({
  backend,
  options,
  value,
  onChange,
}: {
  backend: string;
  options: string[][];
  value: string;
  onChange: (value: string) => void;
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
  function add() {
    const name = search.trim();
    if (!name) return;
    setCustom(name);
    onChange(`custom:${name}`);
    setSearch("");
    setAdding(false);
  }
  return (
    <div className="hardware-picker">
      <fieldset className="hardware-matrix">
        <legend className="sr-only">Hardware target</legend>
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
      <button type="button" className="hardware-add" aria-expanded={adding} onClick={() => setAdding(!adding)}>
        + Find or add {backend === "mlx" ? "a Mac" : "a GPU"}
      </button>
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
          <p>Search the list or add your device. Memory lookup and fit checks are not connected yet.</p>
          <button type="submit" disabled={!search.trim()}>
            Add device
          </button>
        </form>
      )}
    </div>
  );
}
