'use client';

import toggleStyles from "./styles/Toggle.module.css";

interface ToggleProps {
  id: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
  "aria-label"?: string;
}

export default function Toggle({
  id,
  checked,
  onChange,
  disabled = false,
  label,
  "aria-label": ariaLabel,
}: ToggleProps) {
  return (
    <label
      htmlFor={id}
      className={`${toggleStyles.toggle} ${disabled ? toggleStyles.toggleDisabled : ""}`}
      data-checked={checked}
    >
      <input
        type="checkbox"
        id={id}
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        aria-label={ariaLabel ?? label}
      />
      <span className={toggleStyles.toggleTrack} aria-hidden />
      {label != null && (
        <span className={toggleStyles.toggleLabel}>{label}</span>
      )}
    </label>
  );
}
