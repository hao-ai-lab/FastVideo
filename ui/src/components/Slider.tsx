'use client';

import sliderStyles from "./styles/Slider.module.css";

interface SliderProps {
  id: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  showValue?: boolean;
  formatValue?: (value: number) => string;
  "aria-label"?: string;
}

export default function Slider({
  id,
  min,
  max,
  step,
  value,
  onChange,
  disabled = false,
  showValue = true,
  formatValue = (v) => String(v),
  "aria-label": ariaLabel,
}: SliderProps) {
  return (
    <div className={sliderStyles.sliderWrapper}>
      <input
        type="range"
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        aria-label={ariaLabel}
      />
      {showValue && (
        <span className={sliderStyles.sliderValue} aria-hidden>
          {formatValue(value)}
        </span>
      )}
    </div>
  );
}
