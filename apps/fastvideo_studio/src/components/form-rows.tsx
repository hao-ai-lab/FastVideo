'use client';

import * as React from 'react';

import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/utils';

/**
 * Labeled form-field rows shared by the Settings page and the Create Job
 * modal. They mirror the Svelte `Toggle`/`Slider` UX on top of the shadcn
 * `Switch`/`Slider`/`Input` primitives.
 */

export function FieldRow({
  htmlFor,
  label,
  title,
  className,
  children,
}: {
  htmlFor: string;
  label: string;
  title?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      <Label
        htmlFor={htmlFor}
        title={title}
        className="pl-0.5 text-xs font-normal tracking-wide text-muted-foreground"
      >
        {label}
      </Label>
      {children}
    </div>
  );
}

export function SliderRow({
  id,
  label,
  title,
  min,
  max,
  step,
  value,
  onChange,
  disabled,
  format = (v) => String(v),
}: {
  id: string;
  label: string;
  title?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  /** Called once per gesture (pointer release / key press), not per drag tick. */
  onChange: (v: number) => void;
  disabled?: boolean;
  format?: (v: number) => string;
}) {
  // Track the in-progress drag locally so `onChange` only fires on commit;
  // any external `value` change (commit landing, reset button) takes over.
  const [dragValue, setDragValue] = React.useState<number | null>(null);
  React.useEffect(() => {
    setDragValue(null);
  }, [value]);
  const shown = dragValue ?? value;
  return (
    <FieldRow htmlFor={id} label={label} title={title}>
      <div className="flex items-center gap-2">
        <Slider
          id={id}
          min={min}
          max={max}
          step={step}
          value={[shown]}
          onValueChange={(v) => setDragValue(v[0])}
          onValueCommit={(v) => onChange(v[0])}
          disabled={disabled}
          aria-label={label}
          className="min-w-0 flex-1"
        />
        <span
          aria-hidden="true"
          className="min-w-10 shrink-0 text-right text-sm tabular-nums text-muted-foreground"
        >
          {format(shown)}
        </span>
      </div>
    </FieldRow>
  );
}

export function ToggleRow({
  id,
  label,
  title,
  checked,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  title?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <FieldRow htmlFor={id} label={label} title={title}>
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
      />
    </FieldRow>
  );
}

export function NumberRow({
  id,
  label,
  title,
  min,
  max,
  step,
  value,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  title?: string;
  min?: number;
  max?: number;
  step?: number | string;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  // Buffer the raw text so the field can be emptied while retyping; only
  // valid numbers are committed, and blur restores the last committed value.
  const [draft, setDraft] = React.useState<string | null>(null);
  React.useEffect(() => {
    setDraft(null);
  }, [value]);
  return (
    <FieldRow htmlFor={id} label={label} title={title}>
      <Input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        value={draft ?? value}
        onChange={(e) => {
          setDraft(e.target.value);
          const v = e.target.valueAsNumber;
          if (!Number.isNaN(v)) onChange(v);
        }}
        onBlur={() => setDraft(null)}
        disabled={disabled}
      />
    </FieldRow>
  );
}
