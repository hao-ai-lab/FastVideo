'use client';

import * as React from 'react';
import { ChevronDown } from 'lucide-react';

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

export function OptionSection({
  title,
  description,
  defaultOpen = false,
  children,
}: {
  title: string;
  description: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <details
      open={defaultOpen}
      className="group rounded-xl border border-border bg-muted/15 px-4 py-3"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 [&::-webkit-details-marker]:hidden">
        <span>
          <span className="block text-sm font-medium">{title}</span>
          <span className="mt-0.5 block text-xs text-muted-foreground">
            {description}
          </span>
        </span>
        <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" aria-hidden />
      </summary>
      <div className="mt-4 grid grid-cols-[repeat(auto-fit,minmax(min(100%,220px),1fr))] gap-x-5 gap-y-4">
        {children}
      </div>
    </details>
  );
}

function NumericInput({
  value,
  onCommit,
  min,
  max,
  step = 1,
  ...props
}: Omit<React.ComponentProps<typeof Input>, 'value' | 'onChange' | 'min' | 'max' | 'step'> & {
  value: number;
  onCommit: (value: number) => void;
  min?: number;
  max?: number;
  step?: number | string;
}) {
  // Keep incomplete text local. Settings persist on commit, not per keystroke.
  const [draft, setDraft] = React.useState<string | null>(null);
  React.useEffect(() => setDraft(null), [value]);

  const commit = () => {
    if (draft !== null && draft.trim() !== '') {
      let next = Number(draft);
      if (Number.isFinite(next)) {
        const lower = min ?? -Infinity;
        const upper = max ?? Infinity;
        next = Math.max(lower, Math.min(upper, next));
        const increment = Number(step);
        if (Number.isFinite(increment) && increment > 0) {
          const base = min ?? 0;
          const lastStep = Math.floor((upper - base) / increment);
          const steps = Math.min(lastStep, Math.round((next - base) / increment));
          // Remove floating-point noise from decimal step arithmetic.
          next = Number((base + steps * increment).toPrecision(12));
        }
        if (next !== value) onCommit(next);
      }
    }
    setDraft(null);
  };

  return (
    <Input
      {...props}
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft ?? value}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          event.currentTarget.blur();
        } else if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          setDraft(null);
        }
      }}
      className={cn('tabular-nums [&::-webkit-inner-spin-button]:opacity-100', props.className)}
    />
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
  /** Called on slider commit or numeric blur/Enter, not per keystroke. */
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
          id={`${id}-slider`}
          min={min}
          max={max}
          step={step}
          value={[shown]}
          onValueChange={(v) => setDragValue(v[0])}
          onValueCommit={(v) => onChange(v[0])}
          disabled={disabled}
          aria-label={label}
          aria-describedby={Number.isNaN(Number(format(shown))) ? `${id}-hint` : undefined}
          className="min-w-0 flex-1"
        />
        <NumericInput
          id={id}
          aria-label={`${label} value`}
          title={title}
          min={min}
          max={max}
          step={step}
          value={shown}
          onCommit={onChange}
          disabled={disabled}
          className="w-24 shrink-0 rounded-lg px-2"
        />
      </div>
      {Number.isNaN(Number(format(shown))) && (
        <span id={`${id}-hint`} className="text-right text-xs text-muted-foreground">{format(shown)}</span>
      )}
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
  return (
    <FieldRow htmlFor={id} label={label} title={title}>
      <NumericInput
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        onCommit={onChange}
        disabled={disabled}
      />
    </FieldRow>
  );
}
