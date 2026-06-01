'use client';

import React, { useEffect, useRef, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { NativeSelect } from '@/components/ui/native-select';

const STYLE_LABELS: Record<string, string> = {
  none: 'None',
  pixar: 'Pixar',
  transition: 'Transition',
};

export default function LoraControls() {
  const [styles, setStyles] = useState<string[]>(['none']);
  const [strength, setStrength] = useState(0.8);
  const [style, setStyle] = useState('none');
  const [status, setStatus] = useState('idle');
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    fetch('/lora/options')
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data && Array.isArray(data.styles)) {
          setStyles(data.styles);
        }
      })
      .catch(() => setStatus('options unavailable'));

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const applyLora = (nextStrength: number, nextStyle: string) => {
    setStatus('applying…');
    fetch('/lora', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strength: nextStrength, style: nextStyle }),
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return response.json();
      })
      .then((data) => {
        const trigger = data?.trigger ? ` · trigger ${data.trigger}` : '';
        setStatus(`applied omninft@${data.strength} + ${data.style}${trigger}`);
      })
      .catch((error) => setStatus(`error: ${String(error).slice(0, 120)}`));
  };

  const scheduleApply = (nextStrength: number, nextStyle: string) => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => applyLora(nextStrength, nextStyle), 250);
  };

  return (
    <details
      className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm xl:col-span-2"
      open
    >
      <summary className="flex cursor-pointer list-none items-start justify-between gap-4 px-5 py-4">
        <div className="space-y-1">
          <span className="block text-lg font-semibold text-foreground">
            LoRA stack
          </span>
          <span className="block text-sm leading-6 text-muted-foreground">
            OmniNFT strength + optional style adapter (live).
          </span>
        </div>
        <Badge variant="secondary">{status}</Badge>
      </summary>
      <div className="grid gap-5 border-t border-border px-5 py-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="lora-strength">
            OmniNFT strength · {strength.toFixed(2)}
          </Label>
          <input
            id="lora-strength"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={strength}
            onChange={(event) => {
              const next = Number(event.target.value);
              setStrength(next);
              scheduleApply(next, style);
            }}
            className="w-full accent-sky-400"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="lora-style">Style</Label>
          <NativeSelect
            id="lora-style"
            value={style}
            onChange={(event) => {
              const next = event.target.value;
              setStyle(next);
              scheduleApply(strength, next);
            }}
          >
            {styles.map((option) => (
              <option key={option} value={option}>
                {STYLE_LABELS[option] ?? option}
              </option>
            ))}
          </NativeSelect>
        </div>
      </div>
    </details>
  );
}
