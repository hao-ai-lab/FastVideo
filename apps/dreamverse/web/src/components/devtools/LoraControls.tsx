'use client';

import React, { useEffect, useRef, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';

const STYLE_LABELS: Record<string, string> = {
  none: 'None',
  pixar: 'Pixar Toon',
  transition: 'Transition',
};

export default function LoraControls() {
  const [styleKeys, setStyleKeys] = useState<string[]>([]);
  const [labels, setLabels] = useState<Record<string, string>>(STYLE_LABELS);
  const [strength, setStrength] = useState(0.8);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [intensity, setIntensity] = useState<Record<string, number>>({});
  const [status, setStatus] = useState('idle');

  const strengthRef = useRef(0.8);
  const enabledRef = useRef<Record<string, boolean>>({});
  const intensityRef = useRef<Record<string, number>>({});
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    fetch('/lora/options')
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data && Array.isArray(data.styles)) {
          const keys = data.styles.filter((s: string) => s !== 'none');
          setStyleKeys(keys);
          const initIntensity: Record<string, number> = {};
          keys.forEach((k: string) => {
            initIntensity[k] = 1.0;
          });
          setIntensity(initIntensity);
          intensityRef.current = initIntensity;
        }
        if (data && data.labels && typeof data.labels === 'object') {
          setLabels((prev) => ({ ...prev, ...data.labels }));
        }
      })
      .catch(() => setStatus('options unavailable'));

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const applyNow = () => {
    const reqId = ++requestIdRef.current;
    const stylesPayload: Record<string, number> = {};
    for (const key of Object.keys(enabledRef.current)) {
      if (enabledRef.current[key]) {
        stylesPayload[key] = intensityRef.current[key] ?? 1.0;
      }
    }
    setStatus('applying…');
    fetch('/lora', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strength: strengthRef.current, styles: stylesPayload }),
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return response.json();
      })
      .then((data) => {
        if (reqId !== requestIdRef.current) return;
        const active = Object.keys(data?.styles ?? {});
        const desc = active.length
          ? active.map((k) => `${labels[k] ?? k}@${data.styles[k]}`).join(' + ')
          : 'no style';
        setStatus(`applied OmniNFT@${data.strength} + ${desc}`);
      })
      .catch((error) => {
        if (reqId !== requestIdRef.current) return;
        setStatus(`error: ${String(error).slice(0, 120)}`);
      });
  };

  const scheduleApply = () => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(applyNow, 250);
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
            OmniNFT strength + stackable style adapters, each with its own intensity (live).
          </span>
        </div>
        <Badge variant="secondary">{status}</Badge>
      </summary>
      <div className="space-y-5 border-t border-border px-5 py-4">
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
              strengthRef.current = next;
              scheduleApply();
            }}
            className="w-full accent-sky-400"
          />
        </div>

        {styleKeys.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No style adapters for the active model.
          </p>
        ) : (
          styleKeys.map((key) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="flex cursor-pointer items-center gap-2 text-sm font-medium text-foreground">
                  <input
                    type="checkbox"
                    checked={!!enabled[key]}
                    onChange={(event) => {
                      const next = { ...enabledRef.current, [key]: event.target.checked };
                      enabledRef.current = next;
                      setEnabled(next);
                      scheduleApply();
                    }}
                    className="accent-sky-400"
                  />
                  {labels[key] ?? STYLE_LABELS[key] ?? key}
                </label>
                <span className="text-xs text-muted-foreground">
                  {(intensity[key] ?? 1).toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={intensity[key] ?? 1}
                disabled={!enabled[key]}
                onChange={(event) => {
                  const next = { ...intensityRef.current, [key]: Number(event.target.value) };
                  intensityRef.current = next;
                  setIntensity(next);
                  scheduleApply();
                }}
                className="w-full accent-sky-400 disabled:opacity-40"
              />
            </div>
          ))
        )}
      </div>
    </details>
  );
}
