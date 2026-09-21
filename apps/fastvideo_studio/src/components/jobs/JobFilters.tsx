'use client';

import { ChevronDown, Search, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { NativeSelect } from '@/components/ui/native-select';
import type { Model } from '@/lib/api';

interface JobFiltersProps {
  model: string;
  models: Model[];
  prompt: string;
  status?: string;
  onModelChange: (value: string) => void;
  onPromptChange: (value: string) => void;
  onStatusChange?: (value: string) => void;
  count: number;
  total: number;
}

export default function JobFilters({
  model,
  models,
  prompt,
  status = '',
  onModelChange,
  onPromptChange,
  onStatusChange,
  count,
  total,
}: JobFiltersProps) {
  return (
    <div className="mb-5 flex flex-col gap-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
        {onStatusChange && (
          <label className="flex flex-col gap-1.5 text-xs font-medium text-muted-foreground sm:w-40 sm:shrink-0">
            Status
            <span className="relative">
              <NativeSelect
                aria-label="Status"
                value={status}
                onChange={(event) => onStatusChange(event.target.value)}
                className="h-10 pr-9 font-normal"
              >
                <option value="">All statuses</option>
                <option value="pending">Pending</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="stopped">Stopped</option>
              </NativeSelect>
              <ChevronDown className="pointer-events-none absolute right-3 top-3 size-4" aria-hidden />
            </span>
          </label>
        )}
        <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-xs font-medium text-muted-foreground">
          Model name
          <span className="relative">
            <NativeSelect
              aria-label="Model name"
              value={model}
              onChange={(event) => onModelChange(event.target.value)}
              className="h-10 truncate pr-9 font-normal"
            >
              <option value="">All models</option>
              {models.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label === option.id ? option.id : `${option.label} (${option.id})`}
                </option>
              ))}
            </NativeSelect>
            <ChevronDown className="pointer-events-none absolute right-3 top-3 size-4" aria-hidden />
          </span>
        </label>
        <label className="flex min-w-0 flex-[1.5] flex-col gap-1.5 text-xs font-medium text-muted-foreground">
          Prompt contains
          <span className="relative">
            <Search className="pointer-events-none absolute left-3 top-3 size-4" aria-hidden />
            <Input
              value={prompt}
              onChange={(event) => onPromptChange(event.target.value)}
              placeholder="Search prompt text…"
              className="h-10 pl-9 font-normal text-foreground"
            />
          </span>
        </label>
        {(model || prompt || status) && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-10"
            onClick={() => {
              onModelChange('');
              onPromptChange('');
              onStatusChange?.('');
            }}
          >
            <X className="mr-1 size-3.5" aria-hidden />
            Clear
          </Button>
        )}
      </div>
      <p role="status" className="text-xs text-muted-foreground">
        {count === total ? `${total} jobs` : `${count} of ${total} jobs`}
      </p>
    </div>
  );
}
