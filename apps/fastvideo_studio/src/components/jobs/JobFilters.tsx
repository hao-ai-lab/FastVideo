'use client';

import { Search, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface JobFiltersProps {
  model: string;
  prompt: string;
  onModelChange: (value: string) => void;
  onPromptChange: (value: string) => void;
  count: number;
  total: number;
}

export default function JobFilters({
  model,
  prompt,
  onModelChange,
  onPromptChange,
  count,
  total,
}: JobFiltersProps) {
  return (
    <div className="mb-5 flex flex-col gap-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
        <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-xs font-medium text-muted-foreground">
          Model name
          <Input
            value={model}
            onChange={(event) => onModelChange(event.target.value)}
            placeholder="Filter models…"
            className="h-10 font-normal text-foreground"
          />
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
        {(model || prompt) && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-10"
            onClick={() => {
              onModelChange('');
              onPromptChange('');
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
