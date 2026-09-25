'use client';

import * as React from 'react';
import { ChevronDown } from 'lucide-react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

import CreateJobModal from '@/components/jobs/CreateJobModal';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { WORKLOAD_OPTIONS } from '@/lib/jobConfig';
import { parseJobJson } from '@/lib/jobJson';
import type { JobLike } from '@/lib/jobToFields';
import type { JobType } from '@/lib/types';
import { triggerRefresh } from '@/stores/jobsRefresh';

interface CreateJobButtonProps {
  jobType: JobType;
}

export default function CreateJobButton({ jobType }: CreateJobButtonProps) {
  const options = WORKLOAD_OPTIONS[jobType] ?? [];

  const [modalOpen, setModalOpen] = React.useState(false);
  const [workloadType, setWorkloadType] = React.useState(
    options[0]?.type ?? 't2v',
  );

  // An imported job file pre-fills the form; a normal open starts clean.
  const [prefillJob, setPrefillJob] = React.useState<JobLike | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  async function handleImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ''; // allow re-picking the same file
    if (!file) return;
    const result = parseJobJson(await file.text());
    if (!result.ok) {
      toast.error(`Couldn't import ${file.name}: ${result.error}`);
      return;
    }
    openModal(result.workloadType, result.job);
  }

  function openModal(type: string, prefill: JobLike | null = null) {
    setWorkloadType(type);
    setPrefillJob(prefill);
    // Opening the Dialog synchronously inside the DropdownMenu.Item's
    // onSelect races the two Radix layers' pointer-events body lock/unlock:
    // the dropdown's own close cleanup can lose that race, leaving <body>
    // stuck with `pointer-events: none` after this dialog later closes --
    // silently blocking every click, including this same button. Deferring
    // to the next frame lets the dropdown finish closing first.
    requestAnimationFrame(() => setModalOpen(true));
  }

  function handleSuccess() {
    triggerRefresh();
    setModalOpen(false);
  }

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button type="button" className="gap-1.5">
            Create Job
            <ChevronDown className="size-3.5 opacity-85" aria-hidden />
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            align="end"
            sideOffset={4}
            collisionPadding={8}
            className="z-[200] min-w-48 overflow-hidden rounded-lg border border-border bg-popover py-1 text-popover-foreground shadow-lg"
          >
            {options.map((opt) => (
              <DropdownMenu.Item
                key={opt.type}
                onSelect={() => openModal(opt.type)}
                className="flex min-h-11 cursor-pointer select-none flex-col justify-center px-4 py-2 text-left text-sm font-medium outline-none data-[highlighted]:bg-secondary"
              >
                {opt.label}
                <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                  {opt.desc}
                </span>
              </DropdownMenu.Item>
            ))}
            {jobType === 'inference' && (
              <>
                <DropdownMenu.Separator className="my-1 h-px bg-border" />
                <DropdownMenu.Item
                  onSelect={() => fileInputRef.current?.click()}
                  className="flex min-h-11 cursor-pointer select-none flex-col justify-center px-4 py-2 text-left text-sm font-medium outline-none data-[highlighted]:bg-secondary"
                >
                  Import job JSON…
                  <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                    Fill the form from a job file
                  </span>
                </DropdownMenu.Item>
              </>
            )}
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        aria-label="Import job JSON file"
        tabIndex={-1}
        className="sr-only"
        onChange={handleImportFile}
      />
      <CreateJobModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSuccess={handleSuccess}
        jobType={jobType}
        workloadType={workloadType}
        prefillJob={prefillJob}
      />
    </>
  );
}
