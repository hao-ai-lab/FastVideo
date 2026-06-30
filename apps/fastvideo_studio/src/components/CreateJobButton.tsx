'use client';

import * as React from 'react';
import { ChevronDown } from 'lucide-react';

import CreateJobModal from '@/components/CreateJobModal';
import { Button } from '@/components/ui/button';
import { WORKLOAD_OPTIONS } from '@/lib/jobConfig';
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

  function openModal(type: string) {
    setWorkloadType(type);
    setModalOpen(true);
  }

  function handleSuccess() {
    triggerRefresh();
    setModalOpen(false);
  }

  return (
    <>
      <div className="group relative inline-block">
        <Button type="button" className="gap-1.5">
          Create Job
          <ChevronDown className="size-3.5 opacity-85" aria-hidden />
        </Button>
        <div
          role="menu"
          className="invisible absolute right-0 top-full z-[200] mt-1 min-w-full -translate-y-1 rounded-lg border border-border bg-popover py-1 opacity-0 shadow-lg transition-all duration-150 group-hover:visible group-hover:translate-y-0 group-hover:opacity-100"
        >
          {options.map((opt) => (
            <button
              key={opt.type}
              type="button"
              role="menuitem"
              onClick={() => openModal(opt.type)}
              className="block w-full whitespace-nowrap px-4 py-2 text-left text-sm font-medium text-popover-foreground transition-colors hover:bg-secondary"
            >
              {opt.label}
              <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                {opt.desc}
              </span>
            </button>
          ))}
        </div>
      </div>
      <CreateJobModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSuccess={handleSuccess}
        jobType={jobType}
        workloadType={workloadType}
      />
    </>
  );
}
