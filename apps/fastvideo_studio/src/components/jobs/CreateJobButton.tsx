'use client';

import * as React from 'react';
import { ChevronDown } from 'lucide-react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

import CreateJobModal from '@/components/jobs/CreateJobModal';
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
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
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
