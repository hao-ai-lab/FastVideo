'use client';

import CreateJobButton from '@/components/jobs/CreateJobButton';
import { HeaderActions } from '@/components/shell/HeaderActionsContext';
import JobQueue from '@/components/jobs/JobQueue';
import type { JobType } from '@/lib/types';

// 'lora' is a backend job_type (LoRA finetunes) that isn't part of the
// JobType union; the finetuning queue lists both alongside full finetunes.
const FINETUNING_LIST = ['finetuning', 'lora'] as JobType[];

export default function FinetuningPage() {
  return (
    <>
      <HeaderActions>
        <CreateJobButton jobType="finetuning" />
      </HeaderActions>
      <JobQueue jobType="finetuning" jobTypesForList={FINETUNING_LIST} />
    </>
  );
}
